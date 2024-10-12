// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.
#include "config.h"
#include "macro.h"
#include "kernel.h"
#include "models.h"
#include <stdio.h>
#include <cooperative_groups.h>
#include "maths.h"
#include <float.h>

namespace cg = cooperative_groups;

/** Initialize cells */
static __device__ void initializeCell(uint32_t usedIndex, GridDefinition* gridDefinition, Model* model, Global* global);

/** Calculate gaussian probability at state x given mean and covariance (used for first measurement) */
static __device__ double gaussProbability(int32_t* key, GridDefinition* gridDefinition, Measurement* measurements);

/** Calculate gaussian probability at state x given mean and covariance (used for update measurement) */
static __device__ double gaussProbability(double* y, Measurement* measurement);

/** Initialize advection values */
static __device__ void initializeAdv(GridDefinition* gridDefinition, Model* model, Cell* cell);

/** Initialize ik nodes */
static __device__ void initializeIkNodes(Grid* grid, Cell* cell, uint32_t usedIndex);

/** Update ik nodes */
static __device__ void updateIkNodes(int offset, int iterations, Grid* grid);

/** Update ik nodes for one cell */
static __device__ void updateIkNodesCell(Cell* cell, Grid* grid);

/** Initialize boundary value */
static __device__ void initializeBoundary(Cell* cell, Model* model);

/** Initialize Grid boundary */
static __device__ void initializeGridBoundary(int offset, int iterations, double* localArray, GridDefinition* gridDefinition, Global* global);

/** Normalize probability distribution */
static __device__ void normalizeDistribution(int offset, int iterations, double* localArray, double* globalArray, Grid* grid);

/** Compute grid bounds */
static __device__ void gridBounds(double* output, double* localArray, double* globalArray, double boundaryValue, double(*fn)(double, double) );

/** Compute step dt */
static __device__ void checkCflCondition(int offset, int iterations, double* localArray, GridDefinition* gridDefinition, Global* global);

/** Grow grid */
static __device__ void growGrid(int offset, int iterations, GridDefinition* gridDefinition, Grid* grid, Model* model);

/** Grow grid from one cell */
static __device__ void growGridFromCell(Cell* cell, GridDefinition* gridDefinition, Grid* grid, Model* model);

/** Grow grid from one cell in one dimension and direction */
static __device__ void growGridDireccional(Cell* cell, enum Direction direction, GridDefinition* gridDefinition, Grid* grid, Model* model);

/** Grow grid from one cell in one dimension and direction */
static __device__ void growGridEdges(Cell* cell, enum Direction direction, enum Direction directionJ, GridDefinition* gridDefinition, Grid* grid, Model* model);

/** Create new cell in the grid */
static __device__ void createCell(int32_t* state, GridDefinition* gridDefinition, Grid* grid, Model* model);

/** Prune grid */
static __device__ void pruneGrid(int offset, int iterations, GridDefinition* gridDefinition, Grid* grid, Global* global);

/** Mark the cell for deletion if is negligible */
static __device__ void markNegligibleCell(uint32_t usedIndex, GridDefinition* gridDefinition, Grid* grid);

/** Check id exists significant flux from cell and its edges */
static __device__ bool fluxFrom(Cell* cell, enum Direction direction, int dimension, GridDefinition* gridDefinition, Grid* grid);

/** Perform the Godunov scheme on the discretized PDF*/
static __device__ void godunovMethod(int offset, int iterations, GridDefinition* gridDefinition, Grid* grid);

/** Compute the donor cell upwind value for each grid cell */
static __device__ void updateDcu(Cell* cell, Grid* grid, GridDefinition* gridDef);

/** Compute the corner transport upwind values in each direction */
static __device__ void updateCtu(Cell* cell, Grid* grid, GridDefinition* gridDef);

/** Compute flux from the left */
static __device__ double uPlus(double v);

/** Compute flux from the right */
static __device__ double uMinus(double v);

/** MC flux limiter */
static __device__ double fluxLimiter(double th);

/** Update probability */
static __device__ void updateProbability(int offset, int iterations, GridDefinition* gridDefinition, Grid* grid);

/** Update probability for one cell */
static __device__ void updateProbabilityCell(Cell* cell, Grid* grid, GridDefinition* gridDef);

/** Apply measurement */
static __device__ void applyMeasurement(int offset, int iterations, Measurement* measurement, GridDefinition* gridDefinition, Grid* grid, Model* model);

/** Apply measurement for one cell */
static __device__ void applyMeasurementCell(Cell* cell, Measurement* measurement, GridDefinition* gridDefinition, Grid* grid, Model* model);

/** Get offset index to iterate cells */
static __device__ int getOffset();

/** Get index to iterate cells */
static __device__ uint32_t getIndex(int offset, int iteration);


/** --- Device global memory allocations --- */

/**
 * @brief Alloc global device memory 
 * 
 * @param global global struct pointer
 * @param blocks number of concurrent blocks
 * @param iterations number of cell processed per thread
 */
void allocGlobalDevice(Global* global, int blocks, int iterations){
    HANDLE_CUDA(cudaMalloc(&global->reductionArray, blocks * sizeof(double)));
    HANDLE_CUDA(cudaMalloc(&global->blockSums, blocks * iterations * 2 * sizeof(uint32_t)));
    HANDLE_CUDA(cudaMalloc(&global->blockSumsOut, sizeof(int32_t)));
}

/**
 * @brief Free global device memory 
 *  
 * @param global global struct pointer
 */
void freeGlobalDevice(Global* global){
    HANDLE_CUDA(cudaFree(global->reductionArray)); 
    HANDLE_CUDA(cudaFree(global->blockSums)); 
    HANDLE_CUDA(cudaFree(global->blockSumsOut));
}

/**
 * @brief Required shared memory
 * @return the required shared memory by the kernel
 */
size_t requiredSharedMemory(){ // TODO check numbers with the profiler
    size_t sharedMemoryForReduction = sizeof(double) * THREADS_PER_BLOCK;
    size_t sharedMemoryForScan = sizeof(uint32_t) * THREADS_PER_BLOCK * CELLS_PER_THREAD * 2;
    log("Shared memory for reduction %lu\n", sharedMemoryForReduction);
    log("Shared memory for scan %lu\n", sharedMemoryForScan);    
    return sharedMemoryForReduction + sharedMemoryForScan;    
}

/** Main kernel */
    
/** Get offset index to iterate cells */
static __device__ int getOffset(){        
    return threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;    
}

/** Get used index to iterate cells */
static __device__ uint32_t getIndex(int offset, int iteration){        
    return (uint32_t)(offset + iteration * THREADS_PER_BLOCK * BLOCKS);    
}
/** 
 * @brief Initialization kernel function 
 * 
 * @param iterations number of cells that should process the same thread
 * @param model the model
 * @param global global memory data
 */
__global__ void gbeesKernel(int iterations, Model model, Global global){
    // grid synchronization
    cg::grid_group g = cg::this_grid(); 
    
    // shared memory for reduction processes
    __shared__ double localArray[THREADS_PER_BLOCK]; // TODO optimize stride in reduction processes  
    
    // get used list offset index
    int offset = getOffset();     
        
    // initialize cells
    for(int iter=0;iter<iterations;iter++){        
        int usedIndex = getIndex(offset, iter); // index in the used list               
        initializeCell(usedIndex, global.gridDefinition, &model, &global); // initialize cell
    }    
    
    // set grid maximum and minimum bounds
    if(model.useBounds){ // TODO test use bounds
        initializeGridBoundary(offset, iterations, localArray, global.gridDefinition, &global);   
        
        /* if(offset == 0){
            printf("Bounds min %e\n", global.gridDefinition->lo_bound);
            printf("Bounds max %e\n", global.gridDefinition->hi_bound);
        } */
    }
    
    // normalize distribution
    normalizeDistribution(offset, iterations, localArray, global.reductionArray, global.grid);
   
    
    // select first measurement
    Measurement* measurement = &global.measurements[0];
        
    double tt = 0.0;
    
    // for each measurement
    for(int nm=0; nm<model.numMeasurements; nm++){
        int stepCount = 1; // step count
        
        if(model.performOutput){
            LOG("Timestep: %d-%d, Sim. time: %f", nm, stepCount, tt);
            LOG(" TU, Used Cells: %d/%d\n", global.grid->usedSize, global.grid->size);                
        }
        
        // propagate probability distribution until the next measurement
        double mt = 0.0; // time propagated from the last measurement            
        double rt;        
        // slight variation w.r.t to original implementation as record time is recalculated for each measurement
        double recordTime = measurement->T / (model.numDistRecorded-1);
        
        while(fabs(mt - measurement->T) > TOL) {
            rt = 0.0;
           
            while(rt < recordTime) { // time between PDF recordings                  
                g.sync();   

                growGrid(offset, iterations, global.gridDefinition, global.grid, &model);            
                
                if(global.grid->overflow){ // check grid overflow
                    LOG("Grid capacity exceeded\n");
                    return;
                }
                
                updateIkNodes(offset, iterations, global.grid);                      
                checkCflCondition(offset, iterations, localArray, global.gridDefinition, &global);             
                                
                if(threadIdx.x == 0 && blockIdx.x == 0){
                    global.gridDefinition->dt = fmin(global.gridDefinition->dt, recordTime - rt);
                }
                g.sync();
                
                rt += global.gridDefinition->dt;
                godunovMethod(offset, iterations, global.gridDefinition, global.grid);
                g.sync();            
                updateProbability(offset, iterations, global.gridDefinition, global.grid);                
                normalizeDistribution(offset, iterations, localArray, global.reductionArray, global.grid);
                
                //LOG("step duration %f, active cells %d\n", global.gridDefinition->dt, global.grid->usedSize);

                if (stepCount % model.deletePeriodSteps == 0) { // deletion procedure                    
                    //g.sync(); LOG("## pre prune active cells %d\n", global.grid->usedSize); g.sync();
                    pruneGrid(offset, iterations, global.gridDefinition, global.grid, &global);
                    //g.sync();LOG("## post prune active cells %d\n", global.grid->usedSize); g.sync();
                    updateIkNodes(offset, iterations, global.grid);    
                    normalizeDistribution(offset, iterations, localArray, global.reductionArray, global.grid);
                }
            
                if (model.performOutput && (stepCount % model.outputPeriodSteps == 0)) { // print size to terminal
                    LOG("Timestep: %d-%d, Sim. time: %f", nm, stepCount, tt + mt + rt);
                    LOG(" TU, Used Cells: %d/%d\n", global.grid->usedSize, global.grid->size); 
                }
                
                stepCount++;                                            
            } // while(rt < recordTime)

            if (((stepCount-1) % model.outputPeriodSteps != 0) && model.performOutput){ // print size to terminal  
                LOG("Timestep: %d-%d, Sim. time: %f", nm, stepCount-1, tt + mt + rt);
                LOG(" TU, Used Cells: %d/%d\n", global.grid->usedSize, global.grid->size);
            }
            
            if(model.performRecord){ // record PDF
                LOG("Record PDF (not implemented)\n");
                // TODO implement snapshoots
            }            
            mt += rt;
        }
    
        tt += mt;
        // perform Bayesian update for the next measurement
        if(model.performMeasure && nm < model.numMeasurements -1){
            if(model.performOutput){
                LOG("\nPERFORMING BAYESIAN UPDATE AT: %f TU...\n\n", tt);
            }
            
            // select next measurement
            measurement = &global.measurements[nm+1];            
            
            applyMeasurement(offset, iterations, measurement, global.gridDefinition, global.grid, &model);
            normalizeDistribution(offset, iterations, localArray, global.reductionArray, global.grid);
            pruneGrid(offset, iterations, global.gridDefinition, global.grid, &global);
            updateIkNodes(offset, iterations, global.grid);    
            normalizeDistribution(offset, iterations, localArray, global.reductionArray, global.grid);
        }        
    }
    
    LOG("Time marching complete.\n");
}

/** Initialize cells */
static __device__ void initializeCell(uint32_t usedIndex, GridDefinition* gridDefinition, Model* model, Global* global){
    // intialize cells    
    if(usedIndex < global->grid->usedSize){    
        double prob = 0.0;
        Cell* cell = NULL;
    
        // used list entry
        UsedListEntry* usedListEntry = global->grid->usedList + usedIndex;
        
        // obtain key (state coordinates)
        uint32_t hashtableIndex = usedListEntry->hashTableIndex;
        int32_t* key = global->grid->table[hashtableIndex].key;
        
        // compute initial probability    
        prob = gaussProbability(key, gridDefinition, global->measurements);
        
        // update cell          
        cell = getCell(usedIndex, global->grid);
        
        // compute state
        for(int i=0;i<DIM;i++){
            cell->state[i] = key[i]; // state coordinates
            cell->x[i] = gridDefinition->dx[i] * key[i] + gridDefinition->center[i]; // state value
        }
        
        cell->prob = prob; 
        initializeAdv(gridDefinition, model, cell);
        initializeIkNodes(global->grid, cell, usedIndex);    
        
        // initialize bounday value
        if(model->useBounds){
            initializeBoundary(cell, model);
        }
    }    
}

/** Calculate gaussian probability at state x given mean and covariance (used for first measurement) */
static __device__ double gaussProbability(int32_t* key, GridDefinition* gridDefinition, Measurement* measurement){    
    double mInvX[DIM];
    double diff[DIM];
    
    for(int i=0;i<DIM;i++){
        diff[i] = key[i] * gridDefinition->dx[i];
    }
    multiplyMatrixVector( (double*)measurement->covInv, diff, mInvX, DIM);
    double dotProduct = computeDotProduct(diff, mInvX, DIM);
    return exp(-0.5 * dotProduct);
}

/** Calculate gaussian probability at state x given mean and covariance (used for update measurement) */
static __device__ double gaussProbability(double* y, Measurement* measurement){    
    double mInvX[DIM];
    double diff[DIM];
    
    for(int i=0;i<measurement->dim;i++){
        diff[i] = y[i] - measurement->mean[i];
    }
    
    multiplyMatrixVector( (double*)measurement->covInv, diff, mInvX, measurement->dim);
    double dotProduct = computeDotProduct(diff, mInvX, measurement->dim);
    return exp(-0.5 * dotProduct);
}

/** Initialize advection values */
static __device__ void initializeAdv(GridDefinition* gridDefinition, Model* model, Cell* cell){        
    double xk[DIM];
    (*model->callbacks->f)(xk, cell->x, gridDefinition->dx); 

    double sum = 0;
    for(int i = 0; i < DIM; i++){
        cell->v[i] = xk[i];
        sum += fabs(cell->v[i]) / gridDefinition->dx[i];
    }      
    cell->cfl_dt = 1.0/sum;        
}

/**
 * Initialize ik nodes 
 * This function depends on an specific order to fill the usedList ( filled in function initializeHashtable() ).
 */
static __device__ void initializeIkNodes(Grid* grid, Cell* cell, uint32_t usedIndex){        
    uint32_t offset = 1;
    for(int i=DIM-1; ;i--){        
        // if is not the first cell in the dimension i        
        if(cell->state[i] > -(int)grid->initialExtent[i]){
            uint32_t iIndex = usedIndex - offset;
            cell->iNodes[i] = iIndex;
        } else {            
            cell->iNodes[i] = NULL_REFERENCE;
        }
        
        // if is not the last cell in the dimension i        
        if(cell->state[i] < (int)grid->initialExtent[i]){
            uint32_t kIndex = usedIndex + offset;        
            cell->kNodes[i] = kIndex;
        }  else {            
            cell->kNodes[i] = NULL_REFERENCE;
        }
        
        if(i<=0) break;
        offset *= grid->initialExtent[i] * 2 + 1;
    }        
}

/** Update ik nodes */
static __device__ void updateIkNodes(int offset, int iterations, Grid* grid){
    for(int iter=0; iter<iterations; iter++){      
        uint32_t usedIndex = getIndex(offset, iter);
        Cell* cell = getCell(usedIndex, grid);                
        if(cell != NULL) updateIkNodesCell(cell, grid);
    }
}

/** Update ik nodes for one cell */
static __device__ void updateIkNodesCell(Cell* cell, Grid* grid){
    int32_t state[DIM];    
    
    copyKey(cell->state, state);
    
    for(int dim=0; dim<DIM; dim++){
        // node i        
        state[dim] -= 1; // to reach -1
        cell->iNodes[dim] = findCell(state, grid); 
            
        // node k        
        state[dim] += 2; // to reach +1
        cell->kNodes[dim] = findCell(state, grid);        
        
        state[dim] -= 1; // to return to original
    }        
}

/** Initialize boundary value */
static __device__ void initializeBoundary(Cell* cell, Model* model){
    double j = (*model->callbacks->j)(cell->x);
    cell->bound_val = j;
}

static __device__ void initializeGridBoundary(int offset, int iterations, double* localArray, GridDefinition* gridDefinition, Global* global){
    double boundaryValue = -DBL_MAX;    
    for(int iter=0; iter<iterations; iter++){        
        // index in the used list
        uint32_t usedIndex = getIndex(offset, iter);   
        Cell* cell = getCell(usedIndex, global->grid);        
        if(cell != NULL && cell->bound_val > boundaryValue) boundaryValue = cell->bound_val;
    }
    gridBounds(&gridDefinition->hi_bound, localArray, global->reductionArray, boundaryValue, fmax);
    
    boundaryValue = DBL_MAX;   
    for(int iter=0;iter<iterations;iter++){        
        // index in the used list
        uint32_t usedIndex = getIndex(offset, iter);  
        Cell* cell = getCell(usedIndex, global->grid);        
        if(cell != NULL && cell->bound_val < boundaryValue) boundaryValue = cell->bound_val;
    }
    gridBounds(&gridDefinition->lo_bound, localArray, global->reductionArray, boundaryValue, fmin);
}

/** Compute step dt */
static __device__ void checkCflCondition(int offset, int iterations, double* localArray, GridDefinition* gridDefinition, Global* global){
    double minDt = DBL_MAX;
    for(int iter=0; iter<iterations; iter++){        
        // index in the used list
        uint32_t usedIndex = getIndex(offset, iter);   
        Cell* cell = getCell(usedIndex, global->grid);        
        if(cell != NULL && cell->cfl_dt < minDt) minDt = cell->cfl_dt;
    }    
    gridBounds(&gridDefinition->dt, localArray, global->reductionArray, minDt, fmin);    
}

/** Normalize probability distribution */
static __device__ void normalizeDistribution(int offset, int iterations, double* localArray, double* globalArray, Grid* grid){        
    // grid synchronization
    cg::grid_group g = cg::this_grid();      
   
    // store the sum of the cells probability for all the iterations at the local reduction array
    localArray[threadIdx.x] = 0.0;
    for(int iter=0;iter<iterations;iter++){
        uint32_t usedIndex = getIndex(offset, iter);           
        Cell* cell = getCell(usedIndex, grid); 
        if(cell != NULL) localArray[threadIdx.x] += cell->prob;        
    }
    
    __syncthreads();
    
    // reduction process in shared memory
    for(int s=1;s<blockDim.x;s*=2){
        int indexDst = 2 * s * threadIdx.x;
        int indexSrc = indexDst + s;
        if(indexSrc < blockDim.x){
            localArray[indexDst] += localArray[indexSrc];                        
        }
        __syncthreads();
    }
         
    if(threadIdx.x == 0){        
        // store total sum to global array
        globalArray[blockIdx.x] = localArray[0];       
    }
    
    g.sync();        
     
    // reduction process in global memory
    for(int s=1;s<gridDim.x;s*=2){
        if(threadIdx.x == 0){       
            int indexDst = 2 * s * blockIdx.x;
            int indexSrc = indexDst + s;
            if(indexSrc < gridDim.x){
                globalArray[indexDst] += globalArray[indexSrc];            
            }
        }
        g.sync();
    }                 
       
    // at the end, the sum of the probability its at globalArray[0]    
   //LOG("prob sum %1.14e\n", globalArray[0]);
    
    // update the probability of the cells
    for(int iter=0;iter<iterations;iter++){
        uint32_t usedIndex = getIndex(offset, iter);   
        Cell* cell = getCell(usedIndex, grid);         
        if(cell != NULL) cell->prob /= globalArray[0];        
    }
    
    g.sync();
}              

/** Set the grid definition bounds with the max and min boundary values of the initial grid cells */
static __device__ void gridBounds(double* output, double* localArray, double* globalArray, double boundaryValue, double(*fn)(double, double) ){
    // grid synchronization
    cg::grid_group g = cg::this_grid();      
    
    // store cell bounday value in the reduction array
    localArray[threadIdx.x] = boundaryValue;
    
    __syncthreads();
    
    // reduction process in shared memory
    for(int s=1;s<blockDim.x;s*=2){
        int indexDst = 2 * s * threadIdx.x;
        int indexSrc = indexDst + s;
        if(indexSrc < blockDim.x){
            localArray[indexDst] = fn(localArray[indexSrc], localArray[indexDst]);                        
        }
        __syncthreads();
    }
        
    if(threadIdx.x == 0){        
        // store total sum to global array
        globalArray[blockIdx.x] = localArray[0];       
    }
    g.sync();
        
    // reduction process in global memory
    for(int s=1;s<gridDim.x;s*=2){
        if(threadIdx.x == 0) {
            int indexDst = 2 * s * blockIdx.x;
            int indexSrc = indexDst + s;
            if(indexSrc < gridDim.x){
                globalArray[indexDst] = fn(globalArray[indexSrc], globalArray[indexDst]);            
            }            
        }
        g.sync();
    } 
    if(blockIdx.x == 0 && threadIdx.x == 0){
        *output = globalArray[0];        
    }
    g.sync();    
}

/** Grow grid */
static __device__ void growGrid(int offset, int iterations, GridDefinition* gridDefinition, Grid* grid, Model* model){ 
    for(int iter=0;iter<iterations;iter++){
        uint32_t usedIndex = getIndex(offset, iter); // index in the used list            
        Cell* cell = getCell(usedIndex, grid);
        growGridFromCell(cell, gridDefinition, grid, model);        
    }                
}

/** Grow grid from one cell */
static __device__ void growGridFromCell(Cell* cell, GridDefinition* gridDefinition, Grid* grid, Model* model){
    // grid synchronization
    cg::grid_group g = cg::this_grid();
    
    bool performGrow = cell != NULL && cell->prob >= gridDefinition->threshold;
    
    if(performGrow) {          
        growGridDireccional(cell,  FORWARD, gridDefinition, grid, model);                
    }    
    
    g.sync(); // synchronization needed to avoid create duplicated cells
    
    if(performGrow) {          
        growGridDireccional(cell,  BACKWARD, gridDefinition, grid, model);                
    }     
    
    g.sync(); // synchronization needed to avoid create duplicated cells
    
    if(performGrow) {         
        growGridEdges(cell,  FORWARD, FORWARD, gridDefinition, grid, model);                
    } 
    
    g.sync(); // synchronization needed to avoid create duplicated cells
    
    if(performGrow) { 
        growGridEdges(cell, FORWARD, BACKWARD, gridDefinition, grid, model);    
    }
    
    g.sync(); // synchronization needed to avoid create duplicated cells
    
    if(performGrow) { 
        growGridEdges(cell, BACKWARD, FORWARD, gridDefinition, grid, model);         
    } 
    
    g.sync(); // synchronization needed to avoid create duplicated cells
    
    if(performGrow) { 
        growGridEdges(cell, BACKWARD, BACKWARD, gridDefinition, grid, model);        
    } 
    
    g.sync(); // synchronization needed to avoid create duplicated cells
}

/** Grow grid from one cell in one dimension and direction */
static __device__ void growGridDireccional(Cell* cell, enum Direction direction, GridDefinition* gridDefinition, Grid* grid, Model* model){
        
    uint32_t nextFaceIndex = 0; // initialized to null reference
    int32_t state[DIM]; // state indexes for the new cells            
    
    for(int dimension=0;dimension<DIM;dimension++){
        if(direction == FORWARD) {
            if(cell->v[dimension] <= 0.0) continue; 
            nextFaceIndex = cell->kNodes[dimension];
        } else {
            if(cell->v[dimension] >= 0.0) continue; 
            nextFaceIndex = cell->iNodes[dimension]; 
        }
    
        // create next face if not exists
        if(nextFaceIndex == NULL_REFERENCE){
            // create new cell key[dimension] = cell->key[dimension]+direction
            copyKey(cell->state, state);
            state[dimension] += direction;
            createCell(state, gridDefinition, grid, model);
        }
    }    
}

/** Grow grid from one cell in one dimension and direction */
static __device__ void growGridEdges(Cell* cell, enum Direction direction, enum Direction directionJ, GridDefinition* gridDefinition, Grid* grid, Model* model){
        
    int32_t state[DIM]; // state indexes for the new cells
    
    for(int dimension=0;dimension<DIM;dimension++){
        if(cell->v[dimension] * direction <= 0.0 ) continue;
        
        // check edges
        for (int j = 0; j < DIM; j++){
            if(j != dimension){
                if(cell->v[j] * directionJ > 0.0) {                                
                    copyKey(cell->state, state);
                    state[dimension] += direction;
                    state[j] += directionJ;
                    createCell(state, gridDefinition, grid, model);
                } 
            }
        }   
    }        
}

/** Create new cell in the grid if not exists
 *  checks existence only with previous cells, not with other concurrent create cells
 */
static __device__ void createCell(int32_t* state, GridDefinition* gridDefinition, Grid* grid, Model* model){
    Cell cell;
    
    // TODO optional performance improvement, move initialization to insertCellConcurrent (or use a callback) and execute only if the cell not exits
    
    // compute state
    for(int i=0;i<DIM;i++){
        cell.state[i] = state[i];
        cell.x[i] = gridDefinition->dx[i] * state[i] + gridDefinition->center[i]; // state value
        cell.ctu[i] = 0.0;
    }
        
    cell.deleted = false;
    cell.prob = 0.0; 
    cell.dcu = 0.0;
    initializeAdv(gridDefinition, model, &cell);
    
    // insert cell
    insertCellConcurrent(&cell, grid);
}

/** Prune grid */
static __device__ void pruneGrid(int offset, int iterations, GridDefinition* gridDefinition, Grid* grid, Global* global){
    // TODO optimize stride of shared memory, and amout of shared memory used
    // shared memory for scan    
    __shared__ uint32_t buffers[THREADS_PER_BLOCK * CELLS_PER_THREAD * 2]; // shared memory for scan (double buffer)
    const int T = THREADS_PER_BLOCK; // number of threads
    const int TI = THREADS_PER_BLOCK * CELLS_PER_THREAD; // number of threads per number of iterations
    const int BI = BLOCKS * CELLS_PER_THREAD; // number of blocks per number of iterations
    
    // grid synchronization
    cg::grid_group g = cg::this_grid(); 
  
    for(int iter=0;iter<iterations;iter++){ 
        uint32_t usedIndex = getIndex(offset, iter); // index in the used list                    
        markNegligibleCell(usedIndex, gridDefinition, grid);                        
    }

    // load elements into shared memory for local scan (fill first buffer) 
    for(int iter=0; iter<iterations;iter++) {
        uint32_t usedIndex = getIndex(offset, iter); // index in the used list  
        Cell* cell = getCell(usedIndex, grid);
        buffers[threadIdx.x + iter * T] =
            (cell != NULL && !cell->deleted) ? 1: 0;     
    }
    
    __syncthreads();
    
    // scan process in shared memory
    int bufferOut = 0, bufferIn = 1;
    for(int offset=1; offset<blockDim.x; offset*=2) {   
        
        // swap double buffer indices
        bufferOut = 1 - bufferOut; 
        bufferIn = 1 - bufferOut;
        
        for(int iter=0; iter<iterations;iter++) {
            int indexIn = bufferIn * TI + threadIdx.x + iter * T;
            int indexOut = bufferOut * TI + threadIdx.x + iter * T;
            if (threadIdx.x >= offset)
                buffers[indexOut] = buffers[indexIn - offset] + buffers[indexIn];
            else
                buffers[indexOut] = buffers[indexIn];
        }        
    __syncthreads();
    }
    
    g.sync();  
    
    // store block sum in global memory    
    if(threadIdx.x == 0) {
        for(int iter=0; iter<iterations;iter++) {
            global->blockSums[blockIdx.x + iter * BLOCKS] = buffers[bufferOut * TI + (T-1) + iter * T];            
        }
    }        
    
    g.sync();
    
    int32_t blockSumsIn = 1;
    int32_t blockSumsOut = 0;  
    // scan process in global memory
    for(int offset=1; offset< BI ; offset*=2){                
        if(threadIdx.x == 0) {
            // swap double buffer indices            
            blockSumsOut = 1 - blockSumsOut; 
            blockSumsIn = 1 - blockSumsOut;
                
            for(int iter=0; iter<iterations;iter++){                
                int absoluteIndex = iter * BLOCKS  + blockIdx.x;
                int indexIn = blockSumsIn * BI + absoluteIndex;
                int indexOut = blockSumsOut * BI + absoluteIndex;
                if (absoluteIndex >= offset)
                    global->blockSums[indexOut] = global->blockSums[indexIn - offset] + global->blockSums[indexIn];                    
                else
                    global->blockSums[indexOut] = global->blockSums[indexIn];                                                   
            }
        }
        g.sync();
    } 
    
    if(threadIdx.x == 0 && blockIdx.x == 0){
        *global->blockSumsOut = blockSumsOut;
    }
    
    g.sync();
        
    // initialize double buffer used list
    for(int iter=0; iter<iterations;iter++) {
        uint32_t usedIndex = getIndex(offset, iter); // index in the used list  
        grid->usedListTemp[usedIndex].heapIndex = NULL_REFERENCE;
        grid->usedListTemp[usedIndex].hashTableIndex = NULL_REFERENCE;        
    }
    
    g.sync(); 
    
    // compact used list in usedListTemp and update free list
    for(int iter=0; iter<iterations;iter++) {
        int absoluteIndex = iter * BLOCKS  + blockIdx.x; // absolute index in the block sums
        uint32_t blockSum = (absoluteIndex > 0)? global->blockSums[*global->blockSumsOut * BI + (absoluteIndex-1) ] : 0; // get the sum of the previous block                
        uint32_t threadSum = buffers[bufferOut * TI + threadIdx.x + iter * T];
        uint32_t dstIndex = blockSum + threadSum - 1; // minus one because the scan in shared memory is inclusive
        uint32_t srcIndex = getIndex(offset, iter);        
        
        Cell* cell = getCell(srcIndex, grid);
                        
        if(cell != NULL){
            uint32_t hashtableIndex = grid->usedList[srcIndex].hashTableIndex;
            
            if(cell->deleted){                
                // deleted cell                
                uint32_t freeIndex = atomicAdd(&grid->freeSize, 1);  // check and reserve free list location
                grid->freeList[freeIndex] = cell - grid->heap;  // add heap index to the free list                
                grid->table[hashtableIndex].usedIndex = NULL_REFERENCE; // update hashtable used index
            }  else {
                // not deleted cell, compact in usedList               
                grid->usedListTemp[dstIndex].heapIndex = grid->usedList[srcIndex].heapIndex;
                grid->usedListTemp[dstIndex].hashTableIndex = hashtableIndex;    
                grid->table[hashtableIndex].usedIndex = dstIndex; // update hashtable used index
            }
        }        
    }     
    
    g.sync();
    
    // compute new used list size and switch buffers
    if(threadIdx.x == 0 && blockIdx.x == 0) {      
        // get new used list size from the last block scanned
        grid->usedSize = global->blockSums[*global->blockSumsOut * BI + BI-1];                    
        
        // switch buffers
        UsedListEntry* listPtr = grid->usedList;
        grid->usedList = grid->usedListTemp;
        grid->usedListTemp = listPtr; 
    }
    
    g.sync();
    
    // rehash, clean temp table
    for(int i=0; i<iterations; i++){ 
        uint32_t usedIndex = getIndex(offset, i); // index in the used list                    
        for(int k=0; k<HASH_TABLE_RATIO;k++){
            uint32_t hashtableIndex = usedIndex + k * grid->size;            
            grid->tableTemp[hashtableIndex].usedIndex = NULL_REFERENCE;
        }
    }
    
    g.sync();

    // rehash, copy active hash entries
    for(int i=0; i<iterations; i++){ 
        uint32_t usedIndex = getIndex(offset, i); // index in the used list                    
        for(int k=0; k<HASH_TABLE_RATIO;k++){            
            uint32_t hashtableIndex = usedIndex + k * grid->size;
            if(grid->table[hashtableIndex].usedIndex != NULL_REFERENCE){ // if active entry
                insertHashConcurrent(&grid->table[hashtableIndex], grid->tableTemp, grid);
            }
        }        
    }

    g.sync();  
    
    // switch hashtable buffers
    if(threadIdx.x == 0 && blockIdx.x == 0) {        
        // switch buffers
        HashTableEntry* tablePtr = grid->table;
        grid->table = grid->tableTemp;
        grid->tableTemp = tablePtr; 
    }
    
    g.sync();  
}

/** Mark the cell for deletion if is negligible */
static __device__ void markNegligibleCell(uint32_t usedIndex, GridDefinition* gridDefinition, Grid* grid){      
    // get cell
    Cell* cell = getCell(usedIndex, grid);
    if(cell == NULL) return;
    
    // check if its probability is negligible
    if(cell->prob >= gridDefinition->threshold) return;    
        
    for(int i=0;i<DIM;i++){
        // look backwards node
        Cell* iCell = getCell(cell->iNodes[i], grid);
        if( fluxFrom(iCell, FORWARD, i, gridDefinition, grid) ) return;
            
        // look forwards node
        Cell* kCell = getCell(cell->kNodes[i], grid);
        if( fluxFrom(kCell, BACKWARD, i, gridDefinition, grid) ) return;        
    }
    
    // mark for deletion
    cell->deleted = true;           
}

/** Check id exists significant flux from cell and its edges */
static __device__ bool fluxFrom(Cell* cell, enum Direction direction, int dimension, GridDefinition* gridDefinition, Grid* grid){
    if(cell == NULL) return false;
        
    // check flux from itself
    if(cell->prob >= gridDefinition->threshold && cell->v[dimension] * direction > 0.0) return true;
    
    // check flux from edges
    for (int j=0; j<DIM; j++){
        if(j != dimension){
            Cell* iCell = getCell(cell->iNodes[j], grid);
            if(iCell != NULL && iCell->v[dimension] * direction > 0.0 && iCell->v[j] > 0.0 && iCell->prob >= gridDefinition->threshold) return true;
            
            Cell* kCell = getCell(cell->kNodes[j], grid);
            if(kCell != NULL && kCell->v[dimension] * direction > 0.0 && kCell->v[j] < 0.0 && kCell->prob >= gridDefinition->threshold) return true;
        }
    }
    return false;
}

    
/** 
 * Perform the Godunov scheme on the discretized PDF. 
 * This function performs the Godunov scheme on the discretized PDF, which is 2nd-order accurate and total variation diminishing
 */
static __device__ void godunovMethod(int offset, int iterations, GridDefinition* gridDefinition, Grid* grid) {    
    // grid synchronization
    cg::grid_group g = cg::this_grid();    
     
    int usedIndex;
    Cell* cell;
    // initialize cells
    for(int iter=0;iter<iterations;iter++){        
        usedIndex = getIndex(offset, iter); // index in the used list
        cell = getCell(usedIndex, grid); 
        if(cell != NULL){
            updateDcu(cell, grid, gridDefinition);        
        }
    }
    
    g.sync();    
    
    for(int iter=0;iter<iterations;iter++){        
        usedIndex = getIndex(offset, iter); // index in the used list
        cell = getCell(usedIndex, grid); 
        if(cell != NULL){        
            updateCtu(cell, grid, gridDefinition);
        }
    }    
}

static __device__ double uPlus(double v){
  return fmax(v, 0.0);
}

static __device__ double uMinus(double v){
  return fmin(v, 0.0);
}

static __device__ double fluxLimiter(double th) {
  double min1 = fmin((1 + th)/2.0, 2.0);
  return fmax(0.0, fmin(min1, 2*th)); 
}

static __device__ void updateDcu(Cell* cell, Grid* grid, GridDefinition* gridDef) {    
    cell->dcu = 0.0;   
    for(int i=0; i<DIM; i++){
        cell->ctu[i] = 0.0;
        Cell* iCell = getCell(cell->iNodes[i], grid);
        Cell* kCell = getCell(cell->kNodes[i], grid);

        double dcu_p = 0;
        double dcu_m = 0;

        double vDownstream = cell->v[i];
	
        if(kCell != NULL){
          dcu_p = uPlus(vDownstream) * cell->prob + uMinus(vDownstream) * kCell->prob;
        }else{
          dcu_p = uPlus(vDownstream) * cell->prob;
        }
        
        if(iCell != NULL){
            double vUpstream = iCell->v[i]; // original implementation
            // double vUpstream = 0.5*(iCell->v[i] + cell->v[i]); // corrected implementation (valid only for equispaced meshes)
            dcu_m = uPlus(vUpstream) * iCell->prob + uMinus(vUpstream) * cell->prob;
        }
        cell->dcu -= (gridDef->dt/gridDef->dx[i])*(dcu_p-dcu_m);             
    }
}

static __device__ void updateCtu(Cell* cell, Grid* grid, GridDefinition* gridDef) {         
 
    for(int i=0; i<DIM; i++){
        Cell* iCell = getCell(cell->iNodes[i], grid);    
        
        if(iCell == NULL) continue;    
        
        double flux = gridDef->dt*(cell->prob - iCell->prob) / (2.0 * gridDef->dx[i]);                    
        double vUpstream = iCell->v[i];
       
        for(int j=0; j<DIM; j++){
            if(j == i) continue;
            
            Cell* jCell = getCell(cell->iNodes[j], grid);
            Cell* pCell = getCell(iCell->iNodes[j], grid);
            
            atomicAdd(&cell->ctu[j], -uPlus(vUpstream) * uPlus(cell->v[j]) * flux);
            atomicAdd(&iCell->ctu[j], -uMinus(vUpstream) * uPlus(iCell->v[j]) * flux);
                
            if(jCell != NULL){
                atomicAdd(&jCell->ctu[j], -uPlus(vUpstream) * uMinus(jCell->v[j]) * flux);                
            }
            if(pCell != NULL){
                atomicAdd(&pCell->ctu[j], -uMinus(vUpstream) * uMinus(pCell->v[j]) * flux);                
            }
        }
  
	    //High-resolution correction terms
        double th = 0.0;
        if (vUpstream > 0){
            Cell* iiCell = getCell(iCell->iNodes[i], grid);
            th = (iiCell != NULL)? 
                (iCell->prob - iiCell->prob)/(cell->prob - iCell->prob):
                (iCell->prob)/(cell->prob - iCell->prob);
                
        } else {
            Cell* kCell = getCell(cell->kNodes[i], grid);
            th = (kCell != NULL)?
                (kCell->prob - cell->prob)/(cell->prob - iCell->prob):
                (-cell->prob)/(cell->prob - iCell->prob);                             
        }       
        atomicAdd(&iCell->ctu[i], fabs(vUpstream)*(gridDef->dx[i]/gridDef->dt - fabs(vUpstream))*flux*fluxLimiter(th));         
    }   
}

/** Update probability for one cell */
static __device__ void updateProbability(int offset, int iterations, GridDefinition* gridDefinition, Grid* grid) {    
    // initialize cells
    for(int iter=0;iter<iterations;iter++){        
      int usedIndex = getIndex(offset, iter); // index in the used list
      Cell* cell = getCell(usedIndex, grid); 
      if(cell != NULL){
        updateProbabilityCell(cell, grid, gridDefinition);        
      }
    }
}

/** Update probability for one cell */
static __device__ void updateProbabilityCell(Cell* cell, Grid* grid, GridDefinition* gridDef){
    cell->prob += cell->dcu;
        
    for(int i=0; i<DIM; i++){
        Cell* iCell = getCell(cell->iNodes[i], grid);
        cell->prob -= (iCell != NULL)?
            (gridDef->dt / gridDef->dx[i]) * (cell->ctu[i] - iCell->ctu[i]):
            (gridDef->dt / gridDef->dx[i]) * (cell->ctu[i]);         
    }        
    cell->prob = fmax(cell->prob, 0.0);        
}

/** Apply measurement */
static __device__ void applyMeasurement(int offset, int iterations, Measurement* measurement, GridDefinition* gridDefinition, Grid* grid, Model* model){
    // initialize cells
    for(int iter=0;iter<iterations;iter++){        
      int usedIndex = getIndex(offset, iter); // index in the used list
      Cell* cell = getCell(usedIndex, grid); 
      if(cell != NULL){
        applyMeasurementCell(cell, measurement, gridDefinition, grid, model);        
      }
    }
}

/** Apply measurement for one cell */
static __device__ void applyMeasurementCell(Cell* cell, Measurement* measurement, GridDefinition* gridDefinition, Grid* grid, Model* model){
    // call measurement function
    double y[DIM];
    (*model->callbacks->z)(y, cell->x, gridDefinition->dx); 
    
    //  compute and update probability
    double prob = gaussProbability(y, measurement);   
    cell->prob *= prob;
}

/**
 * @brief Dummy kernel to check maximum teoretical concurrent threads
 */
__global__ void dummyKernel(int iterations, Model model, Global global){}