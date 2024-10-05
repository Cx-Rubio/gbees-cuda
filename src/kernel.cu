// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.
#include "config.h"
#include "util.h"
#include "kernel.h"
#include <stdio.h>
#include <cooperative_groups.h>
#include "maths.h"
#include <float.h>

namespace cg = cooperative_groups;

/** Initialize cells */
static __device__ void initializeCell(uint32_t usedIndex, GridDefinition* gridDefinition, Model* model, Global* global);

/** Calculate gaussian probability at state x given mean and covariance */
static __device__ double gaussProbability(int32_t* key, GridDefinition* gridDefinition, Measurement* measurements);

/** Initialize advection values */
static __device__ void initializeAdv(GridDefinition* gridDefinition, Model* model, Cell* cell);

/** Initialize ik nodes */
static __device__ void initializeIkNodes(Grid* grid, Cell* cell, uint32_t usedIndex);

/** Update ik nodes */
static __device__ void updateIkNodes(int offsetIndex, int iterations, Grid* grid);

/** Update ik nodes for one cell */
static __device__ void updateIkNodesCell(Cell* cell, Grid* grid);

/** Initialize boundary value */
static __device__ void initializeBoundary(Cell* cell, Model* model);

/** Initialize Grid boundary */
static __device__ void initializeGridBoundary(int offsetIndex, int iterations, double* localArray, GridDefinition* gridDefinition, Global* global);

/** Normalize probability distribution */
static __device__ void normalizeDistribution(int offsetIndex, int iterations, double* localArray, double* globalArray, Grid* grid);

/** Compute grid bounds */
static __device__ void gridBounds(double* output, double* localArray, double* globalArray, double boundaryValue, double(*fn)(double, double) );

/** Compute step dt */
static __device__ void checkCflCondition(int offsetIndex, int iterations, double* localArray, GridDefinition* gridDefinition, Global* global);

/** Grow grid */
static __device__ void growGrid(int offsetIndex, int iterations, GridDefinition* gridDefinition, Grid* grid, Model* model);

/** Grow grid from one cell */
static __device__ void growGridFromCell(Cell* cell, GridDefinition* gridDefinition, Grid* grid, Model* model, int* synchCount, cg::grid_group g);

/** Grow grid from one cell in one dimension and direction */
static __device__ void growGridDireccional(Cell* cell, int dimension, enum Direction direction, GridDefinition* gridDefinition, Grid* grid, Model* model, int* synchCount, cg::grid_group g);

/** Create new cell in the grid */
static __device__ void createCell(int32_t* state, GridDefinition* gridDefinition, Grid* grid, Model* model);

/** Perform the Godunov scheme on the discretized PDF*/
static __device__ void godunovMethod(int offsetIndex, int iterations, GridDefinition* gridDefinition, Grid* grid);

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
static __device__ void updateProbability(int offsetIndex, int iterations, GridDefinition* gridDefinition, Grid* grid);

/** Update probability for one cell */
static __device__ void updateProbabilityCell(Cell* cell, Grid* grid, GridDefinition* gridDef);
	
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
    __shared__ double localArray[THREADS_PER_BLOCK];   
    
    // get used list offset index
    int offsetIndex = threadIdx.x + blockIdx.x * blockDim.x * iterations;     
        
    // initialize cells
    for(int iter=0;iter<iterations;iter++){        
        int usedIndex = (uint32_t)(offsetIndex + iter * blockDim.x); // index in the used list                
        initializeCell(usedIndex, global.gridDefinition, &model, &global); // initialize cell
    }    
    
    // set grid maximum and minimum bounds
    if(model.useBounds){ // TODO test use bounds
        initializeGridBoundary(offsetIndex, iterations, localArray, global.gridDefinition, &global);   
        
        /* if(offsetIndex == 0){
            printf("Bounds min %e\n", global.gridDefinition->lo_bound);
            printf("Bounds max %e\n", global.gridDefinition->hi_bound);
        } */
    }
    
    // normalize distribution
    normalizeDistribution(offsetIndex, iterations, localArray, global.reductionArray, global.grid);
   
    // for each measurement
    double tt = 0.0;
    for(int nm=0;nm<model.numMeasurements;nm++){
        int stepCount = 1; // step count
        
        LOG("Timestep: %d-%d, Sim. time: %f", nm, stepCount, tt);
        LOG(" TU, Used Cells: %d/%d\n", global.grid->usedSize, global.grid->size);        
        
        // select active measurement
        Measurement* measurement = &global.measurements[nm];
        
        // propagate probability distribution until the next measurement
        double mt = 0.0; // time propagated from the last measurement            
        double rt;        
        // slight variation w.r.t to original implementation as record time is recalculated for each measurement
        double recordTime = measurement->T / (model.numDistRecorded-1);
        
        while(fabs(mt - measurement->T) > TOL) {              
            rt = 0.0;
           
            while(rt < recordTime) { // time between PDF recordings           
                growGrid(offsetIndex, iterations, global.gridDefinition, global.grid, &model);            
                updateIkNodes(offsetIndex, iterations, global.grid);            
                checkCflCondition(offsetIndex, iterations, localArray, global.gridDefinition, &global);             
                rt += global.gridDefinition->dt;
                godunovMethod(offsetIndex, iterations, global.gridDefinition, global.grid);
                g.sync();            
                updateProbability(offsetIndex, iterations, global.gridDefinition, global.grid);
                normalizeDistribution(offsetIndex, iterations, localArray, global.reductionArray, global.grid);
                LOG("step duration %f, active cells %d\n", global.gridDefinition->dt, global.grid->usedSize);
                        
                if (stepCount % model.deletePeriodSteps == 0) { // deletion procedure
                    LOG("prune tree at step %d\n", stepCount); 
                    // prune_tree(); TODO implement
                    normalizeDistribution(offsetIndex, iterations, localArray, global.reductionArray, global.grid);
                }
            
                if ((model.performOutput) && (stepCount % model.outputPeriodSteps == 0)) { // print size to terminal
                    LOG("Timestep: %d-%d, Sim. time: %f", nm, stepCount, tt + mt + rt);
                    LOG(" TU, Used Cells: %d/%d\n", global.grid->usedSize, global.grid->size); 
                }
                
                stepCount++;            
                // TODO check needed sycn
                //g.sync();
                //if(threadIdx.x == 0 && blockIdx.x == 0) global.gridDefinition->dt = DBL_MAX;                    
                //g.sync();
            } // while(rt < recordTime)

            if (((stepCount-1) % model.outputPeriodSteps != 0) || (!model.performOutput)){ // print size to terminal  
                LOG("Timestep: %d-%d, Sim. time: %f", nm, stepCount, tt + mt + rt);
                LOG(" TU, Used Cells: %d/%d\n", global.grid->usedSize, global.grid->size);
            }
            
            if(model.performRecord){ // record PDF
                LOG("Record PDF (not implemented)\n");
            }
            mt += rt;
            
            break;
        }
        tt += mt;
        // perform Bayesian update for the next measurement
        if(model.performMeasure && nm < model.numMeasurements -1){
            /*
            meas_up_recursive();
            normalize_tree();
            prune_tree();
            normalize_tree(); 
            */
        }
        break; // FIXME remove
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
        cell->new_f = 0;
        
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

/** Calculate gaussian probability at state x given mean and covariance */
static __device__ double gaussProbability(int32_t* key, GridDefinition* gridDefinition, Measurement* measurements){    
    double mInvX[DIM];
    double diff[DIM];
    
    for(int i=0;i<DIM;i++){
        diff[i] = key[i] * gridDefinition->dx[i];
    }
    multiplyMatrixVector( (double*)measurements[0].covInv, diff, mInvX, DIM);
    double dotProduct = computeDotProduct(diff, mInvX, DIM);
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
  
    cell->new_f = 1;
    cell->cfl_dt = 1.0/sum;
    
    /*if(cell->state[0]==2 && cell->state[1]==-1 && cell->state[2]==-1){
        printf("cell cfl_df %1.14e, v[0] %1.14e, v[1] %1.14e, v[2] %1.14e \n", cell->cfl_dt, cell->v[0], cell->v[1], cell->v[2] );
        } */
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
            cell->iNodes[i] = iIndex + 1; // reserve 0 for no reference            
        } else {            
            cell->iNodes[i] = 0;
        }
        
        // if is not the last cell in the dimension i        
        if(cell->state[i] < (int)grid->initialExtent[i]){
            uint32_t kIndex = usedIndex + offset;        
            cell->kNodes[i] = kIndex + 1; // reserve 0 for no reference            
        }  else {            
            cell->kNodes[i] = 0;
        }
        
        if(i<=0) break;
        offset *= grid->initialExtent[i] * 2 + 1;
    }    
    cell->ik_f = 1;
}

/** Update ik nodes */
static __device__ void updateIkNodes(int offsetIndex, int iterations, Grid* grid){
    for(int iter=0; iter<iterations; iter++){      
        uint32_t usedIndex = (uint32_t)(offsetIndex + iter * blockDim.x);
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

static __device__ void initializeGridBoundary(int offsetIndex, int iterations, double* localArray, GridDefinition* gridDefinition, Global* global){
    double boundaryValue = -DBL_MAX;    
    for(int iter=0; iter<iterations; iter++){        
        // index in the used list
        uint32_t usedIndex = (uint32_t)(offsetIndex + iter * blockDim.x);   
        Cell* cell = getCell(usedIndex, global->grid);        
        if(cell != NULL && cell->bound_val > boundaryValue) boundaryValue = cell->bound_val;
    }
    gridBounds(&gridDefinition->hi_bound, localArray, global->reductionArray, boundaryValue, fmax);
    
    boundaryValue = DBL_MAX;   
    for(int iter=0;iter<iterations;iter++){        
        // index in the used list
        uint32_t usedIndex = (uint32_t)(offsetIndex + iter * blockDim.x);   
        Cell* cell = getCell(usedIndex, global->grid);        
        if(cell != NULL && cell->bound_val < boundaryValue) boundaryValue = cell->bound_val;
    }
    gridBounds(&gridDefinition->lo_bound, localArray, global->reductionArray, boundaryValue, fmin);
}

/** Compute step dt */
static __device__ void checkCflCondition(int offsetIndex, int iterations, double* localArray, GridDefinition* gridDefinition, Global* global){
    double minDt = DBL_MAX; //gridDefinition->dt; TODO check
    for(int iter=0; iter<iterations; iter++){        
        // index in the used list
        uint32_t usedIndex = (uint32_t)(offsetIndex + iter * blockDim.x);   
        Cell* cell = getCell(usedIndex, global->grid);        
        if(cell != NULL && cell->cfl_dt < minDt) minDt = cell->cfl_dt;
    }    
    gridBounds(&gridDefinition->dt, localArray, global->reductionArray, minDt, fmin);    
}

/** Normalize probability distribution */
static __device__ void normalizeDistribution(int offsetIndex, int iterations, double* localArray, double* globalArray, Grid* grid){        
    // grid synchronization
    cg::grid_group g = cg::this_grid();      
   
    // store the sum of the cells probability for all the iterations at the local reduction array
    localArray[threadIdx.x] = 0.0;
    for(int iter=0;iter<iterations;iter++){
        uint32_t usedIndex = (uint32_t)(offsetIndex + iter * blockDim.x);           
        Cell* cell = getCell(usedIndex, grid); 
        if(cell != NULL) localArray[threadIdx.x] += cell->prob;
    }
    
    __syncthreads();
    
    // reduction process in shared memory (sequencial addressing)
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
     
    // reduction process in global memory (sequencial addressing)
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
   
    /*if(threadIdx.x == 0){
        printf("prob block sum %e\n", localArray[0]);
        }*/
    
    // at the end, the sum of the probability its at globalArray[0]    
   LOG("prob sum %1.14e\n", globalArray[0]); // TODO remove    
    
    // update the probability of the cells
    for(int iter=0;iter<iterations;iter++){
        uint32_t usedIndex = (uint32_t)(offsetIndex + iter * blockDim.x);   
        Cell* cell = getCell(usedIndex, grid);         
        if(cell != NULL) cell->prob /= globalArray[0];        
    }
}              


/** Set the grid definition bounds with the max and min boundary values of the initial grid cells */
static __device__ void gridBounds(double* output, double* localArray, double* globalArray, double boundaryValue, double(*fn)(double, double) ){
    // grid synchronization
    cg::grid_group g = cg::this_grid();      
    
    // store cell bounday value in the reduction array
    localArray[threadIdx.x] = boundaryValue;
    
    __syncthreads();
    
    // reduction process in shared memory (sequencial addressing)
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
        
    // reduction process in global memory (sequencial addressing)   
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
static __device__ void growGrid(int offsetIndex, int iterations, GridDefinition* gridDefinition, Grid* grid, Model* model){
    // grid synchronization
    cg::grid_group g = cg::this_grid(); 
    int synchCount = 0;
    
    uint32_t usedSize = grid->usedSize;
        
    g.sync();    
        
    for(int iter=0;iter<iterations;iter++){ 
        uint32_t usedIndex = (uint32_t)(offsetIndex + iter * blockDim.x); // index in the used list  
        
        if(usedIndex < usedSize){             
            Cell* cell = getCell(usedIndex, grid);
            if(cell->prob >= gridDefinition->threshold){
                growGridFromCell(cell, gridDefinition, grid, model, &synchCount, g);
            }
        }    
    }      

    int maxSync = iterations *4;
    //printf(" synchCount %d\n", synchCount);
    for(; synchCount<maxSync;synchCount++){
        g.sync();
    }
    
    g.sync(); 
}

/** Grow grid from one cell */
static __device__ void growGridFromCell(Cell* cell, GridDefinition* gridDefinition, Grid* grid, Model* model, int* synchCount, cg::grid_group g){
    // grid synchronization
    //cg::grid_group g = cg::this_grid(); 
    
    for(int dimension=0;dimension<DIM;dimension++){
        if(cell->v[dimension] > 0.0){
            growGridDireccional(cell, dimension, FORWARD, gridDefinition, grid, model, synchCount, g);        
        } 
    }
    
    // synchronization to avoid concurrent creation of the same cell
    //*synchCount +=1;
    //g.sync();
    
    for(int dimension=0;dimension<DIM;dimension++){
        if(cell->v[dimension] < 0.0){
            growGridDireccional(cell, dimension, BACKWARD, gridDefinition, grid, model, synchCount, g);    
        }
    }    
}

/** Grow grid from one cell in one dimension and direction */
static __device__ void growGridDireccional(Cell* cell, int dimension, enum Direction direction, GridDefinition* gridDefinition, Grid* grid, Model* model, int* synchCount, cg::grid_group g){
    // grid synchronization
    //cg::grid_group g = cg::this_grid(); 
    
    // check if already exists next face
    uint32_t nextFaceIndex = 0; // initialized to null reference
    int32_t state[DIM]; // state indexes for the new cells
    if(direction == FORWARD) nextFaceIndex = cell->kNodes[dimension];
    else nextFaceIndex = cell->iNodes[dimension];
    
    // create next face if not exists
    if(!nextFaceIndex){
        // create new cell key[dimension] = cell->key[dimension]+direction
        copyKey(cell->state, state);
        state[dimension] += direction;
        createCell(state, gridDefinition, grid, model);
    }
        
    // check edges
    for (int j = 0; j < DIM; j++){
        if(j != dimension){
            if(cell->v[j] > 0.0){
                // create new cell key[dimension] = cell->key[dimension] = key[dimension]+direction & cell->key[j] = cell->key[j]+1
                copyKey(cell->state, state);
                state[dimension] += direction;
                state[j] +=1;
                createCell(state, gridDefinition, grid, model);
            } 
        }
    }   
    
    // synchronization to avoid concurrent creation of the same cell
    //*synchCount +=1;
    //g.sync();
    
    // check edges
    for (int j = 0; j < DIM; j++){
        if(j != dimension){
            if(cell->v[j] < 0.0){
                // create new cell key[dimension] = cell->key[dimension] = key[dimension]+direction & cell->key[j] = cell->key[j]-1
                copyKey(cell->state, state);
                state[dimension] += direction;
                state[j] -=1;
                createCell(state, gridDefinition, grid, model);
            }
        }
    }   
}

/** Create new cell in the grid */
static __device__ void createCell(int32_t* state, GridDefinition* gridDefinition, Grid* grid, Model* model){
    Cell cell;
    
    // compute state
    for(int i=0;i<DIM;i++){
        cell.state[i] = state[i];
        cell.x[i] = gridDefinition->dx[i] * state[i] + gridDefinition->center[i]; // state value
        cell.ctu[i] = 0.0;
    }
        
    cell.prob = 0.0; 
    cell.new_f = 0;
    cell.ik_f = 0;
    cell.dcu = 0.0;
    initializeAdv(gridDefinition, model, &cell);
    
    // insert cell
    insertCellConcurrent(&cell, grid);
}

/** 
 * Perform the Godunov scheme on the discretized PDF. 
 * This function performs the Godunov scheme on the discretized PDF, which is 2nd-order accurate and total variation diminishing
 */
static __device__ void godunovMethod(int offsetIndex, int iterations, GridDefinition* gridDefinition, Grid* grid) {    
    // grid synchronization
    cg::grid_group g = cg::this_grid();    
     
    int usedIndex;
    Cell* cell;
    // initialize cells
    for(int iter=0;iter<iterations;iter++){        
        usedIndex = (uint32_t)(offsetIndex + iter * blockDim.x); // index in the used list
        cell = getCell(usedIndex, grid); 
        if(cell != NULL){
            updateDcu(cell, grid, gridDefinition);        
        }
    }
    
    g.sync();    
    
    for(int iter=0;iter<iterations;iter++){        
        usedIndex = (uint32_t)(offsetIndex + iter * blockDim.x); // index in the used list
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
  /** Adrian
  double min1 = (1.0 + th)/2.0;
  min1 = fmin(min1, 2.0);
  min1 = fmax(min1, 2.0*th);
  return min1; */
  
  // original
  double min1 = fmin((1 + th)/2.0, 2.0);
  return fmax(0.0, fmin(min1, 2*th)); 
}

static __device__ void updateDcu(Cell* cell, Grid* grid, GridDefinition* gridDef) {    
    cell->dcu = 0.0;   
    for(int i=0; i<DIM; i++){
        cell->ctu[i] = 0.0;
        Cell* iCell = getCell(cell->iNodes[i]-1, grid);
        Cell* kCell = getCell(cell->kNodes[i]-1, grid);

        double dcu_p = 0;
        double dcu_m = 0;

        /* Original implementation */        
        double vDownstream = cell->v[i];

        /*Corrected implementation (valid only for equispaced meshes) */
        //double vDownstream = 0.5*(cell->v[i] + kCell->v[i]);	
	
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
        
        /*
        if(cell->state[0] == 5 && cell->state[1] == -1 && cell->state[2] == 0 && i == 0){
            printf("cell->ctu[%d] %f, dcu %f\n", i, cell->ctu[i], cell->dcu);
            printf("G.dt %f, G.dx[%d]: %f, dcu_p %f, dcu_m %f \n", gridDef->dt, i, gridDef->dx[i], dcu_p, dcu_m);
            printf("uPlus(vUpstream): %f, iCell->prob: %f, uMinus(vUpstream): %f, cell->prob: %f \n", uPlus(vUpstream), iCell->prob, uMinus(vUpstream), cell->prob);
            printf("iCell key %d,%d,%d\n", iCell->state[0], iCell->state[1], iCell->state[2]);
            }
             */
    }
}

static __device__ void updateCtu(Cell* cell, Grid* grid, GridDefinition* gridDef) {         
 
    for(int i=0; i<DIM; i++){
        Cell* iCell = getCell(cell->iNodes[i]-1, grid);
        if(iCell == NULL) continue;    
        double flux = gridDef->dt*(cell->prob - iCell->prob) / (2.0 * gridDef->dx[i]);            
        
        double vUpstream = iCell->v[i];

        for(int j=0; j<DIM; j++){
            if(j == i) continue;
            
            Cell* jCell = getCell(cell->iNodes[j]-1, grid);
            Cell* pCell = getCell(iCell->iNodes[j]-1, grid);

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
            Cell* iiCell = getCell(iCell->iNodes[i]-1, grid);
            th = (iiCell != NULL)? 
                (iCell->prob - iiCell->prob)/(cell->prob - iCell->prob):
                (iCell->prob)/(cell->prob - iCell->prob);
                
        } else {
            Cell* kCell = getCell(cell->kNodes[i]-1, grid);
            th = (kCell != NULL)?
                (kCell->prob - cell->prob)/(cell->prob - iCell->prob):
                (-cell->prob)/(cell->prob - iCell->prob);                             
        }       
        atomicAdd(&iCell->ctu[i], fabs(vUpstream)*(gridDef->dx[i]/gridDef->dt - fabs(vUpstream))*flux*fluxLimiter(th));         
    }
}

/** Update probability for one cell */
static __device__ void updateProbability(int offsetIndex, int iterations, GridDefinition* gridDefinition, Grid* grid) {    
    int usedIndex;
    Cell* cell;
    // initialize cells
    for(int iter=0;iter<iterations;iter++){        
      usedIndex = (uint32_t)(offsetIndex + iter * blockDim.x); // index in the used list
      cell = getCell(usedIndex, grid); 
      if(cell != NULL){
        updateProbabilityCell(cell, grid, gridDefinition);        
      }
    }
}

/** Update probability for one cell */
static __device__ void updateProbabilityCell(Cell* cell, Grid* grid, GridDefinition* gridDef){
    cell->prob += cell->dcu;
        
    for(int i=0; i<DIM; i++){
        Cell* iCell = getCell(cell->iNodes[i]-1, grid);
        cell->prob -= (iCell != NULL)?
            (gridDef->dt / gridDef->dx[i]) * (cell->ctu[i] - iCell->ctu[i]):
            (gridDef->dt / gridDef->dx[i]) * (cell->ctu[i]);  

        /*if(cell->state[0] == 2 && cell->state[1] == 1 && cell->state[2] == 7){
            printf("iCell %p, %d,%d,%d\n", iCell, iCell->state[0], iCell->state[1], iCell->state[2]);
            //printf("dt %1.16f, dx %1.16f, ctu %1.16f, ictu %1.16f, dcu %1.16f, idcu %1.16f, i %d\n", gridDef->dt, gridDef->dx[i], cell->ctu[i], iCell->ctu[i], cell->dcu, iCell->dcu, i);
        } */                 
    }    
    
    cell->prob = fmax(cell->prob, 0.0);
    
   /* if(cell->state[0] == 2 && cell->state[1] == 1 && cell->state[2] == 7){
        printf("cell {%d,%d,%d}, prob: %1.16e\n", cell->state[0], cell->state[1], cell->state[2], cell->prob);
    }*/
    //    printf("%1.16e\n", cell->prob);
}