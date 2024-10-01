// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.
#include "config.h"
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
static __device__ void growGridFromCell(Cell* cell, GridDefinition* gridDefinition, Grid* grid, Model* model);

/** Grow grid from one cell in one dimension and direction */
static __device__ void growGridDireccional(Cell* cell, int dimension, enum Direction direction, GridDefinition* gridDefinition, Grid* grid, Model* model);

/** Create new cell in the grid */
static __device__ void createCell(int32_t* state, GridDefinition* gridDefinition, Grid* grid, Model* model);

/** 
 * @brief Initialization kernel function 
 * 
 * @param iterations number of cells that should process the same thread
 * @param gridDefinition the grid definition
 * @param grid the grid object
 * @param model the model
 * @param measurements the list of measurements
 */
__global__ void gbeesKernel(int iterations, GridDefinition gridDefinition, Model model, Global global){
    // grid synchronization
    cg::grid_group g = cg::this_grid(); 
    
    // shared memory for reduction processes
    __shared__ double localArray[THREADS_PER_BLOCK];   
    
    // get used list offset index
    int offsetIndex = threadIdx.x + blockIdx.x * blockDim.x * iterations;     
        
    // initialize cells
    for(int iter=0;iter<iterations;iter++){        
        int usedIndex = (uint32_t)(offsetIndex + iter * blockDim.x); // index in the used list                
        initializeCell(usedIndex, &gridDefinition, &model, &global); // initialize cell
    }    
    
    // set grid maximum and minimum bounds
    if(model.useBounds){ // TODO test use bounds
        initializeGridBoundary(offsetIndex, iterations, localArray, &gridDefinition, &global);   
        
        /* if(offsetIndex == 0){
            printf("Bounds min %e\n", gridDefinition.lo_bound);
            printf("Bounds max %e\n", gridDefinition.hi_bound);
        } */
    }
    
    // normalize distribution
    normalizeDistribution(offsetIndex, iterations, localArray, global.reductionArray, global.grid);

    //if(key[0] == -3 && key[1] == -2 && key[2] == 5) printf("Probability %e\n", prob);
    //if(usedIndex == 100) printf("Probability of %d,%d,%d : %f\n", key[0], key[1], key[2], prob);
    
    /*if(key[0] == 6 && key[1] == 6 && key[2] == 0){
    if(usedIndex == 0){    
        printf("key %d, %d, %d\n",cell->state[0],cell->state[1],cell->state[2]);
        int dim = 0;
        uint32_t iNode = cell->iNodes[dim];
        uint32_t kNode = cell->kNodes[dim];
        
        if(iNode){
            int heapIndexI = (grid.usedList + (iNode-1))->heapIndex;
            Cell* cellI = &grid.heap[heapIndexI];
            printf("I node %d, %d, %d\n", cellI->state[0], cellI->state[1], cellI->state[2]);
        } else {
            printf("Mo iNode\n");
        }
        
        if(kNode){            
            int heapIndexK = (grid.usedList + (kNode-1))->heapIndex;
            Cell* cellK = &grid.heap[heapIndexK];
            printf("K node %d, %d, %d\n", cellK->state[0], cellK->state[1], cellK->state[2]);   
        } else {
            printf("Mo kNode\n");
        }            
    }*/
    
    // for each measurement
    for(int nm=0;nm<model.numMeasurements;nm++){
        // select active measurement
        Measurement* measurement = &global.measurements[nm];
        
        // propagate probability distribution until the next measurement
        double mt = 0.0; // time propagated from the last measurement
        //int stepCount = 1; // step count
        while(fabs(mt - measurement->T) > TOL) { 
            growGrid(offsetIndex, iterations, &gridDefinition, global.grid, &model);
            updateIkNodes(offsetIndex, iterations, global.grid);
            
            // TODO remove
            /*            
            g.sync();    
            if(threadIdx.x == 0) printf("used size: %d\n", global.grid->usedSize);*/
            
            checkCflCondition(offsetIndex, iterations, localArray, &gridDefinition, &global);
    
            /*
            
            godunov_method();
            update_prob();
            normalize_tree();
            
            if (step_count % DEL_STEP == 0) { // deletion procedure
                prune_tree();
                normalize_tree(); 
            }
         
            stepCount++;
            */
            // FIXME take account of G.dt
            break; // FIXME remove
        }
        
        // perform Bayesian update for the next measurement
        if(nm < model.numMeasurements -1){
            /*
            meas_up_recursive();
            normalize_tree();
            prune_tree();
            normalize_tree(); 
            */
        }
        break; // FIXME remove
    }
    
    
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
    
    /*if(cell->state[0]==0 && cell->state[1]==2 && cell->state[2]==2){
        printf("cell cfl_df %e, v[0] %e, v[1] %e, v[2] %e \n", cell->cfl_dt, cell->v[0], cell->v[1], cell->v[2] );
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
    uint32_t usedIndex;
    for(int dim=0; dim<DIM; dim++){
        // node i
        copyKey(cell->state, state);
        state[dim] -= 1;
        usedIndex = findCell(state, grid);
        if(usedIndex){
            cell->iNodes[dim] = usedIndex;
        }
            
        // node k        
        state[dim] += 2; // to reach +1
        usedIndex = findCell(state, grid);
        if(usedIndex){
            cell->iNodes[dim] = usedIndex;
        }
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
    double minDt = gridDefinition->dt;
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
    if(threadIdx.x == 0 && blockIdx.x == 0){
            printf("prob sum %e\n", globalArray[0]);
    }
    
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
    
    uint32_t usedSize = grid->usedSize;
        
    g.sync();    
        
    for(int iter=0;iter<iterations;iter++){ 
        uint32_t usedIndex = (uint32_t)(offsetIndex + iter * blockDim.x); // index in the used list  
        
        if(usedIndex < usedSize){             
            Cell* cell = getCell(usedIndex, grid);
            if(cell->prob >= gridDefinition->threshold){
                growGridFromCell(cell, gridDefinition, grid, model);
            }
        }    
    }          
    g.sync(); 
}

/** Grow grid from one cell */
static __device__ void growGridFromCell(Cell* cell, GridDefinition* gridDefinition, Grid* grid, Model* model){
    for(int dimension=0;dimension<DIM;dimension++){
        if(cell->v[dimension] > 0.0){
            growGridDireccional(cell, dimension, FORWARD, gridDefinition, grid, model);        
        } else if(cell->v[dimension] < 0.0){
            growGridDireccional(cell, dimension, BACKWARD, gridDefinition, grid, model);    
        }
    }
}

/** Grow grid from one cell in one dimension and direction */
static __device__ void growGridDireccional(Cell* cell, int dimension, enum Direction direction, GridDefinition* gridDefinition, Grid* grid, Model* model){
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
            } else if(cell->v[j] < 0.0){
                // create new cell key[dimension] = cell->key[dimension] = key[dimension]+direction & cell->key[j] = cell->key[j]-1
                copyKey(cell->state, state);
                state[dimension] += direction;
                state[j] -=1;
                //createCell(state, gridDefinition, grid, model);
            }
        }
    }        
}

/** Create new cell in the grid */
static __device__ void createCell(int32_t* state, GridDefinition* gridDefinition, Grid* grid, Model* model){
    Cell cell;
    
    // compute state
    for(int i=0;i<DIM;i++){
        cell.state[i] = state[i]; // state coordinates
        cell.x[i] = gridDefinition->dx[i] * state[i] + gridDefinition->center[i]; // state value
    }
        
    cell.prob = 0.0; 
    cell.new_f = 0;
    cell.ik_f = 0;
    initializeAdv(gridDefinition, model, &cell);
    // TODO initialize ctu[] y dcu
    
    // insert cell
    insertCellConcurrent(&cell, grid);
}