// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.
#include "config.h"
#include "kernel.h"
#include <stdio.h>
#include <cooperative_groups.h>
#include "maths.h"
#include <float.h>

namespace cg = cooperative_groups;

/** Calculate gaussian probability at state x given mean and covariance */
static __device__ double gaussProbability(int32_t* key, GridDefinition* gridDefinition, Measurement* measurements);

/** Initialize ADV */
static __device__ void initializeAdv(GridDefinition* gridDefinition, Model* model, Cell* cell);

/** Initialize ik nodes */
static __device__ void initializeIkNodes(Grid* grid, Cell* cell, uint32_t usedIndex);

/** Initialize boundary value */
static __device__ void initializeBoundary(Cell* cell, Model* model);

/** Normalize probability distribution */
static __device__ void normalizeDistribution(double* localArray, double* globalArray, Cell* cell, double prob);

/** Compute grid bounds */
static __device__ void gridBounds(double* output, double* localArray, double* globalArray, double boundaryValue, double(*fn)(double, double) );

/** 
 * @brief Initialization kernel function 
 * 
 * @param gridDefinition the grid definition
 * @param grid the grid object
 * @param model the model
 * @param measurements the list of measurements
 */
__global__ void initializationKernel(GridDefinition gridDefinition, Grid grid, Model model, Global global){ // FIXME if separate the kernels, needs to keep the bounds changes in gridDefinition (check other parameters)
    
    // shared memory for reduction processes
    __shared__ double localArray[THREADS_PER_BLOCK];   
    
    // get used list index
    int usedIndex = (uint32_t)(threadIdx.x + blockIdx.x * blockDim.x);        
    
    // intialize cells
    double prob = 0.0;
    Cell* cell = NULL;
    if(usedIndex < grid.usedSize){    
        // used list entry
        UsedListEntry* usedListEntry = grid.usedList + usedIndex;
        
        // obtain key (state coordinates)
        uint32_t hashtableIndex = usedListEntry->hashTableIndex;
        int32_t* key = grid.table[hashtableIndex].key;
        
        // compute initial probability    
        prob = gaussProbability(key, &gridDefinition, global.measurements);
        
        // update cell
        uint32_t heapIndex = usedListEntry->heapIndex;    
        cell = &grid.heap[heapIndex];
        cell->new_f = 0;
        
        // compute state
        for(int i=0;i<DIM;i++){
            cell->state[i] = key[i]; // state coordinates
            cell->x[i] = gridDefinition.dx[i] * key[i] + gridDefinition.center[i]; // state value
        }
        
        cell->prob = prob; 
        initializeAdv(&gridDefinition, &model, cell);
        initializeIkNodes(&grid, cell, usedIndex);    
        
        // initialize bounday value
        if(model.useBounds){
            initializeBoundary(cell, &model);
        }
    }
    
    if(model.useBounds){
        // set grid maximum and minimum bounds
        double boundaryValue = (cell != NULL)? cell->bound_val : -DBL_MAX;
        gridBounds(&gridDefinition.hi_bound, localArray, global.reductionArray, boundaryValue, fmax);
        
        if(cell == NULL) { boundaryValue = DBL_MAX; };
        gridBounds(&gridDefinition.lo_bound, localArray, global.reductionArray, boundaryValue, fmin);
        
        /*if(usedIndex == 0){
            printf("Bounds min %e\n", gridDefinition.lo_bound);
            printf("Bounds max %e\n", gridDefinition.hi_bound);
        }*/
    }
    
    // normalize distribution
    normalizeDistribution(localArray, global.reductionArray, cell, prob);

    //if(key[0] == -3 && key[1] == -2 && key[2] == 5) printf("Probability %e\n", prob);
    //if(usedIndex == 100) printf("Probability of %d,%d,%d : %f\n", key[0], key[1], key[2], prob);
    
    /*if(key[0] == 6 && key[1] == 6 && key[2] == 0){
    //if(usedIndex == 0){    
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

/** Initialize ADV */
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
    for(int i=DIM-1;;i--){        
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

static __device__ void initializeBoundary(Cell* cell, Model* model){
    double j = (*model->callbacks->j)(cell->x);
    cell->bound_val = j;
}

/** Normalize probability distribution */
static __device__ void normalizeDistribution(double* localArray, double* globalArray, Cell* cell, double prob){    
    // grid synchronization
    cg::grid_group g = cg::this_grid();      
    
    // store cell probability in the reduction array
    localArray[threadIdx.x] = prob;
    
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
        
        g.sync();
        
        // reduction process in global memory (sequencial addressing)
        for(int s=1;s<gridDim.x;s*=2){
            int indexDst = 2 * s * blockIdx.x;
            int indexSrc = indexDst + s;
            if(indexSrc < gridDim.x){
                globalArray[indexDst] += globalArray[indexSrc];            
            }
            g.sync();
        }     
    }    
     
    // at the end, the sum of the probability its at globalArray[0]
    /*
    if(threadIdx.x == 0){
        printf("prob block sum %e\n", localArray[0]);
        }
    */
    if(threadIdx.x == 0 && blockIdx.x == 0){
            printf("prob sum %e\n", globalArray[0]);
    }
    
    // update the probability of the cell
    if(cell != NULL){
        cell->prob /= globalArray[0];    
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
        
        g.sync();
        
        // reduction process in global memory (sequencial addressing)   
        for(int s=1;s<gridDim.x;s*=2){
            int indexDst = 2 * s * blockIdx.x;
            int indexSrc = indexDst + s;
            if(indexSrc < gridDim.x){
                globalArray[indexDst] = fn(globalArray[indexSrc], globalArray[indexDst]);            
            }
            g.sync();
        } 
        if(blockIdx.x == 0){
            *output = globalArray[0];
        }
        g.sync();
    }        
}