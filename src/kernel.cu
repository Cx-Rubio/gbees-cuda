// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.
#include "config.h"
#include "kernel.h"
#include <stdio.h>
#include <cooperative_groups.h>
#include "maths.h"

namespace cg = cooperative_groups;

/** Calculate gaussian probability at state x given mean and covariance */
static __device__ double gaussProbability(int32_t* key, GridDefinition* gridDefinition, Measurement* measurements);

/** Initialize ADV */
static __device__ void initializeAdv(GridDefinition* gridDefinition, Model* model, Cell* cell);

/** Initialize ik nodes */
static __device__ void initializeIkNodes(Grid* grid, Cell* cell, uint32_t usedIndex);

/** Normalize probability distribution */
static __device__ void normalizeDistribution(double* probGlobalArray, Cell* cell, double prob);

/** 
 * @brief Initialization kernel function 
 * 
 * @param gridDefinition the grid definition
 * @param grid the grid object
 * @param model the model
 * @param measurements the list of measurements
 */
__global__ void initializationKernel(GridDefinition gridDefinition, Grid grid, Model model, Global global){
    
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
        for(int i=0;i<DIM;i++){ cell->state[i] = key[i]; }
        cell->prob = prob; 
        initializeAdv(&gridDefinition, &model, cell);
        initializeIkNodes(&grid, cell, usedIndex);    
    }
    
    // normalize distribution
    normalizeDistribution(global.probAccumulator, cell, prob);

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
    double x[DIM];
    for(int i=0; i<DIM; i++){
        x[i] = gridDefinition->dx[i]*cell->state[i]+gridDefinition->center[i];
    }
    
    double xk[DIM];
    (*model->callbacks->f)(xk, x, gridDefinition->dx); 

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

/** Normalize probability distribution */
static __device__ void normalizeDistribution(double* probGlobalArray, Cell* cell, double prob){
    // shared memory for reduction process
    __shared__ double probLocalArray[THREADS_PER_BLOCK];   

    // grid synchronization
    cg::grid_group g = cg::this_grid();  
    
    int threadIndex = threadIdx.x;
    int blockIndex = blockIdx.x;
    int blockCount = gridDim.x;
    int blockSize = THREADS_PER_BLOCK;
    
    // store cell probability in the reduction array
    probLocalArray[threadIndex] = prob;
    
    __syncthreads();
    
    // reduction process in shared memory (sequencial addressing)
    for(int s=1;s<blockSize;s*=2){
        int indexDst = 2 * s * threadIndex;
        int indexSrc = indexDst + s;
        if(indexSrc < blockSize){
            probLocalArray[indexDst] += probLocalArray[indexSrc];                        
        }
        __syncthreads();
    }
        
    if(threadIndex == 0){        
        // store total sum to global array
        probGlobalArray[blockIndex] = probLocalArray[0];       
        
        g.sync();
        
        // reduction process in global memory    
        for(int s=1;s<blockCount;s*=2){
            int indexDst = 2 * s * blockIndex;
            int indexSrc = indexDst + s;
            if(indexSrc < blockCount){
                probGlobalArray[indexDst] += probGlobalArray[indexSrc];            
            }
            g.sync();
        }     
    }    
     
    // at the end, the sum of the probability its at probGlobalArray[0]
    /*
    if(threadIndex == 0){
        printf("prob block sum %e\n", probLocalArray[0]);
        }
    */
    if(threadIndex == 0 && blockIndex == 0){
            printf("prob sum %e\n", probGlobalArray[0]);
    }
    
    // update the probability of the cell
    if(cell != NULL){
        cell->prob /= probGlobalArray[0];    
    }
}