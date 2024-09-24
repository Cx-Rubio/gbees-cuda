// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.
#include "config.h"
#include "kernel.h"
#include <stdio.h>
#include "maths.h"

/** Calculate gaussian probability at state x given mean and covariance */
static __device__ double gaussProbability(int32_t* key, GridDefinition* gridDefinition, Measurement* measurements);

/** Initialize ADV */
static __device__ void initializeAdv(GridDefinition* gridDefinition, Model* model, Cell* cell);

/** Initialize ik nodes */
static __device__ void initializeIkNodes(GridDefinition* gridDefinition, Model* model, Cell* cell);

/** 
 * @brief Initialization kernel function 
 * 
 * @param gridDefinition the grid definition
 * @param grid the grid object
 * @param model the model
 * @param measurements the list of measurements
 */
__global__ void initializationKernel(GridDefinition gridDefinition, Grid grid, Model model, Measurement* measurements){
    // get used list index
    int usedIndex = (uint32_t)(threadIdx.x + blockIdx.x * blockDim.x);    
    
    // check used list size
    if(usedIndex >= grid.usedSize) return;
    
    // used list entry
    UsedListEntry* usedListEntry = grid.usedList + usedIndex;
    
    // obtain key (state coordinates)
    uint32_t hashtableIndex = usedListEntry->hashTableIndex;
    int32_t* key = grid.table[hashtableIndex].key;
    
    // compute initial probability    
    double prob = gaussProbability(key, &gridDefinition, measurements);
    
    // update cell
    uint32_t heapIndex = usedListEntry->heapIndex;    
    Cell* cell = &grid.heap[heapIndex];
    cell->new_f = 0;
    for(int i=0;i<DIM;i++){ cell->state[i] = key[i]; }
    cell->prob = prob; 
    initializeAdv(&gridDefinition, &model, cell);
    initializeIkNodes(&gridDefinition, &model, cell);

    //if(key[0] == -3 && key[1] == -2 && key[2] == 5) printf("Probability %e\n", prob);
    //if(usedIndex == 100) printf("Probability of %d,%d,%d : %f\n", key[0], key[1], key[2], prob);
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

/** Initialize ik nodes */
static __device__ void initializeIkNodes(GridDefinition* gridDefinition, Model* model, Cell* cell){
    for(int i=0; i<DIM; i++){
    }
    
    cell->ik_f = 1;
}