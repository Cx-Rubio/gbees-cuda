// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#include "kernel.h"
#include <stdio.h>
#include "maths.h"

/** Calculate gaussian probability at state x given mean and covariance */
static __device__ double gaussProbability(int32_t* x, Measurement* measurement);

/** 
 * 
 * 
 * @brief Initialization kernel function 
 * 
 * @param grid the grid object
 */
__global__ void initializationKernel(Grid grid, Measurement* measurements){
    // get used list index
    int usedIndex = (uint32_t)(threadIdx.x + blockIdx.x * blockDim.x);    
    
    if(usedIndex == 0) printf("Kernel usedIndex %d\n", usedIndex); // TODO remove
    
    // check used list size
    if(usedIndex >= grid.usedSize) return;
    
    if(usedIndex > 0 ) return; // FIXME remove this line
    
    // used list entry
    UsedListEntry* usedListEntry = grid.usedList + usedIndex;
    
    // obtain key (state coordinates)
    uint32_t hashtableIndex = usedListEntry->hashTableIndex;
    int32_t* key = grid.table[hashtableIndex].key;
    
    // compute initial probability    
    double prob = gaussProbability(key, measurements);
    
    // update cell
    uint32_t heapIndex = usedListEntry->heapIndex;    
    grid.heap[heapIndex].prob = prob; 

    //if(key[0] == 1 && key[1] == 0 && key[2] == 0) printf("Probability %f\n", prob); // TODO remove
    if(usedIndex == 0) printf("Probability of %d,%d,%d : %f\n", key[0], key[1], key[2], prob); // TODO remove    
}

/** Calculate gaussian probability at state x given mean and covariance */
static __device__ double gaussProbability(int32_t* x, Measurement* measurements){    
    double mInvX[DIM];
    double diff[DIM];
    
    for(int i=0;i<DIM;i++){
        diff[i] = x[i] - measurements[0].mean[i];
    }  
    multiplyMatrixVector( (double*)measurements[0].covInv, diff, mInvX, DIM);
    double dotProduct = computeDotProduct(diff, mInvX, DIM);
    return exp(-0.5 * dotProduct);
    
}

