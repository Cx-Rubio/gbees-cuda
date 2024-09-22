// Copyright 2024 by Carlos Rubio, published under BSD 3-Clause License.

#include "kernel.h"
#include <stdio.h>

/** Calculate gaussian probability at state x given mean and covariance */
static __device__ double gaussProbability(int32_t* x, Measurement* measurement);

/** Invert a matrix of dimension DIM */
static __device__ void invertMatrix(double matrix[DIM][DIM], double inverse[DIM][DIM], int size);

/** Multiply a matrix by a vector */
static __device__ void multiplyMatrixVector(double matrix[DIM][DIM], double* vector, double* result, int size);

/** Compute dot product */
static __device__ double computeDotProduct(double* vec1, double* vec2, int size);

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
    double mInv[DIM][DIM];
    double mInvX[DIM];
    double diff[DIM];
    
    for(int i=0;i<DIM;i++){
        diff[i] = x[i] - measurements[0].mean[i];
    }
    invertMatrix(measurements[0].cov, mInv, DIM);
    multiplyMatrixVector(mInv, diff, mInvX, DIM);
    double dotProduct = computeDotProduct(diff, mInvX, DIM);
    return exp(-0.5 * dotProduct);
}

/** Invert a matrix. Max input dimension DIM */
static __device__ void invertMatrix(double matrix[DIM][DIM], double inverse[DIM][DIM], int size) {
    int i, j, k;
    double ratio;
    double augmented[DIM*DIM*2];

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            augmented[i * 2 * size + j] = matrix[i][j];
            augmented[i * 2 * size + (j + size)] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (i = 0; i < size; i++) {
        if (augmented[i * 2 * size + i] == 0) {
            printf("Error: matrix inversion error, zero pivot element\n");
            __trap();
        }
        for (j = 0; j < size; j++) {
            if (i != j) {
                ratio = augmented[j * 2 * size + i] / augmented[i * 2 * size + i];
                for (k = 0; k < 2 * size; k++) {
                    augmented[j * 2 * size + k] -= ratio * augmented[i * 2 * size + k];
                }
            }
        }
    }

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            //inverse[i * size + j] = augmented[i * 2 * size + (j + size)] / augmented[i * 2 * size + i]; TODO check
            inverse[i][j] = augmented[i * 2 * size + (j + size)] / augmented[i * 2 * size + i];
        }
    }
    
}

/** Multiply a matrix by a vector */
static __device__ void multiplyMatrixVector(double matrix[DIM][DIM], double* vector, double* result, int size) {
    // TODO check as alternative using tensor cores
    int i, j;
    for (i = 0; i < size; i++) {
        result[i] = 0;
        for (j = 0; j < size; j++) {
            result[i] += matrix[i][j] * vector[j]; // TODO check
        }
    }
}

/** Compute dot product */
static __device__ double computeDotProduct(double* vec1, double* vec2, int size) {
    int i;
    double result = 0;
    for (i = 0; i < size; i++) {
        result += vec1[i] * vec2[i];
    }
    return result;
}