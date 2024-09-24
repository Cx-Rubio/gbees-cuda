// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#include "maths.h"
#include "error.h"
#include <stdio.h>

/** 
 * @brief Invert a matrix
 * 
 * @param matrix the matrix to invert
 * @param inverse [output] the inverse of matrix to fill in
 * @param size of the matrices
 */
__host__ void invertMatrix(double* matrix, double* inverse, int size) {
    int i, j, k;
    double ratio;
    double* augmented = (double*)malloc(size * size * 2 * sizeof(double));
        
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            augmented[i * 2 * size + j] = matrix[i * size + j];
            augmented[i * 2 * size + (j + size)] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (i = 0; i < size; i++) {
        assertNotZero(augmented[i * 2 * size + i], MATH_ERROR, "Error: matrix inversion error, zero pivot element");
        
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
            inverse[i * size + j] = augmented[i * 2 * size + (j + size)] / augmented[i * 2 * size + i];
        }
    }
    
    free(augmented);    
}

/**
 * @brief Multiply a square matrix by a vector
 * 
 * @param matrix the matrix
 * @param vector the vector
 * @param result [output] the output vector
 * @param size the size of the matrix
 */
__device__ void multiplyMatrixVector(double* matrix, double* vector, double* result, int size) {
    // TODO check as alternative using tensor cores
    int i, j;
    for (i = 0; i < size; i++) {
        result[i] = 0;
        for (j = 0; j < size; j++) {            
            result[i] += matrix[i * size + j] * vector[j];
        }
    }
}

/**
 * @brief Compute dot product
 * @param vec1 first vector
 * @param vec2 second vector
 * @param size vector size
 * @return the dot product
 */
__device__ double computeDotProduct(double* vec1, double* vec2, int size) {
    int i;
    double result = 0;
    for (i = 0; i < size; i++) {
        result += vec1[i] * vec2[i];
    }
    return result;
}
