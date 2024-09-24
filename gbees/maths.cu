#include "maths.h"
#include <stdio.h>

/** Invert a matrix. Max input dimension DIM */
__device__ void invertMatrix(double matrix[DIM][DIM], double inverse[DIM][DIM], int size) {
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
            printf("Error: matrix inversion error, zero pivot element\n"); // FIXME change in host
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
__device__ void multiplyMatrixVector(double matrix[DIM][DIM], double* vector, double* result, int size) {
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
__device__ double computeDotProduct(double* vec1, double* vec2, int size) {
    int i;
    double result = 0;
    for (i = 0; i < size; i++) {
        result += vec1[i] * vec2[i];
    }
    return result;
}
