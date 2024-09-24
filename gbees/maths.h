#ifndef MATHS_H
#define MATHS_H

#include "../cuda/config.h"

/** Invert a matrix of dimension DIM */ // FIXME change to host
__device__ void invertMatrix(double matrix[DIM][DIM], double inverse[DIM][DIM], int size);

/** Multiply a matrix by a vector */
__device__ void multiplyMatrixVector(double matrix[DIM][DIM], double* vector, double* result, int size);

/** Compute dot product */
__device__ double computeDotProduct(double* vec1, double* vec2, int size);

#endif