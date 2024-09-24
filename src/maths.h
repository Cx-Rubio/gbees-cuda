// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#ifndef MATHS_H
#define MATHS_H

#include "config.h"

/** 
 * @brief Invert a matrix
 * 
 * @param matrix the matrix to invert
 * @param inverse [output] the inverse of matrix to fill in
 * @param size of the matrices
 */
__host__  void invertMatrix(double* matrix, double* inverse, int size);

/**
 * @brief Multiply a square matrix by a vector
 * 
 * @param matrix the matrix
 * @param vector the vector
 * @param result [output] the output vector
 * @param size the size of the matrix
 */
__device__ void multiplyMatrixVector(double* matrix, double* vector, double* result, int size);

/**
 * @brief Compute dot product
 * @param vec1 first vector
 * @param vec2 second vector
 * @param size vector size
 * @return the dot product
 */
__device__ double computeDotProduct(double* vec1, double* vec2, int size);

#endif