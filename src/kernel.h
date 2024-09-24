// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#ifndef KERNEL_H
#define KERNEL_H

#include "grid.h"
#include "measurement.h"

/** 
 * @brief Initialization kernel function 
 * 
 * @param grid the grid object
 */
__global__ void initializationKernel(Grid grid, Measurement* measurements);

#endif