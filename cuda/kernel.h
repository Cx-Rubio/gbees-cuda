// Copyright 2024 by Carlos Rubio, published under BSD 3-Clause License.

#ifndef KERNEL_H
#define KERNEL_H

#include "grid.h"
#include "../gbees/measurement.h"

/** 
 * @brief Initialization kernel function 
 * 
 * @param grid the grid object
 */
__global__ void initializationKernel(Grid grid, Measurement* measurements);

#endif