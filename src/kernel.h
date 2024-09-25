// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#ifndef KERNEL_H
#define KERNEL_H

#include "grid.h"
#include "measurement.h"

/** Global working memory */
typedef struct {
    double* reductionArray; // global array for reduction processes
    Measurement* measurements;
} Global;


/** 
 * @brief Initialization kernel function 
 * 
 * @param gridDefinition the grid definition
 * @param grid the grid object
 * @param model the model
 * @param measurements the list of measurements
 */
__global__ void initializationKernel(GridDefinition gridDefinition, Grid grid, Model model, Global global);

#endif