// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#ifndef KERNEL_H
#define KERNEL_H

#include "grid.h"
#include "measurement.h"

/** Global working memory */
typedef struct {
    double* reductionArray; // global array for reduction processes
    uint32_t* blockSums; // double buffer to store scan block totals
    Measurement* measurements;
    Grid* grid;
    GridDefinition* gridDefinition;    
} Global;

/** Time step tolerance */
#define TOL 1E-8

/** Enum to codify the direction of grid growing */
enum Direction {FORWARD=1, BACKWARD=-1};

/** 
 * @brief Initialization kernel function 
 * 
 * @param iterations number of cells that should process the same thread
 * @param model the model
 * @param global global memory data
 */
__global__ void gbeesKernel(int iterations, Model model, Global global);

#endif