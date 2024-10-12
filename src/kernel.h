// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#ifndef KERNEL_H
#define KERNEL_H

#include "grid.h"
#include "measurement.h"

/** Global working memory */
typedef struct {
    double* reductionArray; // global array for reduction processes
    uint32_t* threadSums; // to store scan block values
    uint32_t* blockSums; // double buffer to store scan block totals
    int32_t* blockSumsOut; // buffer selected at the end of the scan process
    Measurement* measurements;
    Grid* grid;
    GridDefinition* gridDefinition;    
} Global;

/** Time step tolerance */
#define TOL 1E-8

/** Enum to codify the direction of grid growing */
enum Direction {FORWARD=1, BACKWARD=-1};

/** --- Device global memory allocations --- */

/**
 * @brief Alloc global device memory 
 * 
 * @param global global struct pointer
 * @param blocks number of concurrent blocks
 * @param iterations number of cell processed per thread
 */
void allocGlobalDevice(Global* global, int blocks, int iterations);

/**
 * @brief Free global device memory 
 *  
 * @param global global struct pointer
 */
void freeGlobalDevice(Global* global);

/**
 * @brief Required shared memory
 * @return the required shared memory by the kernel
 */
size_t requiredSharedMemory(void);

/** Main kernel */

/** 
 * @brief Initialization kernel function 
 * 
 * @param iterations number of cells that should process the same thread
 * @param model the model
 * @param global global memory data
 */
__global__ void gbeesKernel(int iterations, Model model, Global global);


/**
 * @brief Dummy kernel to check maximum teoretical concurrent threads
 */
__global__ void dummyKernel(int iterations, Model model, Global global);

#endif