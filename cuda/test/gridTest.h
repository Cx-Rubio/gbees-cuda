// Copyright 2024 by Carlos Rubio, published under BSD 3-Clause License.

#ifndef HASH_MAP_TEST_H
#define HASH_MAP_TEST_H

#include "../grid.h"

/** Kernel function */
__global__ void gridTest(Grid grid);

__device__ void initializeFreeList(Grid* grid);

__device__ void printGrid(Grid* grid);

#endif