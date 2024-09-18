// Copyright 2024 by Carlos Rubio, published under BSD 3-Clause License.

#ifndef HASH_MAP_TEST_H
#define HASH_MAP_TEST_H

#include "../memory.h"

/** Kernel function */
__global__ void hashTableTest(HashTable hashTable);

__device__ void initializeFreeList(HashTable* hashTable);

__device__ void printHashTable(HashTable* hashTable);

#endif