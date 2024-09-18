// Copyright 2024 by Carlos Rubio, published under BSD 3-Clause License.

#ifndef MEMORY_H
#define MEMORY_H

#include <stdint.h>
#include "config.h"
#include "../gbees/gbees.h"

/** Hash table entry */
typedef struct {
    int32_t  key[DIM];
    uint32_t  usedIndex;    
    } HashTableEntry;

/** Hash table type */
typedef struct {
    uint32_t size;
    HashTableEntry* table;         // device pointer
    uint32_t usedSize;
    uint32_t* usedList;   // device pointer (indexed from [1] as 0 is the null value)
    uint32_t freeSize;
    uint32_t* freeList;    // device pointer (indexed from [1] as 0 is the null value)  
    Cell* heap;     // device pointer (indexed from [1] as 0 is the null value)
    } HashTable;


/** --- Device global memory allocations --- */

/** Alloc hash-table in device global memory */
void allocHashTableDevice(HashTable* hashTable);

/** --- Device global memory de-allocations --- */

/** Free hash-table in device global memory */
void freeHashTableDevice(HashTable* hashTable);


__host__ __device__ void insertCell(Cell* cell, HashTable* hashTable);  

__host__ __device__ void deleteCell(int32_t* state, HashTable* hashTable);

__host__ __device__ Cell* findCell(int32_t* state, HashTable* hashTable);

__host__ __device__ Cell* getCell(uint32_t index, HashTable* hashTable);


#endif