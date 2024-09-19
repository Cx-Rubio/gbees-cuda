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

/** Used list entry */
typedef struct {
    uint32_t heapIndex;
    uint32_t hashTableIndex;
} UsedListEntry;

/** Grid data structure */
typedef struct {
    bool overflow;
    uint32_t size;
    HashTableEntry* table; 
    uint32_t usedSize;
    UsedListEntry* usedList; 
    uint32_t freeSize;
    uint32_t* freeList; 
    Cell* heap; 
    } Grid;

/** --- Device global memory allocations --- */

/**
 * @brief Alloc grid in device global memory
 * requires grid->size be already filled
 * 
 * @param grid grid pointer with the field size already filled
 */
void allocGridDevice(Grid* grid);

/**
 * @brief Free grid in device global memory
 * 
 * @param grid grid pointer
 */
void freeGridDevice(Grid* grid);

/**
 * @brief Insert a new cell
 * @throws GRID_FULL_ERROR if there are not free cells
 * @throws ILLEGAL_STATE_ERROR if incosistency is detected
 * 
 * @param cell new cell pointer
 * @param grid grid pointer
 */
__device__ void insertCell(Cell* cell, Grid* grid);  

 /**
 * @brief Delete a new cell
 * If the cell do not exists, do nothing
 * 
 * @param state state coordinates of the cell to delete
 * @param grid hash-table pointer
 */
__device__ void deleteCell(int32_t* state, Grid* grid);

 /**
 * @brief Get cell by state position
 * Search using the hash-code
 * 
 * @param state state coordinates of the cell to find
 * @param grid grid pointer
 * @return cell pointer or null if the cell is not found
 */
__device__ Cell* findCell(int32_t* state, Grid* grid);

 /**
 * @brief Get cell by index in the used list
 * 
 * @param index index in the used list (starting with 0)
 * @param grid grid pointer
 * @return cell pointer or null if the cell is not found
 */
__device__ Cell* getCell(uint32_t index, Grid* grid);


#endif