// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#ifndef GRID_H
#define GRID_H

#include <stdint.h>
#include "config.h"
#include "measurement.h"

/** Null reference for i and k nodes and used index in hashtable */
#define NULL_REFERENCE UINT32_MAX

/** Reserved hashtable slot */
#define RESERVED UINT32_MAX-1

/** Grid definition */
typedef struct __align__(8) {
    int maxCells;        
    double center[DIM];
    double dx[DIM];
    double dt;
    double threshold;    
    double hi_bound;
    double lo_bound;
} GridDefinition;

/** Cell definition */
typedef struct Cell Cell;
struct __align__(8) Cell {
    bool deleted;
    double prob;
    double v[DIM];
    double ctu[DIM];
    int32_t state[DIM];
    double x[DIM];
    uint32_t iNodes[DIM];
    uint32_t kNodes[DIM];
    double dcu;
    double cfl_dt;
    double bound_val;    
};

/** Hash table entry */
typedef struct __align__(4) {
    int32_t key[DIM];
    uint32_t usedIndex;    
    uint32_t hashIndex;        
    } HashTableEntry;

/** Used list entry */
typedef struct __align__(4) {
    uint32_t heapIndex;
    uint32_t hashTableIndex;
} UsedListEntry;

/** Grid data structure */
typedef struct __align__(8) {
    bool overflow;
    uint32_t size;
    uint32_t initialExtent[DIM];
    HashTableEntry* table; 
    HashTableEntry* tableTemp;  // double buffer for compact
    uint32_t usedSize;
    UsedListEntry* usedList; 
    UsedListEntry* usedListTemp; // double buffer for compact
    uint32_t freeSize;
    uint32_t* freeList; 
    Cell* heap; 
    uint32_t* scanBuffer; // buffer to perform exclusive-scan in the prune operation
    } Grid;

/** --- Device global memory allocations --- */

/**
 * @brief Alloc grid definition in device global memory 
 * 
 * @param gridDefinitionDevice address of the pointer to the grid definition device struct
 */
void allocGridDefinitionDevice(GridDefinition** gridDefinitionDevice);

/**
 * @brief Free grid definition in device global memory
 *  
 * @param gridDefinitionDevice grid definition device pointer
 */
void freeGridDefinition(GridDefinition* gridDefinitionDevice);

/**
 * @brief Initialize grid definition in device memory
 * 
 * @param gridDefinitionHost grid definition host pointer
 * @param gridDefinitionDevice grid definition device pointer
 */
void initializeGridDefinitionDevice(GridDefinition* gridDefinitionHost, GridDefinition* gridDefinitionDevice);

/**
 * @brief Alloc grid in device global memory 
 * 
 * @param size maximum number of cells
 * @param grid address of the pointer to the host device struct
 * @param gridDevice address of the pointer to the grid device struct
 */
void allocGridDevice(uint32_t size, Grid* grid, Grid** gridDevice);

/**
 * @brief Free grid in device global memory
 * 
 * @param grid grid host pointer
 * @param gridDevice grid device pointer
 */
void freeGridDevice(Grid* grid, Grid* gridDevice);

/**
 * @brief Initialize hashtable and free list in host and copy to device
 * 
 * @param grid grid host pointer
 * @param gridDevice grid device pointer
 * @param gridDefinition grid definition pointer
 * @param firstMeasurement first measurement
 */
void initializeGridDevice(Grid* grid, Grid* gridDevice, GridDefinition* gridDefinition, Measurement* firstMeasurement);

/**
 * @brief Insert a new cell
 * 
 * @param cell new cell pointer
 * @param grid grid pointer
 */
__device__ void insertCell(Cell* cell, Grid* grid);  

/**
 * @brief Insert a new hash entry (concurrent version)
 * do not checks existence with previous cells
 * 
 * @param hashEntry new hash table entry pointer
 * @param table table pointer
 * @param grid grid pointer
 */
__device__ void insertHashConcurrent(HashTableEntry* hashEntry, HashTableEntry* table, Grid* grid);

/**
 * @brief Insert a new cell (concurrent version) if not exists
 * 
 * @param cell new cell pointer
 * @param grid grid pointer
 */
__device__ void insertCellConcurrent(Cell* cell, Grid* grid);

 /**
 * @brief Delete a new cell
 * If the cell do not exists, do nothing
 * 
 * @param state state coordinates of the cell to delete
 * @param grid hash-table pointer
 */
__device__ void deleteCell(int32_t* state, Grid* grid);

/**
 * @brief Get cell by state indexes
 * Search using the hash-code
 * 
 * @param state state coordinates of the cell to find
 * @param grid grid pointer
 * @return used index for the cell or NULL_REFERENCE if not exists
 */
__device__ uint32_t findCell(int32_t* state, Grid* grid);

 /**
 * @brief Get cell by index in the used list
 * 
 * @param index index in the used list (starting with 0)
 * @param grid grid pointer
 * @return cell pointer or null if the cell is not found
 */
__device__ Cell* getCell(uint32_t index, Grid* grid);

/**
 * @brief Copy cell key (state indexes)
 * @param src origin
 * @param dst destination
 */
__host__ __device__ void copyKey(int32_t* src, int32_t* dst);

#endif