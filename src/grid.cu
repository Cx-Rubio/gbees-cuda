// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#include "grid.h"
#include "macro.h"
#include <string.h>
#include <cooperative_groups.h>
#include <stdint.h>

namespace cg = cooperative_groups;

/**  Private functions declaration (host) */
static void initializeHashtable(HashTableEntry* hashtable, UsedListEntry* usedList, uint32_t* initialExtent, int32_t* key, uint32_t gridSize, uint32_t* usedSizePtr, int level);
static void insertKey(int32_t* key, HashTableEntry* hashtable, UsedListEntry* usedList, uint32_t gridSize, uint32_t* usedSizePtr);

/**  Private functions declaration (device) */
static __host__ __device__ uint32_t computeHash(int32_t* state);
static __device__ bool equalState(int32_t* state1, int32_t* state2);
static __device__ void copyCell(Cell* src, Cell* dst);

/** --- Device global memory allocations  (host) --- */

/**
 * @brief Alloc grid definition in device global memory 
 * 
 * @param gridDefinitionDevice address of the pointer to the grid definition device struct
 */
void allocGridDefinitionDevice(GridDefinition** gridDefinitionDevice){
    HANDLE_CUDA( cudaMalloc(gridDefinitionDevice, sizeof(GridDefinition) ) );   
}

/**
 * @brief Free grid definition in device global memory
 *  
 * @param gridDefinitionDevice grid definition device pointer
 */
void freeGridDefinition(GridDefinition* gridDefinitionDevice){
    HANDLE_CUDA( cudaFree(gridDefinitionDevice) ); 
}

/**
 * @brief Initialize grid definition in device memory
 * 
 * @param gridDefinitionHost grid definition host pointer
 * @param gridDefinitionDevice grid definition device pointer
 */
void initializeGridDefinitionDevice(GridDefinition* gridDefinitionHost, GridDefinition* gridDefinitionDevice){
    HANDLE_CUDA( cudaMemcpy( gridDefinitionDevice , gridDefinitionHost, sizeof(GridDefinition), cudaMemcpyHostToDevice) );
}

/**
 * @brief Alloc grid in device global memory 
 * 
 * @param size maximum number of cells
 * @param grid address of the pointer to the host device struct
 * @param gridDevice address of the pointer to the grid device struct
 */
void allocGridDevice(uint32_t size, Grid* grid, Grid** gridDevice){            
    grid->size = size;
    grid->overflow = false;
    grid->usedSize = 0;
    grid->freeSize = 0;    
    HANDLE_CUDA( cudaMalloc( &grid->table, 2 * size * sizeof(HashTableEntry) ) );        
    HANDLE_CUDA( cudaMalloc( &grid->usedList, size * sizeof(UsedListEntry) ) );
    HANDLE_CUDA( cudaMalloc( &grid->freeList, size * sizeof(uint32_t) ) );
    HANDLE_CUDA( cudaMalloc( &grid->heap, size * sizeof(Cell) ) );
    
    HANDLE_CUDA( cudaMalloc( gridDevice, sizeof(Grid) ) );    
}

/**
 * @brief Free grid in device global memory
 * 
 * @param grid grid host pointer
 * @param gridDevice grid device pointer
 */
void freeGridDevice(Grid* grid, Grid* gridDevice){
     HANDLE_CUDA( cudaFree( grid->table) ); 
     HANDLE_CUDA( cudaFree( grid->usedList) ); 
     HANDLE_CUDA( cudaFree( grid->freeList) ); 
     HANDLE_CUDA( cudaFree( grid->heap) ); 
     HANDLE_CUDA( cudaFree( gridDevice) ); 
}

/**
 * @brief Initialize hashtable and free list in host and copy to device
 * 
 * @param grid grid host pointer
 * @param gridDevice grid device pointer
 * @param gridDefinition grid definition pointer
 * @param firstMeasurement first measurement
 */
void initializeGridDevice(Grid* grid, Grid* gridDevice, GridDefinition* gridDefinition, Measurement* firstMeasurement){
    uint32_t size = grid->size; // size of all the grid space (number of cells)    
    
    // compute initial grid size in each dimension
    for(int i=0;i<DIM;i++){
        grid->initialExtent[i] = (int) round(3.0 * pow(firstMeasurement->cov[i][i], 0.5) / gridDefinition->dx[i]);
    }
    
    // allocate free list in host
    uint32_t* freeListHost = (uint32_t*)malloc(size * sizeof(uint32_t));
    assertNotNull(freeListHost, MALLOC_ERROR, "Error allocating host memory for free list initialization");
    
    // compute the number of used and free cells
    int usedCells = grid->initialExtent[0] * 2 + 1; // used cells for the first dimension
    for(int i=1;i<DIM;i++){ // used cells for the other dimensions
        usedCells *= (grid->initialExtent[i] * 2 + 1);
    }
    int freeCells = size - usedCells;
    
    assertPositiveOrZero(freeCells, GRID_ERROR, "Not enough cells for grid initialization, size %d, required %d", size, usedCells);
    
    printf("\n -- Initialization --\n");
    printf("Max cells %d\n", size);
    printf("Used cells %d\n", usedCells);
    printf("Free cells %d\n", freeCells);
    
    // set free list size    
    grid->freeSize = freeCells;
    
    // fill free list in host
    for(int i=0;i<freeCells;i++){
        freeListHost[i] = size - i - 1;
    }
    
    // copy free list from host to device
    HANDLE_CUDA( cudaMemcpy( grid->freeList , freeListHost, size * sizeof(uint32_t), cudaMemcpyHostToDevice) );
    
    // free host memory
    free(freeListHost);
    
    // allocate used list in host
    UsedListEntry* usedListHost = (UsedListEntry*)malloc(size * sizeof(UsedListEntry));
    assertNotNull(usedListHost, MALLOC_ERROR, "Error allocating host memory for used list initialization");
    
    // allocate hashtable in host
    HashTableEntry* hashtableHost = (HashTableEntry*)malloc(2 * size * sizeof(HashTableEntry));
    assertNotNull(hashtableHost, MALLOC_ERROR, "Error allocating host memory for hashtable initialization");
    
    // clean hashtable memory
    memset(hashtableHost, 0, 2 * size * sizeof(HashTableEntry));
    
    // recursive initialization of the hashtable and used list 
    int32_t key[DIM];
    uint32_t usedSize = 0;
    initializeHashtable(hashtableHost, usedListHost, grid->initialExtent, key, size, &usedSize, 0);
     
    // set used list size    
    grid->usedSize = usedSize;  
    
    // copy used list from host to device
    HANDLE_CUDA( cudaMemcpy( grid->usedList , usedListHost, size * sizeof(UsedListEntry), cudaMemcpyHostToDevice) );
    
    // copy hashtable from host to device
    HANDLE_CUDA( cudaMemcpy( grid->table , hashtableHost, 2 * size * sizeof(HashTableEntry), cudaMemcpyHostToDevice) );
    
    // copy Grid root fields
    HANDLE_CUDA( cudaMemcpy( gridDevice , grid, sizeof(Grid), cudaMemcpyHostToDevice) );   
    
    // free host memory
    free(usedListHost);
    free(hashtableHost);
}

/**  --- Private functions implementation (host) ---  */

/** Recursive initialization of the hashtable and used list  */
static void initializeHashtable(HashTableEntry* hashtable, UsedListEntry* usedList, uint32_t* initialExtent, int32_t* key, uint32_t gridSize, uint32_t* usedSizePtr, int level){
    if(level == DIM){
        insertKey(key, hashtable, usedList, gridSize, usedSizePtr);
        return;
    }
    
    int span = (int)initialExtent[level];
    for(int i=-span; i<=span;i++){            
        key[level] = i;
        initializeHashtable(hashtable, usedList, initialExtent, key, gridSize, usedSizePtr, level+1);
    }
}

/** Insert a new key in the hashtable and update the used list (only for initialization) */
static void insertKey(int32_t* key, HashTableEntry* hashtable, UsedListEntry* usedList, uint32_t gridSize, uint32_t* usedSizePtr){
   uint32_t hash = computeHash(key);   
   uint32_t capacity = 2 * gridSize;
   
   for(uint32_t counter = 0; counter < capacity; counter++){
        uint32_t hashIndex = (hash + counter) % capacity;
        if(!hashtable[hashIndex].usedIndex){            
            uint32_t usedIndex = *usedSizePtr;
            
            // update hashtable
            hashtable[hashIndex].usedIndex = usedIndex + 1; // 0 is reserved to mark not used cell                        
            copyKey(key,  hashtable[hashIndex].key); 
            
            // update used list 
            usedList[usedIndex].heapIndex = usedIndex;
            usedList[usedIndex].hashTableIndex = hashIndex;
            (*usedSizePtr)++; 
            return;            
        }
    }            
} 

/**
 * @brief Copy cell key (state indexes)
 * @param src origin
 * @param dst destination
 */
__host__ __device__ void copyKey(int32_t* src, int32_t* dst){
    for(int i=0; i<DIM; i++){
        dst[i] = src[i];
    }   
}

/**  --- Private functions implementation (device) ---  */

/** Compute hash value from the state coordinates (BuzHash) */
static __host__ __device__ uint32_t computeHash(int32_t* state){    
    uint64_t hash = 0;  // Initialize the hash value
    uint64_t prime = 0x5bd1e995;  // A prime number used in the hash function

    for (int i = 0; i < DIM; ++i) {
        hash ^= (uint64_t)state[i]; // Combine the current integer with the hash
        hash *= prime;              // Multiply by a prime number
        hash ^= hash >> 47;         // Mix the bits
    }

    // Finalization step
    hash *= prime;                  // Final multiplication
    hash ^= hash >> 47;             // Final mixing step

    return hash;                    // Return the first 32 bits of the 64-bit hash value
}  

/** Check if the state coordinates are equal */
static __device__ bool equalState(int32_t* state1, int32_t* state2){
    for(int i=0; i<DIM; i++){
        if(state1[i] != state2[i]) return false;
    }    
    return true;
}

/** Copy cell contents */
static __device__ void copyCell(Cell* src, Cell* dst){
    char *d = (char *)dst;
    const char *s = (const char *)src;
    for(int i=0;i<sizeof(Cell);i++){
        d[i] = s[i];
    }    
}

/** --- Grid operations  (device)  --- */

/**
 * @brief Insert a new cell
 * 
 * @param cell new cell pointer
 * @param grid grid pointer
 */
__device__ void insertCell(Cell* cell, Grid* grid){
    
    if(grid->usedSize >= grid->size){
        grid->overflow = true;
        return;
    }
    
   uint32_t hash = computeHash(cell->state);   
   uint32_t capacity = 2 * grid->size;
   
    for(uint32_t counter = 0; counter < capacity; counter++){
        uint32_t hashIndex = (hash + counter) % capacity;
        if(!grid->table[hashIndex].usedIndex){            
            uint32_t usedIndex = grid->usedSize;
            
            // update hashtable
            grid->table[hashIndex].usedIndex = usedIndex + 1; // 0 is reserved to mark not used cell
            copyKey(cell->state,  grid->table[hashIndex].key); 
            
            // update used list 
            grid->usedList[usedIndex].heapIndex = grid->freeList[ grid->freeSize -1 ];
            grid->usedList[usedIndex].hashTableIndex = hashIndex;
            grid->usedSize++;            
            
            // update free list
            grid->freeSize--;
            
            // update heap content
            Cell* dstCell = grid->heap + grid->usedList[usedIndex].heapIndex;
            copyCell(cell, dstCell);
            return;
            }
    }            
} 


/**
 * @brief Insert a new cell (concurrent version) if not exists
 * 
 * @param cell new cell pointer
 * @param grid grid pointer
 */
__device__ void insertCellConcurrent(Cell* cell, Grid* grid){    
   uint32_t hash = computeHash(cell->state);   
   uint32_t capacity = 2 * grid->size;      
   
    for(uint32_t counter = 0; counter < capacity; counter++){
        uint32_t hashIndex = (hash + counter) % capacity;
        
        while(true){ // if the slot is reserved, wait for the other threads until obtain the final usedIndex  
        
            // check if the hashtable slot is free. If is free reserve with the UINT32_MAX value, if not free obtain the current used index
            uint32_t existingUsedIndex = atomicCAS( &grid->table[hashIndex].usedIndex, 0, UINT32_MAX);    
        
            // return if the existing cell is the same as the new cell
            if(existingUsedIndex > 0 && existingUsedIndex != UINT32_MAX){                
                if(equalState(grid->table[hashIndex].key, cell->state)) return; // if already exits, return
                else break; // continue with the next hashtable cell
            }
           
            // create a new cell
            if(!existingUsedIndex){                                    
                // check and reserve used list location
                uint32_t usedIndex = atomicAdd(&grid->usedSize, 1);   

                if(usedIndex >= grid->size){
                    grid->overflow = true;
                    return;
                }
        
                // update hashtable                
                copyKey(cell->state,  grid->table[hashIndex].key);  
                // TODO check if needed __threadfence_system(); and atomic in the next assigment
                grid->table[hashIndex].usedIndex = usedIndex + 1; // 0 is reserved to mark not used cell                
                
                // reserve one free slot and obtain its index
                uint32_t freeIndex = atomicDec(&grid->freeSize, UINT32_MAX) - 1;
                
                // update used list 
                grid->usedList[usedIndex].heapIndex = grid->freeList[ freeIndex ];
                grid->usedList[usedIndex].hashTableIndex = hashIndex;
                
                // update heap content
                Cell* dstCell = grid->heap + grid->usedList[usedIndex].heapIndex;
                
                copyCell(cell, dstCell);
                
                //printf("new cell [%d, %d, %d], usedIndex %d, hash %d hashIndex %d\n",cell->state[0],cell->state[1],cell->state[2], usedIndex, hash, hashIndex);
                        
                return;
            }
            //printf("wait for reserved slot\n");
            // TODO check in needed g.sync() to allow execution of the other blocks when all threads if this block are waiting for reserved hashtable cells
        }
    }
} 

 /**
 * @brief Delete a new cell
 * If the cell do not exists, do nothing
 * 
 * @param state state coordinates of the cell to delete
 * @param grid hash-table pointer
 */
__device__ void deleteCell(int32_t* state, Grid* grid){
   uint32_t hash = computeHash(state);   
   uint32_t capacity = 2 * grid->size;
   
   for(uint32_t counter = 0; counter < capacity; counter++){
        uint32_t hashIndex = (hash + counter) % capacity;
        if(grid->table[hashIndex].usedIndex && equalState(grid->table[hashIndex].key, state)){ // if not deleted and match state
            uint32_t usedIndex = grid->table[hashIndex].usedIndex - 1; // 0 is reserved to mark not used cell
            uint32_t usedHeap = grid->usedList[usedIndex].heapIndex;
            
            // mark the cell in the hash-table as emtpy
            grid->table[hashIndex].usedIndex = 0;
            
            // add the index to the free list
            grid->freeList[ grid->freeSize ] = usedHeap;
            grid->freeSize++;
            
            // remove the index from the used list (compact-up the table)
            for(int i=usedIndex+1;i < grid->usedSize; i++){
                uint32_t hashIndex = grid->usedList[i].hashTableIndex;
                grid->table[hashIndex].usedIndex = i; 
                grid->usedList[i-1] = grid->usedList[i];
            }
            grid->usedSize--;
            break;
        }
    }
}

/**
 * @brief Get cell by state indexes
 * Search using the hash-code
 * 
 * @param state state coordinates of the cell to find
 * @param grid grid pointer
 * @return used index stored in the hashtable (one more than the real index of the used list array) for the cell or 0 if not exists
 */
__device__ uint32_t findCell(int32_t* state, Grid* grid){
   uint32_t hash = computeHash(state);   
   uint32_t capacity = 2 * grid->size;
   
   for(uint32_t counter = 0; counter < capacity; counter++){
        uint32_t hashIndex = (hash + counter) % capacity;
        if(grid->table[hashIndex].usedIndex && equalState(grid->table[hashIndex].key, state)){ // if exists and match state
            return grid->table[hashIndex].usedIndex; // 0 is reserved to mark not used cell
        }
    } 
    return 0;
}

 /**
 * @brief Get cell by index in the used list
 * 
 * @param index index in the used list (starting with 0)
 * @param grid grid pointer
 * @return cell pointer or null if the cell is not found
 */
__device__ Cell* getCell(uint32_t index, Grid* grid){
    if(index < grid->usedSize){
        uint32_t heapIndex = grid->usedList[index].heapIndex;   
        return grid->heap + heapIndex;
    } else {
        return NULL;
    }
}

