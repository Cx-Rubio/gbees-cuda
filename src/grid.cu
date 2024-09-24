// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#include "grid.h"
#include "macro.h"
#include <string.h>

/**  Private functions declaration (host) */
static void initializeHashtable(HashTableEntry* hashtable, UsedListEntry* usedList, uint32_t* initialExtent, int32_t* key, uint32_t gridSize, uint32_t* usedSizePtr, int level);
static void insertKey(int32_t* key, HashTableEntry* hashtable, UsedListEntry* usedList, uint32_t gridSize, uint32_t* usedSizePtr);

/**  Private functions declaration (device) */
static __host__ __device__ uint32_t computeHash(int32_t* state);
static __device__ bool equalState(int32_t* state1, int32_t* state2);
static __host__ __device__ void copyKey(int32_t* src, int32_t* dst);
static __device__ void copyCell(Cell* src, Cell* dst);

/** --- Device global memory allocations  (host) --- */

/**
 * @brief Alloc grid in device global memory
 * requires grid->size be already filled
 * 
 * @param grid grid pointer with the field size already filled
 */
void allocGridDevice(Grid* grid){    
    uint32_t size = grid->size;
    grid->overflow = false;
    grid->usedSize = 0;
    grid->freeSize = 0;
    HANDLE_CUDA( cudaMalloc( &grid->table, 2 * size * sizeof(HashTableEntry) ) );
    HANDLE_CUDA( cudaMalloc( &grid->usedList, size * sizeof(UsedListEntry) ) );
    HANDLE_CUDA( cudaMalloc( &grid->freeList, size * sizeof(uint32_t) ) );
    HANDLE_CUDA( cudaMalloc( &grid->heap, size * sizeof(Cell) ) );
}

/**
 * @brief Free grid in device global memory
 * 
 * @param grid grid pointer
 */
void freeGridDevice(Grid* grid){
     HANDLE_CUDA( cudaFree( grid->table) ); 
     HANDLE_CUDA( cudaFree( grid->usedList) ); 
     HANDLE_CUDA( cudaFree( grid->freeList) ); 
     HANDLE_CUDA( cudaFree( grid->heap) ); 
}

/**
 * @brief Initialize hashtable and free list in host and copy to device
 * 
 * @param grid grid pointer
 * @param gridDefinition grid definition pointer
 * @param firstMeasurement first measurement
 */
void initializeGridDevice(Grid* grid, GridDefinition* gridDefinition, Measurement* firstMeasurement){
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
    HashTableEntry* hashtableHost = (HashTableEntry*)malloc(size * sizeof(HashTableEntry));
    assertNotNull(hashtableHost, MALLOC_ERROR, "Error allocating host memory for hashtable initialization");
    
    // clean hashtable memory
    memset(hashtableHost, 0, size * sizeof(HashTableEntry));
    
    // recursive initialization of the hashtable and used list 
    int32_t key[DIM];
    uint32_t usedSize = 0;
    initializeHashtable(hashtableHost, usedListHost, grid->initialExtent, key, size, &usedSize, 0);
     
    // set used list size    
    grid->usedSize = usedSize;  
    
    // copy used list from host to device
    HANDLE_CUDA( cudaMemcpy( grid->usedList , usedListHost, size * sizeof(UsedListEntry), cudaMemcpyHostToDevice) );
    
    // copy hashtable from host to device
    HANDLE_CUDA( cudaMemcpy( grid->table , hashtableHost, size * sizeof(HashTableEntry), cudaMemcpyHostToDevice) );
    
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


/**  --- Private functions implementation (device) ---  */

/** Compute hash value from the state coordinates */
static __host__ __device__ uint32_t computeHash(int32_t* state){
    uint32_t hash = 0;
    for(int i=0;i<DIM;i++) hash ^= state[i];    
    return hash;
} // TODO improve hash computation if needed

/** Check if the state coordinates are equal */
static __device__ bool equalState(int32_t* state1, int32_t* state2){
    for(int i=0; i<DIM; i++){
        if(state1[i] != state2[i]) return false;
    }    
    return true;
}

/** Copy key */
static __host__ __device__ void copyKey(int32_t* src, int32_t* dst){
    for(int i=0; i<DIM; i++){
        dst[i] = src[i];
    }   
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
 * @brief Get cell by state position
 * Search using the hash-code
 * 
 * @param state state coordinates of the cell to find
 * @param grid grid pointer
 * @return cell pointer or null if the cell is not found
 */
__device__ Cell* findCell(int32_t* state, Grid* grid){
   uint32_t hash = computeHash(state);   
   uint32_t capacity = 2 * grid->size;
   
   for(uint32_t counter = 0; counter < capacity; counter++){
        uint32_t hashIndex = (hash + counter) % capacity;
        if(grid->table[hashIndex].usedIndex && equalState(grid->table[hashIndex].key, state)){ // if not deleted and match state
            uint32_t usedIndex = grid->table[hashIndex].usedIndex - 1; // 0 is reserved to mark not used cell
            uint32_t heapIndex = grid->usedList[usedIndex].heapIndex;
            return grid->heap + heapIndex;
            }
    }    
    return NULL;
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

