// Copyright 2024 by Carlos Rubio, published under BSD 3-Clause License.

#include "grid.h"
#include "macro.h"

/**  Private functions declaration  */
static __device__ uint32_t computeHash(int32_t* state);
static __device__ bool equalState(int32_t* state1, int32_t* state2);
static __device__ void copyKey(int32_t* src, int32_t* dst);
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
    HANDLE_CUDA( cudaMemset(grid->table, 0, 2 * size * sizeof(HashTableEntry) ) ); 
    HANDLE_CUDA( cudaMalloc( &grid->usedList, size * sizeof(uint32_t) ) );
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
 */
void initializeGridDevice(Grid* grid){
    uint32_t size = grid->size;
    
    // allocate list in host
    uint32_t* list = (uint32_t*)malloc(size * sizeof(uint32_t));    
    assertNotNull(list, MALLOC_ERROR, "Error allocating host memory for free list initialization");
    
    // fill list in host    
    for(int i=0;i<size;i++){
        list[i] = size - i - 1;
    }
    
    // copy list from host to device
    HANDLE_CUDA( cudaMemcpy( grid->freeList , list, size * sizeof(uint32_t), cudaMemcpyHostToDevice) ); 
    
    // set free list size    
    grid->freeSize = size;
    
    // free host memory
    free(list);
}

/**  --- Private functions implementation (device) ---  */

/** Compute hash value from the state coordinates */
static __device__ uint32_t computeHash(int32_t* state){
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
static __device__ void copyKey(int32_t* src, int32_t* dst){
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
 * @throws GRID_FULL_ERROR if there are not free cells
 * @throws ILLEGAL_STATE_ERROR if incosistency is detected
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

