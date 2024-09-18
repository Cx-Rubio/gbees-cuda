// Copyright 2024 by Carlos Rubio, published under BSD 3-Clause License.

#include "memory.h"
#include "macro.h"

/** --- Private functions declaration --- */
static __device__ uint32_t computeHash(int32_t* state);
static __device__ bool equalState(int32_t* state1, int32_t* state2);
static __device__ void copyKey(int32_t* src, int32_t* dst);
static __device__ void copyCell(Cell* src, Cell* dst);



/** --- Device global memory allocations --- */

/** Alloc hash-table in device global memory */
void allocHashTableDevice(HashTable* hashTable, Grid* grid){
    hashTable->size = grid->maxCells;
    hashTable->usedSize = 0;
    hashTable->freeSize = 0;
    HANDLE_CUDA( cudaMalloc( &hashTable->table, 2 * hashTable->size * sizeof(HashTableEntry) ) );
    HANDLE_CUDA( cudaMemset(hashTable->table, 0, 2 * hashTable->size * sizeof(HashTableEntry) ) ); 
    HANDLE_CUDA( cudaMalloc( &hashTable->usedList, hashTable->size * sizeof(uint32_t) ) );
    HANDLE_CUDA( cudaMalloc( &hashTable->freeList, hashTable->size * sizeof(uint32_t) ) ); // TODO intialize freeList in kernel 1, 2, 3, ...
    HANDLE_CUDA( cudaMalloc( &hashTable->heap, hashTable->size * sizeof(Cell) ) );
}

/** --- Device global memory de-allocations --- */

/** Free hash-table in device global memory */
void freeHashTableDevice(HashTable* hashTable){
     HANDLE_CUDA( cudaFree( hashTable->table) ); 
     HANDLE_CUDA( cudaFree( hashTable->usedList) ); 
     HANDLE_CUDA( cudaFree( hashTable->freeList) ); 
     HANDLE_CUDA( cudaFree( hashTable->heap) ); 
}


// Compute hash value from the state coordinates TODO
static __device__ uint32_t computeHash(int32_t* state){
    uint32_t hash = 0;
    for(int i=0;i<DIM;i++) hash ^= state[i];
    return hash;
}

static __device__ bool equalState(int32_t* state1, int32_t* state2){
    for(int i=0; i<DIM; i++){
        if(state1[i] != state2[i]) return false;
    }    
    return true;
}

// Copy hashtable key
static __device__ void copyKey(int32_t* src, int32_t* dst){
    for(int i=0; i<DIM; i++){
        dst[i] = src[i];
    }   
}

// Copy Cell
static __device__ void copyCell(Cell* src, Cell* dst){
    char *d = (char *)dst;
    const char *s = (const char *)src;
    for(int i=0;i<sizeof(Cell);i++){
        d[i] = s[i];
    }    
}

/** Hashtable operations  (device) */

__device__ void insertCell(Cell* cell, HashTable* hashTable){
    if(hashTable->usedSize >= hashTable->size){
        // TODO launch GRID_FULL_ERROR
        }
    
   uint32_t hash = computeHash(cell->state);   
   uint32_t capacity = 2 * hashTable->size;
   
    for(uint32_t counter = 0; counter < capacity; counter++){
        uint32_t hashIndex = (hash + counter) % capacity;
        if(!hashTable->table[hashIndex].usedIndex){
            uint32_t usedIndex = hashTable->usedSize;
            hashTable->table[hashIndex].usedIndex = usedIndex + 1; // 0 is reserved to mark not used cell
            copyKey(cell->state,  hashTable->table[hashIndex].key); 
            hashTable->usedList[usedIndex] = hashTable->freeList[ hashTable->freeSize -1 ];
            Cell* dstCell = hashTable->heap + hashTable->usedList[usedIndex];
            copyCell(cell, dstCell);
            hashTable->freeSize--;
            hashTable->usedSize++;            
            return;
            }
    }    
    // TODO launch ILLEGAL_STATE_ERROR   
}

__device__ void deleteCell(int32_t* state, HashTable* hashTable){
   uint32_t hash = computeHash(state);   
   uint32_t capacity = 2 * hashTable->size;
   
   for(uint32_t counter = 0; counter < capacity; counter++){
        uint32_t hashIndex = (hash + counter) % capacity;
        if(equalState(hashTable->table[hashIndex].key, state) ){
            uint32_t usedIndex = hashTable->table[hashIndex].usedIndex - 1; // 0 is reserved to mark not used cell            
            
            // mark the cell in the hash-table as emtpy
            hashTable->table[hashIndex].usedIndex = 0;
            
            // add the index to the free list
            hashTable->freeList[ hashTable->freeSize ] = usedIndex;
            hashTable->freeSize++;
            
            // remove the index from the used list (compact-up the table)
            for(int i=usedIndex+1;i < hashTable->usedSize; i++){
                hashTable->usedList[i-1] = hashTable->usedList[i];
            }
            hashTable->usedSize--;  
            break;                                        
        }
    }        
}

/** Get cell by grid position (hashcode from the table) */
__device__ Cell* findCell(int32_t* state, HashTable* hashTable){
   uint32_t hash = computeHash(state);   
   uint32_t capacity = 2 * hashTable->size;
   
   for(uint32_t counter = 0; counter < capacity; counter++){
        uint32_t hashIndex = (hash + counter) % capacity;
        if(equalState(hashTable->table[hashIndex].key, state) ){
            uint32_t usedIndex = hashTable->table[hashIndex].usedIndex - 1; // 0 is reserved to mark not used cell
            uint32_t heapIndex = hashTable->usedList[usedIndex];
            return hashTable->heap + heapIndex;
            }
    }    
    return NULL;  
}

/** Get cell by index from the used list */
__device__ Cell* getCell(uint32_t index, HashTable* hashTable){
    if(index < hashTable->usedSize){
        uint32_t heapIndex = hashTable->usedList[index];
        return hashTable->heap + heapIndex;
    } else {
        return NULL;  
    }
}

