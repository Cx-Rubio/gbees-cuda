// Copyright 2024 by Carlos Rubio, published under BSD 3-Clause License.

#include "hashmapTest.h"
#include "../config.h"
#include <stdio.h>

/** Kernel function */
__global__ void hashTableTest(HashTable hashTable){
    
    printf("In kernel test\n");
    initializeFreeList(&hashTable);
    printHashTable(&hashTable);
    
    // Insert cells    
    Cell cell1;
    cell1.prob = 1.1;
    cell1.state[0] = -1;
    cell1.state[1] = -2;
    printf("Insert cell1\n");    
    insertCell(&cell1, &hashTable);    
    
    // Insert cells    
    Cell cell2;
    cell1.prob = 1.2;
    cell1.state[0] = 0;
    cell1.state[1] = 1;
    printf("Insert cell2\n");    
    insertCell(&cell2, &hashTable);    
    
    // Insert cells    
    Cell cell3;
    cell1.prob = 1.3;
    cell1.state[0] = 1;
    cell1.state[1] = 2;
    printf("Insert cell3\n");    
    insertCell(&cell3, &hashTable);    
    
    
    // Delete cell
    deleteCell(cell1.state, &hashTable);
    
    printHashTable(&hashTable);    
        
}


__device__ void initializeFreeList(HashTable* hashTable){
    for(int i=0;i<hashTable->size;i++){
        hashTable->freeList[i] = hashTable->size - i;
    }
    hashTable->freeSize = hashTable->size;
}

/** Print hashtable contents */
__device__ void printHashTable(HashTable* hashTable){
    printf("Hashtable:\n");
    printf("----------------\n");
    printf("Size:%d\n", hashTable->size);
    
    printf("---Map----------\n");
    for(int i=0;i<2*hashTable->size;i++){
        if(hashTable->table[i].usedIndex == 0) {
            printf("Empty register\n");
        } else {
            printf("  Key: [");
            for(int j=0;j<DIM;j++){
                printf("%d", hashTable->table[i].key[j]);
                if(j < DIM-1) printf(", ");
                }        
            printf("], Used index: %d\n", hashTable->table[i].usedIndex);            
        }
    }
    
    printf("---Used list----\n");
    for(int i=0;i<hashTable->usedSize;i++){
        printf("Heap index %d\n", hashTable->usedList[i]);
    }
    
    printf("---Free list----\n");
    for(int i=0;i<hashTable->freeSize;i++){
        printf("Heap index %d\n", hashTable->freeList[i]);
    }
    
    printf("---Heap --------\n");
    for(int i=0;i<hashTable->usedSize;i++){
        Cell *cellPtr = hashTable->heap + hashTable->usedList[i];
        printf("Cell prob %f\n", cellPtr->prob);
    }
    
    printf("----------------\n");    
}