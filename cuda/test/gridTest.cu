// Copyright 2024 by Carlos Rubio, published under BSD 3-Clause License.

#include "gridTest.h"
#include "../config.h"
#include <stdio.h>

/** Kernel function */
__global__ void gridTest(Grid grid){
    
    printf("$ In kernel test\n");
    initializeFreeList(&grid);
    printGrid(&grid);
    
    // Insert cells    
    Cell cell1;
    cell1.prob = 1.1;
    cell1.state[0] = -13;
    cell1.state[1] = -2;
    printf("$ Insert cell1\n");    
    insertCell(&cell1, &grid);    
    
    // Insert cells    
    Cell cell2;
    cell2.prob = 1.2;
    cell2.state[0] = 0;
    cell2.state[1] = 31;
    printf("$ Insert cell2\n");    
    insertCell(&cell2, &grid);    
    
    // Insert cells    
    Cell cell3;
    cell3.prob = 1.3;
    cell3.state[0] = 1;
    cell3.state[1] = 5;
    printf("$ Insert cell3\n");    
    insertCell(&cell3, &grid);    
    
    printGrid(&grid);    
    
    // Delete cell
    printf("$ Delete cell1\n");    
    deleteCell(cell1.state, &grid);
    // Delete cell
    printf("$ Delete cell2\n");    
    deleteCell(cell2.state, &grid);    
    // Delete cell
    printf("$ Delete cell3\n");    
    deleteCell(cell3.state, &grid);
    
    printf("$ Insert cell3\n");    
    insertCell(&cell3, &grid);    
    
    printf("$ Insert cell2\n");    
    insertCell(&cell2, &grid);    
    
    // Delete cell
    printf("$ Delete cell2\n");    
    deleteCell(cell2.state, &grid);  
    
    printGrid(&grid);    
        
}


__device__ void initializeFreeList(Grid* grid){
    for(int i=0;i<grid->size;i++){
        grid->freeList[i] = grid->size - i - 1;
    }
    grid->freeSize = grid->size;
}

/** Print grid contents */
__device__ void printGrid(Grid* grid){
    printf("Grid:\n");
    printf("----------------\n");
    printf("Size:%d\n", grid->size);
    
    printf("---Map----------\n");
    for(int i=0;i<2*grid->size;i++){
        if(grid->table[i].usedIndex == 0) {
            printf("Empty register\n");
        } else {
            printf("  Key: [");
            for(int j=0;j<DIM;j++){
                printf("%d", grid->table[i].key[j]);
                if(j < DIM-1) printf(", ");
                }        
            printf("], Used index: %d [%d]\n", grid->table[i].usedIndex, grid->table[i].usedIndex-1);            
        }
    }
    
    printf("---Used list----\n");
    for(int i=0;i<grid->usedSize;i++){
        printf("Hash index %d, Heap index %d\n", grid->usedList[i].hashTableIndex, grid->usedList[i].heapIndex);
    }
    
    printf("---Free list----\n");
    for(int i=0;i<grid->freeSize;i++){
        printf("Heap index %d\n", grid->freeList[i]);
    }
    
    printf("---Heap --------\n");
    for(int i=0;i<grid->usedSize;i++){
        Cell *cellPtr = grid->heap + grid->usedList[i].heapIndex;
        //printf("Cell addr %p\n", cellPtr);
        printf("Cell prob %f\n", cellPtr->prob);
    }
    
    printf("----------------\n");    
}