// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#include "record.h"
#include "error.h"
#include "macro.h"

/** Record one cell */
static void recordCell(Cell* cell, FILE* fd);

/**
 * @brief Record result
 * 
 * @param grid 
 * @param gridDefinition
 */
void recordResult(Grid* gridDevice, GridDefinition* gridDefinition){
    
    Grid grid;
    
    // copy grid information to host
    HANDLE_CUDA( cudaMemcpy( &grid, gridDevice, sizeof(Grid), cudaMemcpyDeviceToHost) );
        
    int size = grid.size;
    
    // alloc host memory
    UsedListEntry* usedList = (UsedListEntry*)malloc(size * sizeof(UsedListEntry));
    Cell* heap = (Cell*)malloc(size * sizeof(Cell));    
    assertNotNull(usedList, MALLOC_ERROR, "Error allocating host memory for result record");
    assertNotNull(heap, MALLOC_ERROR, "Error allocating host memory for result record");
    
    // copy device to host
    HANDLE_CUDA( cudaMemcpy( usedList, grid.usedList, size * sizeof(UsedListEntry), cudaMemcpyDeviceToHost) );
    HANDLE_CUDA( cudaMemcpy( heap, grid.heap, size * sizeof(Cell), cudaMemcpyDeviceToHost) );
    
    // output file
    FILE* fd = fopen(RESULT_FILE_NAME, "w");
    assertNotNull(fd, IO_ERROR, "Error opening output file");
    
    log("Record grid with %d cells to file %s\n", grid.usedSize, RESULT_FILE_NAME);
    
    for(uint32_t usedIndex = 0; usedIndex < grid.usedSize; usedIndex++){
        uint32_t heapIndex = usedList[usedIndex].heapIndex;
        Cell* cell = &heap[heapIndex];
        if(cell->prob > gridDefinition->threshold){
            recordCell(&heap[heapIndex], fd);        
        }
    }
    
    fclose(fd);
    
    // free host memory
    free(usedList);
    free(heap);
}

/** Record one cell */
static void recordCell(Cell* cell, FILE* fd){    
    fprintf(fd, "%.10e", cell->prob);
    for (int i=0; i<DIM; i++) {
        fprintf(fd, " %.10e", cell->x[i]);
    }
    fprintf(fd, "\n");    
}