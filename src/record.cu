// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#include "record.h"
#include "error.h"
#include "macro.h"

/** Record one cell */
static void recordCell(Cell* cell, FILE* fd);

/** Record one distribution */
static void recordDistribution(Snapshoot* snapshootsHost, Snapshoot* snapshootsDevice, Model* model, int gridSize, int nm, int nr, double threshold);

/**
 * @brief Record distributions
 * 
 * @param snapshootsHost snapshoots host pointer
 * @param snapshootsDevice snapshoots device pointer
 * @param model the model
 * @param grid the grid
 * @param gridDefinition grid definition
 */
void recordDistributions(Snapshoot* snapshootsHost, Snapshoot* snapshootsDevice, Model* model, Grid* grid, GridDefinition* gridDefinition){
    for(int nm =0; nm < model->numMeasurements; nm++) {
        for(int nr=0; nr < model->numDistRecorded; nr++) {
            recordDistribution(snapshootsHost, snapshootsDevice, model, grid->size, nm, nr, gridDefinition->threshold);
        }
    }
}

/** Record one distribution */
static void recordDistribution(Snapshoot* snapshootsHost, Snapshoot* snapshootsDevice, Model* model, int gridSize, int nm, int nr, double threshold){
    Snapshoot snapshoot;
    int index = nr + nm * model->numDistRecorded;
    
    // check if should be generated
    int mod = index % model->recordDivider;
    if(mod != model->recordSelected) return;

    // compute source index
    int snapshootIndex = index / model->recordDivider;

    // copy snapshoot from device to host
    HANDLE_CUDA( cudaMemcpy( &snapshoot, &snapshootsDevice[snapshootIndex], sizeof(Snapshoot), cudaMemcpyDeviceToHost) );
    
    // alloc host memory
    UsedListEntry* usedList = (UsedListEntry*)malloc(gridSize * sizeof(UsedListEntry));
    Cell* heap = (Cell*)malloc(gridSize * sizeof(Cell));
    
    assertNotNull(usedList, MALLOC_ERROR, "Error allocating host memory for record distribution");
    assertNotNull(heap, MALLOC_ERROR, "Error allocating host memory for record distribution");
    
    // copy snapshoot from device to host
    HANDLE_CUDA( cudaMemcpy( usedList, snapshoot.usedList, gridSize * sizeof(UsedListEntry), cudaMemcpyDeviceToHost) );
    HANDLE_CUDA( cudaMemcpy( heap, snapshoot.heap, gridSize * sizeof(Cell), cudaMemcpyDeviceToHost) );
    
    // output file
    char fileName[200];        
    snprintf(fileName, sizeof(fileName), "%s/P%d_pdf_%d.txt", model->pDir, nm, nr);    
    FILE* fd = fopen(fileName, "w");
    assertNotNull(fd, IO_ERROR, "Error opening output file");
        
    log("Record grid for time %f with %d cells to file %s\n", snapshoot.time, snapshoot.usedSize, fileName);
    
    // record time
    fprintf(fd, "%f\n", snapshoot.time);
    
    // record cells
    for(uint32_t usedIndex = 0; usedIndex < snapshoot.usedSize; usedIndex++){
        uint32_t heapIndex = usedList[usedIndex].heapIndex;
        Cell* cell = &heap[heapIndex];
        if(cell->prob > threshold){
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