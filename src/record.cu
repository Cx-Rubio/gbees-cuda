// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#include "record.h"
#include "error.h"
#include "macro.h"

/** Record one cell */
static void recordCell(SnapshotCell* cell, FILE* fd);

/** Record one distribution */
static void recordDistribution(Snapshot* snapshotsHost, Snapshot* snapshotsDevice, Model* model, int gridSize, int nm, int nr, double threshold);

/**
 * @brief Record distributions
 * 
 * @param snapshotsHost snapshots host pointer
 * @param snapshotsDevice snapshots device pointer
 * @param model the model
 * @param grid the grid
 * @param gridDefinition grid definition
 */
void recordDistributions(Snapshot* snapshotsHost, Snapshot* snapshotsDevice, Model* model, Grid* grid, GridDefinition* gridDefinition){
    for(int nm =0; nm < model->numMeasurements; nm++) {
        for(int nr=0; nr < model->numDistRecorded; nr++) {
            recordDistribution(snapshotsHost, snapshotsDevice, model, grid->size, nm, nr, gridDefinition->threshold);
        }
    }
}

/** Record one distribution */
static void recordDistribution(Snapshot* snapshotsHost, Snapshot* snapshotsDevice, Model* model, int gridSize, int nm, int nr, double threshold){
    Snapshot snapshot;
    int index = nr + nm * model->numDistRecorded;
    
    // check if should be generated
    int mod = index % model->recordDivider;
    if(mod != model->recordSelected) return;

    // compute source index
    int snapshotIndex = index / model->recordDivider;

    // copy snapshot from device to host
    HANDLE_CUDA( cudaMemcpy( &snapshot, &snapshotsDevice[snapshotIndex], sizeof(Snapshot), cudaMemcpyDeviceToHost) );
    
    // alloc host memory
    UsedListEntry* usedList = (UsedListEntry*)malloc(gridSize * sizeof(UsedListEntry));
    SnapshotCell* heap = (SnapshotCell*)malloc(gridSize * sizeof(SnapshotCell));
    
    assertNotNull(usedList, MALLOC_ERROR, "Error allocating host memory for record distribution");
    assertNotNull(heap, MALLOC_ERROR, "Error allocating host memory for record distribution");
    
    // copy snapshot from device to host
    HANDLE_CUDA( cudaMemcpy( usedList, snapshot.usedList, gridSize * sizeof(UsedListEntry), cudaMemcpyDeviceToHost) );
    HANDLE_CUDA( cudaMemcpy( heap, snapshot.heap, gridSize * sizeof(SnapshotCell), cudaMemcpyDeviceToHost) );
    
    // output file
    char fileName[200];        
    snprintf(fileName, sizeof(fileName), "%s/P%d_pdf_%d.txt", model->pDir, nm, nr);    
    FILE* fd = fopen(fileName, "w");
    assertNotNull(fd, IO_ERROR, "Error opening output file");
        
    log("Record grid for time %f with %d cells to file %s\n", snapshot.time, snapshot.usedSize, fileName);
    
    // record time
    fprintf(fd, "%f\n", snapshot.time);
    
    // record cells
    for(uint32_t usedIndex = 0; usedIndex < snapshot.usedSize; usedIndex++){
        uint32_t heapIndex = usedList[usedIndex].heapIndex;
        SnapshotCell* cell = &heap[heapIndex];
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
static void recordCell(SnapshotCell* cell, FILE* fd){
    fprintf(fd, "%.10e", cell->prob);
    for (int i=0; i<DIM; i++) {
        fprintf(fd, " %.10e", cell->x[i]);
    }
    fprintf(fd, "\n");    
}