// Copyright 2024 by Carlos Rubio, published under BSD 3-Clause License.

#include "memory.h"
#include "macro.h"

/** --- Device global memory allocations --- */

/** Alloc hash-table in device global memory */
void allocHashTableDevice(float** hashTableDevicePtr, Config config){
    HANDLE_CUDA( cudaMalloc( hashTableDevicePtr, hypersize(config) * sizeof(float) ) );
}

/** Alloc occuped list in device global memory */
void allocOccupedListDevice(float** occupedListDevicePtr, Config config){
    }

/** Alloc empty list in device global memory */
void allocEmptyListDevice(float** emptyListDevicePtr, Config config){
    }

/** Alloc grid cells in device global memory */
void allocGridDevice(float** gridDevicePtr, Config config){
    }

/** --- Device global memory de-allocations --- */

/** Free hash-table in device global memory */
void freeHashTableDevice(float* hashTableDevicePtr){
    }

/** Free occuped list in device global memory */
void freeOccupedListDevice(float* occupedListDevicePtr){
    }

/** Free empty list in device global memory */
void freeEmptyListDevice(float* emptyListDevicePtr){
    }

/** Free grid cells in device global memory */
void freeGridDevice(float* gridDevicePtr){
    }
