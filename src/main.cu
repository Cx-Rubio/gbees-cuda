// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#include <unistd.h>  
#include <stdio.h>
#include <signal.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include "config.h"
#include "macro.h"
#include "device.h"
#include "kernel.h"
#include "grid.h"
#include "measurement.h"
#include "record.h"
#include "models.h"
#include "models/lorenz3D.h"
#include "models/pcr3bp.h"
#include "models/cr3bp.h"
#include "models/lorenz6D.h"

/** Register ctrl-C handler */
static void registerSignalHandlers(void);

/** Ctrl+C handler */
static void signalHandler(int signal);

/** Print usage and exit */
static void printUsageAndExit(const char* command);

/** Execute GBEES algorithm */
static void executeGbees(int device);

/** Check if the number of kernel colaborative blocks fits in the GPU device */
static void checkCooperativeKernelSize(int blocks, int threads, void (*kernel)(int, Model, Global, Snapshot*), size_t sharedMemory, int device);

/**
 * @brief Main function 
 */
int main(int argc, char **argv) {      
    // parameters check    
    if(argc != 1) printUsageAndExit(argv[0]);
    
    // manage ctrl+C
    registerSignalHandlers();

    // select and print device info
    int device = selectBestDevice();
    printDevice(device);
    HANDLE_CUDA(cudaSetDevice(device) );  


#ifdef ENABLE_LOG
        // elapsed time measurement
        struct timespec start, end;
        clock_gettime(CLOCK_REALTIME, &start); // start time measurement
#endif
    
    // execute GBEES algorithm    
    executeGbees(device);     

#ifdef ENABLE_LOG
        // elapsed time measurement
        clock_gettime(CLOCK_REALTIME, &end);
        double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;
        // print elapsed time
        log("Elapsed: %f ms\n", time_spent*1000.0);
#endif
             
    return EXIT_SUCCESS;
}

/** Register ctrl-C handler */
void registerSignalHandlers(){        
    struct sigaction action;
    action.sa_handler = signalHandler;    
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;    
    sigaction(SIGINT, &action, NULL);
    sigaction(SIGTERM, &action, NULL);
    sigaction(SIGQUIT, &action, NULL);    
}

/** On ctrl+C */ 
void signalHandler(int signal){ 
    exit(EXIT_SUCCESS);       
}

/** Print usage and exit */
void printUsageAndExit(const char* command){
    printf("Usage: %s\n", command);
    exit(EXIT_SUCCESS);
}

/** Execute GBEES algorithm */
static void executeGbees(int device){
    // grid configuration
    int threads = THREADS_PER_BLOCK;
    int blocks = BLOCKS;
    int iterations = CELLS_PER_THREAD;
    
    Model model;
    
    // configure model
    configureLorenz3D(&model);
    //configurePcr3bp(&model);
    //configureCr3bp(&model);
    //configureLorenz6D(&model);
    
    // allocate measurements memory
    int numMeasurements = model.numMeasurements;
    Measurement* measurementsHost = allocMeasurementsHost(numMeasurements);
    Measurement* measurementsDevice = allocMeasurementsDevice(numMeasurements);
    
    // read measurements files and copy to device
    readMeasurements(measurementsHost, model.mDim, model.mDir, numMeasurements); 
#ifdef ENABLE_LOG   
    printMeasurements(measurementsHost, numMeasurements);
#endif
    copyHostToDeviceMeasurements(measurementsHost, measurementsDevice, numMeasurements);
    
    // fill grid definition (max cells, probability threshold, center, grid width, ...) 
    GridDefinition gridDefinitionHost;
    GridDefinition *gridDefinitionDevice;
    model.configureGrid(&gridDefinitionHost, measurementsHost);
    gridDefinitionHost.maxCells = threads * blocks * iterations;
    allocGridDefinitionDevice(&gridDefinitionDevice);
    initializeGridDefinitionDevice(&gridDefinitionHost, gridDefinitionDevice);
    
    // allocate grid (hashtable, lists, and heap)
    Grid gridHost;
    Grid *gridDevice;        
    allocGridDevice(gridDefinitionHost.maxCells, &gridHost, &gridDevice);
    initializeGridDevice(&gridHost, gridDevice, &gridDefinitionHost, &measurementsHost[0]);
    
    // allocate snapshots
    Snapshot *snapshotsHost; // host
    Snapshot *snapshotsDevice; // device
    allocSnapshotsHost(&snapshotsHost, &model);
    allocSnapshotsDevice(gridDefinitionHost.maxCells, snapshotsHost, &snapshotsDevice, &model);
    initializeSnapshotsDevice(snapshotsHost, snapshotsDevice, &model);
    
    // global memory for kernel
    Global global; // global memory
    global.measurements = measurementsDevice;
    global.grid = gridDevice;
    global.gridDefinition = gridDefinitionDevice;
    allocGlobalDevice(&global, blocks, iterations);
        
    // check if the block count can fit in the GPU
    size_t staticSharedMemory = requiredSharedMemory();
    size_t dynamicSharedMemory = 0;
    log("Required shared memory: static: %lu, dynamic %lu\n", staticSharedMemory, dynamicSharedMemory);
    checkCooperativeKernelSize(blocks, threads, gbeesKernel, dynamicSharedMemory, device);
    
    log("\n -- Launch kernel with %d blocks of %d threads -- \n", blocks, threads);      
    
    void *kernelArgs[] = { &iterations, &model, &global, &snapshotsDevice};
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);        
    cudaLaunchCooperativeKernel((void*)gbeesKernel, dimGrid, dimBlock, kernelArgs, dynamicSharedMemory);
    checkKernelError();
    
    if(model.performRecord){
        recordDistributions(snapshotsHost, snapshotsDevice, &model, &gridHost, &gridDefinitionHost);
    }
  
    cudaDeviceSynchronize();    

    // free device memory    
    freeSnapshotsDevice(snapshotsHost, snapshotsDevice, &model);
    freeSnapshotsHost(snapshotsHost, &model);
    freeGridDevice(&gridHost, gridDevice);    
    freeGridDefinition(gridDefinitionDevice);
    freeMeasurementsDevice(measurementsDevice);
    freeModel(&model);  
    freeGlobalDevice(&global);   
    
    // free host memory    
    freeMeasurementsHost(measurementsHost);
}

/** Check if the number of kernel colaborative blocks fits in the GPU device */
static void checkCooperativeKernelSize(int blocks, int threads, void (*kernel)(int, Model, Global, Snapshot*), size_t dynamicsSharedMemory, int device){
    cudaDeviceProp prop;
    int numBlocksPerSm = 0;
    int numBlocksPerSmLimit = 0;
    HANDLE_CUDA(cudaGetDeviceProperties(&prop, device));
    HANDLE_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, threads, dynamicsSharedMemory));
    HANDLE_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSmLimit, dummyKernel, threads, dynamicsSharedMemory));
    int maxBlocks =  prop.multiProcessorCount * numBlocksPerSm;
    int limitBlocks = prop.multiProcessorCount * numBlocksPerSmLimit;
    
    log("- Kernel size check: intended %d blocks of %d threads, capacity %d blocks (limit for small kernel %d)\n",blocks, threads, maxBlocks, limitBlocks);

    if(blocks > maxBlocks){        
        handleError(GPU_ERROR, "Error: Required blocks (%d) exceed GPU capacity (%d) for cooperative kernel launch\n", blocks, maxBlocks);
    }
}
