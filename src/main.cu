// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#include <unistd.h>  
#include <stdio.h>
#include <signal.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <cuda_profiler_api.h>
#include "config.h"
#include "macro.h"
#include "device.h"
#include "kernel.h"
#include "grid.h"
#include "test/gridTest.h"
#include "models.h"
#include "measurement.h"

/** Register ctrl-C handler */
static void registerSignalHandlers(void);

/** Ctrl+C handler */
static void signalHandler(int signal);

/** Print usage and exit */
static void printUsageAndExit(const char* command);

/** Execute GBEES algorithm */
static void executeGbees(bool autotest, int measurementCount, int device);

/** Check if the number of kernel colaborative blocks fits in the GPU device */
static void checkCooperativeKernelSize(int blocks, int threads, void (*kernel)(int, GridDefinition, Model, Global), size_t sharedMemory, int device);

/**
 * @brief Main function 
 */
int main(int argc, char **argv) {  
    // autotest executes one preconfigured computation, useful for testing purposes
    bool autotest = false;

    // parameters check
    if(argc == 2){
        if(!strcmp(argv[1], "autotest" )) autotest = true;
        else printUsageAndExit(argv[0]);
    }
    if(argc != 1 && argc != 2) printUsageAndExit(argv[0]);
    
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
    int measurementCount = 2; // TODO read parameter measurement count

    // execute GBEES algorithm
    executeGbees(autotest, measurementCount, device); 

#ifdef ENABLE_LOG
        // elapsed time measurement
        clock_gettime(CLOCK_REALTIME, &end);
        double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;
#endif

    

#ifdef ENABLE_LOG
        // print elapsed time
        printf("Elapsed: %f ms\n", time_spent*1000.0);
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
}

/** Print usage and exit */
void printUsageAndExit(const char* command){
    printf("Usage: %s measuremetsFolder {autotest}\n", command);
    exit(EXIT_SUCCESS);
}

/** Execute GBEES algorithm */
static void executeGbees(bool autotest, int measurementCount, int device){
    // grid configuration
    int threads = THREADS_PER_BLOCK;
    int blocks = BLOCKS;
    int iterations = CELLS_PER_THREAD;    
        
    // obtain model
    Model model;
    configureLorenz3D(&model);
        
    // allocate measurements memory
    Measurement* measurementsHost = allocMeasurementsHost(measurementCount);
    Measurement* measurementsDevice = allocMeasurementsDevice(measurementCount);
    
    // read measurements files and copy to device
    readMeasurements(measurementsHost, &model, measurementCount);    
    printMeasurements(measurementsHost, measurementCount);
    copyHostToDeviceMeasurements(measurementsHost, measurementsDevice, measurementCount);
    
    // fill grid definition (max cells, probability threshold, center, grid width, ...) 
    GridDefinition gridDefinition;
    model.configureGrid(&gridDefinition, &measurementsHost[0]);
    gridDefinition.maxCells = threads * blocks * iterations;
    
    // allocate grid (hashtable, lists, and heap)
    Grid gridHost;
    Grid *gridDevice;        
    allocGridDevice(gridDefinition.maxCells, &gridHost, &gridDevice);
    initializeGridDevice(&gridHost, gridDevice, &gridDefinition, &measurementsHost[0]);
    
    // global memory for kernel
    Global global; // global memory
    global.measurements = measurementsDevice;
    global.grid = gridDevice;
    
    if(autotest){        
        printf("Launch test kernel\n");        
        gridTest<<<1,1>>>(gridHost);    
    } else {            
        // check if the block count can fit in the GPU
        size_t sharedMemorySize = sizeof(double) * THREADS_PER_BLOCK;
        checkCooperativeKernelSize(blocks, threads, gbeesKernel, sharedMemorySize, device);
        
        HANDLE_CUDA(cudaMalloc(&global.reductionArray, blocks * sizeof(double)));
        
#ifdef ENABLE_LOG        
        printf("\n -- Launch kernel with %d blocks of %d threads -- \n", blocks, threads);      
#endif
        
        void *kernelArgs[] = { &iterations, &gridDefinition, &model, &global };
        dim3 dimBlock(threads, 1, 1);
        dim3 dimGrid(blocks, 1, 1);        
        cudaLaunchCooperativeKernel((void*)gbeesKernel, dimGrid, dimBlock, kernelArgs, sharedMemorySize);
 
        HANDLE_CUDA(cudaFree(global.reductionArray)); 
    }
    
    // check kernel error
    cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel error: %s\n", cudaGetErrorString(err));
    }   

    cudaDeviceSynchronize();    

    // free device memory
    freeGridDevice(&gridHost, gridDevice);
    freeMeasurementsDevice(measurementsDevice);
    freeModel(&model);    
    
    // free host memory    
    freeMeasurementsHost(measurementsHost);
}

/** Check if the number of kernel colaborative blocks fits in the GPU device */
static void checkCooperativeKernelSize(int blocks, int threads, void (*kernel)(int, GridDefinition, Model, Global), size_t sharedMemory, int device){  
    // TODO add condition to avoid the check to improve performance
    cudaDeviceProp prop;
    int numBlocksPerSm = 0;
    HANDLE_CUDA(cudaGetDeviceProperties(&prop, device));
    HANDLE_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, threads, sharedMemory));
    int maxBlocks =  prop.multiProcessorCount * numBlocksPerSm;
    
#ifdef ENABLE_LOG    
    printf("- Kernel size check: intended %d blocks of %d threads, capacity %d blocks\n",blocks, threads, maxBlocks);
#endif

    if(blocks > maxBlocks){        
        handleError(GPU_ERROR, "Error: Required blocks (%d) exceed GPU capacity (%d) for cooperative kernel launch\n", blocks, maxBlocks);
    }
}