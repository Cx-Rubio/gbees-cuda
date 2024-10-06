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
#include "test/gridTest.h"
#include "models.h"
#include "measurement.h"
#include "record.h"

/** Register ctrl-C handler */
static void registerSignalHandlers(void);

/** Ctrl+C handler */
static void signalHandler(int signal);

/** Print usage and exit */
static void printUsageAndExit(const char* command);

/** Execute GBEES algorithm */
static void executeGbees(bool autotest, int device);

/** Check if the number of kernel colaborative blocks fits in the GPU device */
static void checkCooperativeKernelSize(int blocks, int threads, void (*kernel)(int, Model, Global), size_t sharedMemory, int device);

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
    
    // execute GBEES algorithm
    executeGbees(autotest, device); 

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
    printf("Usage: %s {autotest}\n", command);
    exit(EXIT_SUCCESS);
}

/** Execute GBEES algorithm */
static void executeGbees(bool autotest, int device){
    // grid configuration
    int threads = THREADS_PER_BLOCK;
    int blocks = BLOCKS;
    int iterations = CELLS_PER_THREAD;    
        
    // obtain model
    Model model;
    configureLorenz3D(&model);
    int numMeasurements = model.numMeasurements;
        
    // allocate measurements memory
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
    
    // global memory for kernel
    Global global; // global memory
    global.measurements = measurementsDevice;
    global.grid = gridDevice;
    global.gridDefinition = gridDefinitionDevice;
    
    if(autotest){  // TODO remove autotest ?      
        log("Launch test kernel\n");        
        gridTest<<<1,1>>>(gridHost); 
        checkKernelError();   
    } else {            
        // check if the block count can fit in the GPU
        size_t sharedMemorySize = sizeof(double) * THREADS_PER_BLOCK + sizeof(uint32_t) * THREADS_PER_BLOCK * CELLS_PER_THREAD * 2;        
        checkCooperativeKernelSize(blocks, threads, gbeesKernel, sharedMemorySize, device);
        
        // TODO move to fn
        HANDLE_CUDA(cudaMalloc(&global.reductionArray, blocks * sizeof(double)));
        HANDLE_CUDA(cudaMalloc(&global.blockSums, blocks * 2 * sizeof(uint32_t)));
        
        log("\n -- Launch kernel with %d blocks of %d threads -- \n", blocks, threads);      
        
        void *kernelArgs[] = { &iterations, &model, &global };
        dim3 dimBlock(threads, 1, 1);
        dim3 dimGrid(blocks, 1, 1);        
        cudaLaunchCooperativeKernel((void*)gbeesKernel, dimGrid, dimBlock, kernelArgs, sharedMemorySize);
        checkKernelError();
 
        // TODO move to fn
        HANDLE_CUDA(cudaFree(global.reductionArray)); 
        HANDLE_CUDA(cudaFree(global.blockSums)); 
        
        if(model.performRecord){
            recordResult(gridDevice, &gridDefinitionHost);        
        }
    }
  
    cudaDeviceSynchronize();    

    // free device memory
    freeGridDevice(&gridHost, gridDevice);
    freeGridDefinition(gridDefinitionDevice);
    freeMeasurementsDevice(measurementsDevice);
    freeModel(&model);    
    
    // free host memory    
    freeMeasurementsHost(measurementsHost);
}

/** Check if the number of kernel colaborative blocks fits in the GPU device */
static void checkCooperativeKernelSize(int blocks, int threads, void (*kernel)(int, Model, Global), size_t sharedMemory, int device){      
    cudaDeviceProp prop;
    int numBlocksPerSm = 0;
    HANDLE_CUDA(cudaGetDeviceProperties(&prop, device));
    HANDLE_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, threads, sharedMemory));
    int maxBlocks =  prop.multiProcessorCount * numBlocksPerSm;
    
    log("- Kernel size check: intended %d blocks of %d threads, capacity %d blocks\n",blocks, threads, maxBlocks);

    if(blocks > maxBlocks){        
        handleError(GPU_ERROR, "Error: Required blocks (%d) exceed GPU capacity (%d) for cooperative kernel launch\n", blocks, maxBlocks);
    }
}
