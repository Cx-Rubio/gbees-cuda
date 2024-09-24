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
static void executeGbees(bool autotest, int measurementCount);

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
    executeGbees(autotest, measurementCount); 

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
static void executeGbees(bool autotest, int measurementCount){
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
    
    // allocate grid (hashtable, lists, and heap)
    Grid grid;
    grid.size = gridDefinition.maxCells;
    allocGridDevice(&grid);
    initializeGridDevice(&grid, &gridDefinition, &measurementsHost[0]);
    
    if(autotest){
        int blocks = 1;
        int threads = 1;
        
        printf("Launch test kernel\n");
        
        gridTest<<<blocks,threads>>>(grid);    
    } else {
        int threads = THREADS_PER_BLOCK;
        int blocks = (grid.usedSize+threads-1)/threads;
        
        printf("\n -- Launch initialization kernel with %d blocks of %d threads -- \n", blocks, threads);
        initializationKernel<<<blocks,threads>>>(gridDefinition, grid, model, measurementsDevice);
    }
    
    // check kernel error
    cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel error: %s\n", cudaGetErrorString(err));
    }   

    cudaDeviceSynchronize();    

    // free device memory
    freeGridDevice(&grid);
    freeMeasurementsDevice(measurementsDevice);
    freeModel(&model);
    
    // free host memory    
    freeMeasurementsHost(measurementsHost);
    
    
}