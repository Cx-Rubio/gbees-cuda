// Copyright 2024 Carlos Rubio, published under BSD 3-Clause License.

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
#include "gbees.h"

/** Register ctrl-C handler */
void registerSignalHandlers(void);

/** Ctrl+C handler */
void signalHandler(int signal);

/** Print usage and exit */
void printUsageAndExit(const char* command);

/** Main function */
int main(int argc, char **argv) {  
    // autotest executes one preconfigured computation, useful for testing purposes
    bool autotest = false;

    // parameters check
    if(argc == 3){
        if(!strcmp(argv[2], "autotest" )) autotest = true;
        else printUsageAndExit(argv[0]);
    }
    if(argc != 2 && argc != 3) printUsageAndExit(argv[0]);
    
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

    // call the kernel
    int blocks = 1;
    int threads = 1;
    kernel<<<blocks,threads>>>();

    cudaDeviceSynchronize();

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
