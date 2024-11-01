/**
 * This file includes the error macros as defined in the book: 
 *  CUDA by Example: An Introduction to General-Purpose GPU Programming
 *  by Jason Sanders and Edward Kandrot
 *  https://developer.nvidia.com/cuda-example
 */
 
#ifndef MACRO_H
#define MACRO_H

#include "error.h"
#include <stdio.h>

/**
 * @brief Handle app error, print the error name, the file and line where occurs, and exit
 * 
 * @param err error index in the error enumeration
 * @param file source file name
 * @param line source file line
 */
static void HandleError( int err, const char *file, int line) {
    if (err != 0) {
        printf( "Error: %s in %s at line %d\n", getErrorString(err), file, line );
        exit( EXIT_CODE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

/**
 * @brief Handle CUDA error, print the error name, the file and line where occurs, and exit
 * 
 * @param err error index in the error enumeration
 * @param file source file name
 * @param line source file line
 */
static void HandleErrorCuda( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_CODE );
    }
}
#define HANDLE_CUDA( err ) (HandleErrorCuda( err, __FILE__, __LINE__ ))

/**
 * @brief Handle no GPU detected error
 */
static void HandleNoGpu(){
    printf("Not found any CUDA device or insufficient driver.\n");
    exit( EXIT_CODE );
}
#define HANDLE_NO_GPU() (HandleNoGpu())

/**
 * @brief Check kernel error
 */
__host__ void checkKernelError();

/**
 * @brief Log a message if enabled log in config.h (host)
 */
__host__ void log(const char* msg, ...);

/** Macro for log from device */
#ifdef ENABLE_LOG  
    #define LOG(...) if(threadIdx.x == 0 && blockIdx.x == 0) printf(__VA_ARGS__)
#else
    #define LOG(...) void()
#endif

/**
 * Profiling macros
 */
#define TIME_INIT unsigned long long start, stop, cycles = 0; global.gridDefinition->cycles = 0;
#define TIME_START g.sync(); if(threadIdx.x == 0 && blockIdx.x == 0) { start = clock64(); }
#define TIME_STOP g.sync(); if(threadIdx.x == 0 && blockIdx.x == 0) { stop = clock64();  cycles += stop - start; }
#define TIME_PRINT if(threadIdx.x == 0 && blockIdx.x == 0) printf("Cycles %lld\n", cycles);

#define TIME_INIT_INNER unsigned long long start, stop, cycles = 0;
#define TIME_START_INNER g.sync(); start = clock64();
#define TIME_STOP_INNER g.sync(); stop = clock64();  cycles += stop - start; gridDefinition->cycles += cycles; 

#define TIME_START_INNER_NO_SYNC start = clock64();
#define TIME_STOP_INNER_NO_SYNC stop = clock64();  cycles += stop - start; gridDefinition->cycles += cycles; 

#define TIME_PRINT_INNER if(threadIdx.x == 0 && blockIdx.x == 0) printf("Cycles inner %lld\n", global.gridDefinition->cycles);

#endif