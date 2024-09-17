/**
 * This file includes the error macros as defined in the book: 
 *  CUDA by Example: An Introduction to General-Purpose GPU Programming
 *  by Jason Sanders and Edward Kandrot
 *  https://developer.nvidia.com/cuda-example
 */
 
#ifndef MACRO_H
#define MACRO_H

#include <stdio.h>

static const int EXIT_CODE = -1;
enum error {MALLOC_ERROR=1, KERNEL_ERROR, GPU_ERROR, IO_ERROR, FORMAT_ERROR};

/** Error description */
static const char* getErrorString(int err){
    switch(err){
        case MALLOC_ERROR: return "malloc error";
        case KERNEL_ERROR: return "kernel error";
        case GPU_ERROR: return "gpu error";
        case IO_ERROR: return "IO error";
        case FORMAT_ERROR: return "format error";
        default: return "";
    }
}

/**
 * Handle app error, print the error name, the file and line where occurs, and exit
 */
static void HandleError( int err, const char *file, int line) {
    if (err != 0) {
        printf( "Error: %s in %s at line %d\n", getErrorString(err), file, line );
        exit( EXIT_CODE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

/**
 * Handle CUDA error, print the error name, the file and line where occurs, and exit
 */
static void HandleErrorCuda( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_CODE );
    }
}
#define HANDLE_CUDA( err ) (HandleErrorCuda( err, __FILE__, __LINE__ ))

/**
 * Handle no GPU detected error
 */
static void HandleNoGpu(){
    printf("Not found any CUDA device or insufficient driver.\n");
    exit( EXIT_CODE );
}
#define HANDLE_NO_GPU() (HandleNoGpu())

#endif