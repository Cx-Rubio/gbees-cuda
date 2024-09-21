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

#endif