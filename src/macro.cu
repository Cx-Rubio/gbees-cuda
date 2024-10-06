// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#include "macro.h"
#include "config.h"
#include <stdio.h>
#include <stdarg.h>
#include <cuda_profiler_api.h>

/**
 * @brief Check kernel error
 */
__host__ void checkKernelError(){
    // check kernel error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
    }    
}
    
/**
 * @brief Log a message if enabled log in config.h (host)
 */
__host__ void log(const char* msg, ...){
#ifdef ENABLE_LOG     
    va_list args;
    va_start(args, msg);
    vprintf(msg, args);    
    va_end(args);    
#endif
}
