// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#include "error.h"
#include <stdio.h>
#include <stdarg.h>

/** 
 *@brief Gets error description from error code
 * 
 *@param err error index in the error enumeration
 */
const char* getErrorString(int err){
    switch(err){
        case MALLOC_ERROR: return "malloc error";
        case KERNEL_ERROR: return "kernel error";
        case GPU_ERROR: return "gpu error";
        case IO_ERROR: return "IO error";
        case FORMAT_ERROR: return "format error";        
        case DIM_ERROR: return "dimension error";
        default: return "";
    }
}

/**
 * @brief Assert that ptr is not null, print error and exit otherwise
 *
 * @param ptr the pointer to check
 * @param errorCode the associated error code to the assertion fail
 * @param msg message to print in case of assertion fail
 */
void assertNotNull(void *ptr, enum Error errorCode, const char* msg, ...){
    if(ptr == NULL) {
        va_list args;
        va_start(args, msg);
        vprintf(msg, args);
        printf("\n");
        va_end(args);
        exit(errorCode);
    }
}

/**
 * @brief Assert that the value is positive, print error and exit otherwise
 *
 * @param value the value to check
 * @param errorCode the associated error code to the assertion fail
 * @param msg message to print in case of assertion fail
 */
void assertPositiveOrZero(int value, enum Error errorCode, const char* msg, ...){
    if(value < 0) {
        va_list args;
        va_start(args, msg);
        vprintf(msg, args);
        printf("\n");
        va_end(args);
        exit(errorCode);
    }
}

/** 
 * @brief Launch error and exit 
 * 
 * @param errorCode the error code
 * @param msg message to print
 */
void handleError(enum Error errorCode, const char* msg, ...){
    va_list args;
    va_start(args, msg);    
    printf(msg, args);
    printf("\n");
    exit(errorCode);
}