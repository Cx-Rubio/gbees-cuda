// Copyright 2024 by Carlos Rubio, published under BSD 3-Clause License.

#ifndef ERROR_H
#define ERROR_H

/** Error codes */
static const int EXIT_CODE = -1;
enum Error {MALLOC_ERROR=1, KERNEL_ERROR, GPU_ERROR, IO_ERROR, FORMAT_ERROR, DIM_ERROR};

/** 
 *@brief Gets error description from error code
 * 
 *@param err error index in the error enumeration
 */
const char* getErrorString(int err);

/**
 * @brief Assert that ptr is not null, print error and exit otherwise
 *
 * @param ptr the pointer to check
 * @param errorCode the associated error code to the assertion fail
 * @param msg message to print in case of assertion fail
 */
void assertNotNull(void *ptr, enum Error errorCode, const char* msg, ...);

/** 
 * @brief Launch error and exit 
 * 
 * @param errorCode the error code
 * @param msg message to print
 */
void handleError(enum Error errorCode, const char* msg, ...);

#endif