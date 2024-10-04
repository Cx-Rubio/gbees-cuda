// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#ifndef UTIL_H
#define UTIL_H

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

#endif