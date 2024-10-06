// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#include "macro.h"
#include "config.h"
#include <stdio.h>
#include <stdarg.h>

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
