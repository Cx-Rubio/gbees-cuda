// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#include "config.h"
#include "macro.h"
#include "models.h"
#include <float.h>

/**
 * @brief Free model memory
 */
void freeModel(Model* model){
    HANDLE_CUDA( cudaFree( model->callbacks) ); 
}
