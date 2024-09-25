// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#ifndef GBEES_H
#define GBEES_H

#include <stdint.h>

/** Grid definition */
typedef struct {
    int maxCells;        
    double center[DIM];
    double dx[DIM];
    double dt;
    double threshold;    
    double hi_bound;
    double lo_bound;
} GridDefinition;

/** Cell definition */
typedef struct Cell Cell;
struct Cell {    
    double prob;
    double v[DIM];
    double ctu[DIM];
    int32_t state[DIM];
    double x[DIM];
    uint32_t iNodes[DIM]; // used list indexes
    uint32_t kNodes[DIM]; // used list indexes
    double dcu;
    double cfl_dt;
    int new_f;
    int ik_f;
    double bound_val;    
};

#endif