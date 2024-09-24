// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#ifndef GBEES_H
#define GBEES_H

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
    Cell **i_nodes;
    Cell **k_nodes;
    double dcu;
    double cfl_dt;
    int new_f;
    int ik_f;
    double bound_val;    
};

#endif