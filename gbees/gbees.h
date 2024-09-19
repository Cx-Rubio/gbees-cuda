// Copyright 2024 by Benjamin Hanson, published under BSD 3-Clause License.

#ifndef GBEES_H
#define GBEES_H

/** Grid definition */
typedef struct {
    int maxCells;
    int dim; 
    double thresh;
    double dt;
    double *center;
    double *dx;
    double hi_bound;
    double lo_bound;
} GridDefinition;

/** Trajectory information */
typedef struct Traj {
    double *coef;
} Traj;

/** Cell definition */
typedef struct Cell Cell;
struct Cell {    
    double prob;
    double *v;
    double *ctu;
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