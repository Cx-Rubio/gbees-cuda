// Copyright 2024 by Benjamin Hanson, published under BSD 3-Clause License.


#ifndef GBEES_H
#define GBEES_H

typedef struct Grid {
    int dim; 
    double thresh;
    double dt;
    double *center;
    double *dx;
    double hi_bound;
    double lo_bound;
} Grid;


typedef struct Traj {
    double *coef;
} Traj;

typedef struct Cell Cell;

struct Cell {
    uint64_t key;
    double prob;
    double *v;
    double *ctu;
    int *state;
    Cell **i_nodes;
    Cell **k_nodes;
    double dcu;
    double cfl_dt;
    int new_f;
    int ik_f;
    double bound_val;    
};

#endif