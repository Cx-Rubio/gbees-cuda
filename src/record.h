// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#ifndef RECORD_H
#define RECORD_H

#include "kernel.h"
#include "grid.h"
#include "models.h"

/**
 * @brief Record distributions
 * 
 * @param snapshotsHost snapshots host pointer
 * @param snapshotsDevice snapshots device pointer
 * @param model the model
 * @param grid the grid
 * @param gridDefinition grid definition
 */
void recordDistributions(Snapshot* snapshotsHost, Snapshot* snapshotsDevice, Model* model, Grid* grid, GridDefinition* gridDefinition);

#endif