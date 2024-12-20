// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#ifndef CONFIG_H
#define CONFIG_H

/** Grid dimension */
#define DIM 3

/** Number of blocks */
#define BLOCKS 24

/** Number of threads per block */
#define THREADS_PER_BLOCK 512

/** Number of cells that process one thread */
#define CELLS_PER_THREAD 2

/** Enable logs (comment out to disable logs) */
#define ENABLE_LOG

/** Size of the hashtable with respect the maximum number of cells*/
#define HASH_TABLE_RATIO 2

/**  Left uncommented for single precision in the snapshots, comment it for double precision */
#define SINGLE_PRECISION_SNAPSHOTS

#endif