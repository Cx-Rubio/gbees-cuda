// Copyright 2024 by Carlos Rubio, published under BSD 3-Clause License.

#ifndef DEVICE_H
#define DEVICE_H

/**
 * Prints some info of all detected CUDA GPUs
 */
void printDevices(void);

/**
 * Prints some info of one local CUDA GPUs
 */
void printDevice(int device);

/**
 * Selects the GPU with the max number of multiprocessors
 */
int selectBestDevice();

/**
 * Gets the maximum number of threads per block of one local CUDA GPU
 */
int getMaxThreadsPerBlock(int device);

#endif