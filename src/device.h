// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#ifndef DEVICE_H
#define DEVICE_H

/**
 * @brief Prints some info of all detected CUDA GPUs
 */
void printDevices(void);

/**
 * @brief Prints some info of one local CUDA GPUs
 * 
 * @param device the device id
 */
void printDevice(int device);

/**
 * @brief Selects the GPU with the max number of multiprocessors
 */
int selectBestDevice();

/**
 * @brief Gets the maximum number of threads per block of one local CUDA GPU
 * 
 * @param device the device id
 * @return the maximun number of threads per block
 */
int getMaxThreadsPerBlock(int device);

#endif