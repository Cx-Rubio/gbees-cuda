// Copyright 2024 by Carlos Rubio, published under BSD 3-Clause License.

#include "macro.h"
#include "device.h"
#include <math.h>

// Not declared as unsigned integer because it is
// used in divisions where we want decimal results
static const double GB = 1024*1024*1024;

/**
 * @brief Selects the GPU with the max number of multiprocessors
 */
int selectBestDevice(){    
    int maxMultiprocessors = 0;
    int device = -1;
    cudaDeviceProp prop;
    int count;

    if(cudaGetDeviceCount(&count) != cudaSuccess)
        HANDLE_NO_GPU();

    for(int i=0;i<count;i++){
        HANDLE_CUDA(cudaGetDeviceProperties(&prop, i));
        if (maxMultiprocessors < prop.multiProcessorCount) {
            maxMultiprocessors = prop.multiProcessorCount;
            device = i;
          }        
    }    
    return device;
}

/**
 * @brief Gets the maximum number of threads per block of one local CUDA GPU
 * 
 * @param device the device id
 * @return the maximun number of threads per block
 */
int getMaxThreadsPerBlock(int device){
    cudaDeviceProp prop;
    HANDLE_CUDA(cudaGetDeviceProperties(&prop, device));
    return prop.maxThreadsPerBlock;
}

/**
 * @brief Prints some info of all detected CUDA GPUs
 */
void printDevices(void){    
    int count;

    if(cudaGetDeviceCount(&count) != cudaSuccess)
        HANDLE_NO_GPU();

    for(int i=0;i<count;i++){
       printDevice(i);   
    }
}

/**
 * @brief Prints some info of one local CUDA GPUs
 * 
 * @param device the device id
 */
void printDevice(int device){
    cudaDeviceProp prop;
    HANDLE_CUDA(cudaGetDeviceProperties(&prop, device));    
    printf("\nDevice %d, %s, rev: %d.%d\n",device, prop.name, prop.major, prop.minor);    
    printf("  max threads per block %d\n", prop.maxThreadsPerBlock);    
    printf("  max threads %d %d %d\n", prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);            
    printf("  max blocks %d %d %d\n", prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
    printf("  multiprocessors %d\n", prop.multiProcessorCount);    
    printf("  shared memory per block %lu\n", prop.sharedMemPerBlock);
    printf("  shared memory per multiprocessor %lu\n", prop.sharedMemPerMultiprocessor);    

    size_t freeMemory;
    size_t totalMemory;
    HANDLE_CUDA(cudaSetDevice(device));      
    HANDLE_CUDA(cudaMemGetInfo(&freeMemory, &totalMemory));
    printf("  total memory: %.2f GB\n", (totalMemory / GB));
}
