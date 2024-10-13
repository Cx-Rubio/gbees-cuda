// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

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

    // Check if the device support cooperative launch
    if(!supportsCooperativeLaunch(device)){
        handleError(GPU_ERROR, "Selected device do not support cooperative launch\n");
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
 * @brief Check if the device supports cooperative launch
 * 
 * @param device the device if
 * @return if supports cooperative launch
 */
bool supportsCooperativeLaunch(int device){
    int supportsCoopLaunch = 0;
    HANDLE_CUDA(cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, device));    
    return supportsCoopLaunch;
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
    if(supportsCooperativeLaunch(device)){
        printf("  supports cooperative launch\n");    
    }
    printf("  multiprocessors %d\n", prop.multiProcessorCount);    
    printf("  max threads per block %d\n", prop.maxThreadsPerBlock);    
    printf("  max threads per multiprocessor %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  max simultaneous theads %d\n", prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount);
    printf("  shared memory per block %lu\n", prop.sharedMemPerBlock);
    printf("  shared memory per multiprocessor %lu\n", prop.sharedMemPerMultiprocessor);    

    size_t freeMemory;
    size_t totalMemory;
    HANDLE_CUDA(cudaSetDevice(device));      
    HANDLE_CUDA(cudaMemGetInfo(&freeMemory, &totalMemory));
    printf("  total memory: %.2f GB\n", (totalMemory / GB));
}
