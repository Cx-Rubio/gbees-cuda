#include "measurement.h"
#include <stdio.h>
#include "../cuda/error.h"
#include "models.h"

/**
 * @brief Allocate measurements memory in host
 * 
 * @param size number of measurement structs
 * @return pointer to allocated memory
 */
Measurement* allocMeasurementsHost(int size){
    Measurement* ptr = (Measurement*)malloc(size * sizeof(Measurement));
    assertNotNull(ptr, MALLOC_ERROR, "Error allocating host memory for measurements");
    return ptr;
}

/**
 * @brief Allocate measurements memory in device
 * 
 * @param size number of measurement structs
 * @return pointer to allocated memory
 */
Measurement* allocMeasurementsDevice(int size){
    Measurement* ptr;
    HANDLE_CUDA( cudaMalloc( &ptr, size * sizeof(Measurement)) );
    return ptr;
}

/**
 * @brief Copy measurements from host to device
 * @param measurements pointer to origin
 * @param measurementsDevice pointer to destination
 * @param size number of measurement elements
 */
void copyHostToDeviceMeasurements(Measurement *measurements, Measurement *measurementsDevice, int size){
    HANDLE_CUDA( cudaMemcpy( measurementsDevice , measurements, size * sizeof(Measurement), cudaMemcpyHostToDevice) ); 
}

/**
 * @brief Free measurements memory at host
 * 
 * @param ptr measurements array pointer
 */
void freeMeasurementsHost(Measurement* ptr){
    free(ptr);
}

/**
 * @brief Free measurements memory at device
 * 
 * @param ptr measurements array pointer
 */
void freeMeasurementsDevice(Measurement* ptr){
    HANDLE_CUDA( cudaFree(ptr) ); 
}

/** 
 * @brief Read measurements from files
 * 
 * @param measurement [output] measurement array
 * @param model model pointer
 * @param count number of measurement files to read 
 */
void readMeasurements(Measurement* measurements, Model* model, int count){
    for(int i=0;i<count;i++){
        int dimension = (i == 0)? DIM : model->mDim;
        readMeasurement(&measurements[i], dimension, model->mDir, i);
        computeCovarianceInverse(&measurements[i], dimension);
    }
}

/** 
 * @brief Read one measurement from file
 * By convention the file names should be measurement[index].txt, for example for 
 * index = 0, the file name should be measurement0.txt
 * 
 * @param measurement [output] measurement structure to fill in
 * @param dim dimensionality of measurement structure
 * @param dir measurements folder
 * @param index measurement index
 */
void readMeasurement(Measurement *measurement, int dim, const char* dir, int index) {
    char path[256];
    snprintf(path, sizeof(path), "%s/measurement%d.txt", dir, index);

    FILE *fd = fopen(path, "r");
    assertNotNull(fd, IO_ERROR, "Error: could not open file %s", path);
       
    measurement->dim = dim; 

    char line[256];
    char *token;
    int count = 0;

    assertNotNull(fgets(line, sizeof(line), fd), FORMAT_ERROR, "Error reading measurement file %s", path); // skip label line
    assertNotNull(fgets(line, sizeof(line), fd), FORMAT_ERROR, "Error reading measurement file %s", path); // mean vector
    token = strtok(line, " ");
    while (token != NULL && count < dim) { // read mean vector
        measurement->mean[count++] = strtod(token, NULL);
        token = strtok(NULL, " ");
    }
    count = 0;

    // Read covariance matrix
    assertNotNull(fgets(line, sizeof(line), fd), FORMAT_ERROR, "Error reading measurement file %s", path); // skip blank line
    assertNotNull(fgets(line, sizeof(line), fd), FORMAT_ERROR, "Error reading measurement file %s", path); // skip label line
    for (int i = 0; i < dim; i++) { // read covariance matrix
        assertNotNull(fgets(line, sizeof(line), fd), FORMAT_ERROR, "Error reading measurement file %s", path);
        token = strtok(line, " ");
        while (token != NULL && count < dim) {
            measurement->cov[i][count++] = strtod(token, NULL);
            token = strtok(NULL, " ");
        }
        count = 0;
    }

    assertNotNull(fgets(line, sizeof(line), fd), FORMAT_ERROR, "Error reading measurement file %s", path); // skip blank line
    assertNotNull(fgets(line, sizeof(line), fd), FORMAT_ERROR, "Error reading measurement file %s", path); // skip label line
    assertNotNull(fgets(line, sizeof(line), fd), FORMAT_ERROR, "Error reading measurement file %s", path); // read T value
    measurement->T = strtod(line, NULL);

    fclose(fd);    
}

/** 
 * @brief Compute the inverse of the covariance matrix
 * 
 * @param measurement measurement structure pointer to update the covariance inverse matrix
 * @param dim dimensionality of measurement structure
 */
void computeCovarianceInverse(Measurement *measurement, int dim){
    
}

/**
 * @brief Print measurements info
 * 
 * @param measurement measurement array pointer
 * @param size number of elements
 */
void printMeasurements(Measurement *measurements, int count){
    printf("\n -- Measurements -- \n");
    for(int i=0;i<count;i++){
        printMeasurement(&measurements[i]);
    }    
}

/**
 * @brief Print measurement info
 * 
 * @param measurement measurement pointer
 */
void printMeasurement(Measurement *measurement){
    printf("Measurement: dim %d\n", measurement->dim);
    printf("  Mean: {");
    for(int i=0;i<measurement->dim;i++){
        printf("%f", measurement->mean[i]);
        if(i < measurement->dim-1) printf(", ");
    }  
    printf("}\n");
    printf("  Covariance: {\n");
    for(int i=0;i<measurement->dim;i++){
        for(int j=0;j<measurement->dim;j++){
            if(j == 0) printf("    ");
            printf("%f", measurement->cov[i][j]);
            if(j < measurement->dim-1) printf(", ");
            else printf("\n");        
        }
    }  
    printf("  }\n");
    printf("  T: %f\n", measurement->T);
}