// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#ifndef MEASUREMENT_H
#define MEASUREMENT_H

#include "config.h"
#include "macro.h"

/** Forward declaration */
struct Model;

/** Measurement structure */
typedef struct Measurement Measurement;
struct Measurement{
    int dim;                 // dimensionality of measurement mean and covariance
    double mean[DIM];        // measurement mean
    double cov[DIM][DIM];    // covariance matrix
    double covInv[DIM][DIM]; // covariance inverse matrix
    double T;                // period of continuous-time propagation before next measurement update
};

/**
 * @brief Allocate measurements memory in host
 * 
 * @param size number of measurement structs
 * @return pointer to allocated memory
 */
Measurement* allocMeasurementsHost(int size);

/**
 * @brief Allocate measurements memory in device
 * 
 * @param size number of measurement structs
 * @return pointer to allocated memory
 */
Measurement* allocMeasurementsDevice(int size);

/**
 * @brief Copy measurements from host to device
 * @param measurements pointer to origin
 * @param measurementsDevice pointer to destination
 * @param size number of measurement elements
 */
void copyHostToDeviceMeasurements(Measurement *measurements, Measurement *measurementsDevice, int size);

/**
 * @brief Free measurements memory at host
 * 
 * @param ptr measurements array pointer
 */
void freeMeasurementsHost(Measurement* ptr);

/**
 * @brief Free measurements memory at device
 * 
 * @param ptr measurements array pointer
 */
void freeMeasurementsDevice(Measurement* ptr);

/** 
 * @brief Read measurements from files
 * 
 * @param measurement [output] measurement array
 * @param mDimension dimension of measurements (except the first one with dimension DIM)
 * @param mDir measurements folder
 * @param count number of measurement files to read 
 */
void readMeasurements(Measurement* measurements, int mDimension, char* mDir, int count);

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
void readMeasurement(Measurement *measurement, int dim, const char* dir, int index);

/** 
 * @brief Compute the inverse of the covariance matrix
 * 
 * @param measurement measurement structure pointer to update the covariance inverse matrix
 * @param dim dimensionality of measurement structure
 */
void computeCovarianceInverse(Measurement *measurement, int dim);

/**
 * @brief Print measurements info
 * 
 * @param measurement measurement array pointer
 * @param count number of elements
 */
void printMeasurements(Measurement *measurements, int count);

/**
 * @brief Print measurement info
 * 
 * @param measurement measurement pointer
 */
void printMeasurement(Measurement *measurement);

#endif