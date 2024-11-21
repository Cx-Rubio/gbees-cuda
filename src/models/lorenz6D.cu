// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#include "../config.h"
#include "../macro.h"
#include "../models.h"
#include "lorenz6D.h"
#include <float.h>

/** --- Lorenz6D --- */

/** Private declarations (model callbacks) */
static void configureGridLorenz6D(GridDefinition *grid, Measurement *firstMeasurement);
__device__ static void fLorenz6D(double* f, double* x, double* dx);
// __device__ static void zLorenz6D(double* h, double* x, double* dx);
__global__ static void initializeCallbacksLorenz6D(Callbacks* model);

/** Default configuration parameters for Lorenz6D */
char pDirLorenz6D[] = "./results/Lorenz6D";
char mDirLorenz6D[] = "./measurements/Lorenz6D";
char mFileLorenz6D[] = "measurement0.txt";

/** 
 * @brief Get Lorenz6D default configuration
 */
void configureLorenz6D(Model* model){
    // sanity check
    if(DIM != 6){
        printf( "Error: inconsistent dimension, DIM in config.h should be defined as %d for Lorenz6D model\n", 6);
        exit( DIM_ERROR );   
    }
    
    model->pDir = pDirLorenz6D;      // Saved PDFs path
    model->mDir = mDirLorenz6D;      // Measurement path
    model->mFile = mFileLorenz6D;    // Measurement file        
    model->mDim = 1;                 // Measurement dimension
    model->numDistRecorded = 2;      // Number of distributions recorded per measurement
    model->recordDivider = 1;        // Use a value greater than 1 to record only a fraction of the total distributions
    model->recordSelected = 0;       // Select which fraction of the total records are recorded
    model->numMeasurements = 1;      // Number of measurements
    model->deletePeriodSteps = 20;   // Number of steps per deletion procedure
    model->outputPeriodSteps = 20;   // Number of steps per output to terminal
    model->performOutput = true;     // Write info to terminal
    model->performRecord = true;     // Write PDFs to .txt file
    model->performMeasure = false;   // Take discrete measurement updates
    model->useBounds = false;        // Add inadmissible regions to grid
    model->configureGrid = &configureGridLorenz6D; // Grid configuration callback
    
    HANDLE_CUDA(cudaMalloc(&model->callbacks, sizeof(Callbacks)));
    initializeCallbacksLorenz6D<<<1,1>>>(model->callbacks);       
}

/** Initialize callbacks */
__global__ static void initializeCallbacksLorenz6D(Callbacks* callbacks){
    callbacks->f = fLorenz6D;
    callbacks->z = NULL;  
    callbacks->j = NULL;
}

/**
 * @brief This function defines the dynamics model
 * 
 * @param f [output] output vector (dx/dt)
 * @param x current state
 * @param dx grid with in each dimension 
 */
__device__ static void fLorenz6D(double* f, double* x, double* dx){
    double coef[] = {4};
    f[0] = (x[1] - x[4]) * x[5] - x[0] + coef[0];
    f[1] = (x[2] - x[5]) * x[0] - x[1] + coef[0];
    f[2] = (x[3] - x[0]) * x[1] - x[2] + coef[0];
    f[3] = (x[4] - x[1]) * x[2] - x[3] + coef[0];
    f[4] = (x[5] - x[2]) * x[3] - x[4] + coef[0];
    f[5] = (x[0] - x[3]) * x[4] - x[5] + coef[0];
}

// /**
//  * @brief  This function defines the measurement model(required if MEASURE == true)
//  * 
//  * @param h [output] output vector
//  * @param x current state
//  * @param dx grid with in each dimension
//  */
// __device__ static  void zLorenz6D(double* h, double* x, double* dx){
//     h[0] = x[2];
// }

/**
 * @brief Ask to the model to define the grid configuration
 * 
 * @param grid [output] the grid definition object to configure
 * @param firstMeasurement the first measurement
 */
static void configureGridLorenz6D(GridDefinition *grid, Measurement *firstMeasurement){    
    grid->dt = DBL_MAX;
    grid->threshold = 8E-9;    
    grid->hi_bound = DBL_MAX;
    grid->lo_bound = -DBL_MAX;    
    
    // Grid width, default is half of the std. dev. from the initial measurement 
    for(int i=0; i<DIM; i++){
        grid->center[i] = firstMeasurement->mean[i];
        grid->dx[i] = pow(firstMeasurement->cov[i*DIM +i],0.5)/2.0;
    }
}