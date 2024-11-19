// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#include "../config.h"
#include "../macro.h"
#include "../models.h"
#include "lorenz3D.h"
#include <float.h>

/** --- Lorenz3D --- */

/** Private declarations (model callbacks) */
static void configureGridLorenz3D(GridDefinition *grid, Measurement *firstMeasurement);
__device__ static void fLorenz3D(double* f, double* x, double* dx);
__device__ static void zLorenz3D(double* h, double* x, double* dx);
__global__ static void initializeCallbacksLorenz3D(Callbacks* model);

/** Default configuration parameters for Lorenz3D */
char pDirLorenz3D[] = "./results";
char mDirLorenz3D[] = "./measurements/Lorenz3D";
char mFileLorenz3D[] = "measurement0.txt";

/** 
 * @brief Get Lorenz3D default configuration
 */
void configureLorenz3D(Model* model){
    // sanity check
    if(DIM != 3){
        printf( "Error: inconsistent dimension, DIM in config.h should be defined as %d for Lorenz3D model\n", 3);
        exit( DIM_ERROR );   
    }
    
    model->pDir = pDirLorenz3D;      // Saved PDFs path
    model->mDir = mDirLorenz3D;      // Measurement path
    model->mFile = mFileLorenz3D;    // Measurement file        
    model->mDim = 1;                 // Measurement dimension
    model->numDistRecorded = 5;      // Number of distributions recorded per measurement
    model->recordDivider = 1;        // Use a value greater than 1 to record only a fraction of the total distributions
    model->recordSelected = 0;       // Select which fraction of the total records are recorded
    model->numMeasurements = 2;      // Number of measurements
    model->deletePeriodSteps = 20;   // Number of steps per deletion procedure
    model->outputPeriodSteps = 20;   // Number of steps per output to terminal
    model->performOutput = true;     // Write info to terminal
    model->performRecord = false;    // Write PDFs to .txt file
    model->performMeasure = true;    // Take discrete measurement updates
    model->useBounds = false;        // Add inadmissible regions to grid
    model->configureGrid = &configureGridLorenz3D; // Grid configuration callback
    
    HANDLE_CUDA(cudaMalloc(&model->callbacks, sizeof(Callbacks)));
    initializeCallbacksLorenz3D<<<1,1>>>(model->callbacks);       
}

/** Initialize callbacks */
__global__ static void initializeCallbacksLorenz3D(Callbacks* callbacks){
    callbacks->f = fLorenz3D;
    callbacks->z = zLorenz3D;  
    callbacks->j = NULL;
}

/**
 * @brief This function defines the dynamics model
 * 
 * @param f [output] output vector (dx/dt)
 * @param x current state
 * @param dx grid with in each dimension 
 */
__device__ static void fLorenz3D(double* f, double* x, double* dx){
    double coef[] = {4.0, 1.0, 48.0};    
    f[0] = coef[0]*(x[1]-(x[0]+(dx[0]/2.0)));
    f[1] = -(x[1]+(dx[1]/2.0))-x[0]*x[2];
    f[2] = -coef[1]*(x[2]+(dx[2]/2.0))+x[0]*x[1]-coef[1]*coef[2];
}

/**
 * @brief  This function defines the measurement model(required if MEASURE == true)
 * 
 * @param h [output] output vector
 * @param x current state
 * @param dx grid with in each dimension
 */
__device__ static  void zLorenz3D(double* h, double* x, double* dx){
    h[0] = x[2];
}

/**
 * @brief Ask to the model to define the grid configuration
 * 
 * @param grid [output] the grid definition object to configure
 * @param firstMeasurement the first measurement
 */
static void configureGridLorenz3D(GridDefinition *grid, Measurement *firstMeasurement){    
    grid->dt = DBL_MAX;
    grid->threshold = 5E-6;    
    grid->hi_bound = DBL_MAX;
    grid->lo_bound = -DBL_MAX;    
    
    // Grid width, default is half of the std. dev. from the initial measurement 
    for(int i=0; i<DIM; i++){
        grid->center[i] = firstMeasurement->mean[i];
        grid->dx[i] = pow(firstMeasurement->cov[i*DIM +i],0.5)/2.0;
    }
}