// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#include "../config.h"
#include "../macro.h"
#include "../models.h"
#include "pcr3bp.h"
#include <float.h>

/** --- PCR3BP --- */

/** Private declarations (model callbacks) */
static void configureGridPcr3bp(GridDefinition *grid, Measurement *firstMeasurement);
__device__ static void fPcr3bp(double* f, double* x, double* dx);
__device__ static void zPcr3bp(double* h, double* x, double* dx);
__device__ static double jPcr3bp(double* x);
__global__ static void initializeCallbacksPcr3bp(Callbacks* model);

/** Default configuration parameters for PCR3BP */
char pDirPcr3bp[] = "./results";
char mDirPcr3bp[] = "./measurements/PCR3BP";
char mFilePcr3bp[] = "measurement0.txt";

// trajectory coefficients
__device__ static const double coef[] = {1.901109735892602E-07}; // PCR3BP trajectory attributes (mu)

/** 
 * @brief Get PCR3BP default configuration
 */
void configurePcr3bp(Model* model){
    // sanity check
    if(DIM != 4){
        printf( "Error: inconsistent dimension, DIM in config.h should be defined as %d for PCR3BP model\n", 4);
        exit( DIM_ERROR );   
    }
    
    model->pDir = pDirPcr3bp;      // Saved PDFs path
    model->mDir = mDirPcr3bp;      // Measurement path
    model->mFile = mFilePcr3bp;    // Measurement file        
    model->mDim = 3;                 // Measurement dimension
    model->numDistRecorded = 8;      // Number of distributions recorded per measurement
    model->recordDivider = 1;        // Use a value greater than 1 to record only a fraction of the total distributions
    model->recordSelected = 0;       // Select which fraction of the total records are recorded
    model->numMeasurements = 4;      // Number of measurements
    model->deletePeriodSteps = 20;   // Number of steps per deletion procedure
    model->outputPeriodSteps = 20;   // Number of steps per output to terminal
    model->performOutput = true;     // Write info to terminal
    model->performRecord = false;     // Write PDFs to .txt file
    model->performMeasure = true;    // Take discrete measurement updates
    model->useBounds = true;        // Add inadmissible regions to grid
    model->configureGrid = &configureGridPcr3bp; // Grid configuration callback
    
    HANDLE_CUDA(cudaMalloc(&model->callbacks, sizeof(Callbacks)));
    initializeCallbacksPcr3bp<<<1,1>>>(model->callbacks);       
}

/** Initialize callbacks */
__global__ static void initializeCallbacksPcr3bp(Callbacks* callbacks){
    callbacks->f = fPcr3bp;
    callbacks->z = zPcr3bp;  
    callbacks->j = jPcr3bp;
}

/**
 * @brief This function defines the dynamics model
 * 
 * @param f [output] output vector (dx/dt)
 * @param x current state
 * @param dx grid with in each dimension 
 */
__device__ static void fPcr3bp(double* f, double* x, double* dx){    
    double r1 = pow(pow(x[0]+coef[0],2)+pow(x[1],2),1.5);
    double r2 = pow(pow(x[0]-1+coef[0],2)+pow(x[1],2),1.5);
    f[0] = x[2];
    f[1] = x[3];
    f[2] = 2*x[3]+x[0]-(coef[0]*(x[0]-1+coef[0])/r2)-((1-coef[0])*(x[0]+coef[0])/r1);
    f[3] = -2*x[2]+x[1]-(coef[0]*x[1]/r2)-((1-coef[0])*x[1]/r1);
}

/**
 * @brief  This function defines the measurement model(required if MEASURE == true)
 * 
 * @param h [output] output vector
 * @param x current state
 * @param dx grid with in each dimension
 */
__device__ static  void zPcr3bp(double* h, double* x, double* dx){    
    h[0] = pow(pow(x[0] - (1 - coef[0]), 2) + pow(x[1], 2),0.5); 
    h[1] = atan2(x[1], x[0] - (1 - coef[0])); 
    h[2] = ((x[0] - (1 - coef[0]))*x[2] + x[1]*x[3])/h[0];
}

/**
 * @brief Initial boundary function
 * @param x current state
 * @return J
 */
__device__ static double jPcr3bp(double* x){    
    double r1 = pow(pow(x[0]+coef[0],2)+pow(x[1],2), 0.5);
    double r2 = pow(pow(x[0]-1+coef[0],2)+pow(x[1],2), 0.5);
    double J = pow(x[0], 2.0) + pow(x[1], 2.0) + (2*(1-coef[0])/r1) + (2*coef[0]/r2) + coef[0]*(1 - coef[0]) - (pow(x[2], 2.0) + pow(x[3], 2.0));
    return J;
}

/**
 * @brief Ask to the model to define the grid configuration
 * 
 * @param grid [output] the grid definition object to configure
 * @param firstMeasurement the first measurement
 */
static void configureGridPcr3bp(GridDefinition *grid, Measurement *firstMeasurement){    
    grid->dt = DBL_MAX;
    grid->threshold = 1E-7;    
    grid->hi_bound = DBL_MAX;
    grid->lo_bound = -DBL_MAX;    
    
    // Grid width, default is half of the std. dev. from the initial measurement 
    for(int i=0; i<DIM; i++){
        grid->center[i] = firstMeasurement->mean[i];
        grid->dx[i] = pow(firstMeasurement->cov[i*DIM +i],0.5)/2.0;
    }
}