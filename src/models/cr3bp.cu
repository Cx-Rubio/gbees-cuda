// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#include "../config.h"
#include "../macro.h"
#include "../models.h"
#include "cr3bp.h"
#include <float.h>

/** --- CR3BP --- */

/** Private declarations (model callbacks) */
static void configureGridCr3bp(GridDefinition *grid, Measurement *firstMeasurement);
__device__ static void fCr3bp(double* f, double* x, double* dx);
__device__ static double jCr3bp(double* x);
__global__ static void initializeCallbacksCr3bp(Callbacks* model);

/** Default configuration parameters for CR3BP */
char pDirCr3bp[] = "./results";
char mDirCr3bp[] = "./measurements/CR3BP";
char mFileCr3bp[] = "measurement0.txt";

// trajectory coefficients
__device__ static const double coef[] = {2.528017528540000E-5}; // CR3BP trajectory attributes (mu)

/** 
 * @brief Get CR3BP default configuration
 */
void configureCr3bp(Model* model){
    // sanity check
    if(DIM != 6){
        printf( "Error: inconsistent dimension, DIM in config.h should be defined as %d for CR3BP model\n", 6);
        exit( DIM_ERROR );   
    }
    
    model->pDir = pDirCr3bp;      // Saved PDFs path
    model->mDir = mDirCr3bp;      // Measurement path
    model->mFile = mFileCr3bp;    // Measurement file        
    model->mDim = 6;                 // Measurement dimension
    model->numDistRecorded = 17;      // Number of distributions recorded per measurement
    model->recordDivider = 1;        // Use a value greater than 1 to record only a fraction of the total distributions
    model->recordSelected = 0;       // Select which fraction of the total records are recorded
    model->numMeasurements = 1;      // Number of measurements
    model->deletePeriodSteps = 20;   // Number of steps per deletion procedure
    model->outputPeriodSteps = 20;   // Number of steps per output to terminal
    model->performOutput = true;     // Write info to terminal
    model->performRecord = false;     // Write PDFs to .txt file
    model->performMeasure = false;    // Take discrete measurement updates
    model->useBounds = true;        // Add inadmissible regions to grid
    model->configureGrid = &configureGridCr3bp; // Grid configuration callback
    
    HANDLE_CUDA(cudaMalloc(&model->callbacks, sizeof(Callbacks)));
    initializeCallbacksCr3bp<<<1,1>>>(model->callbacks);       
}

/** Initialize callbacks */
__global__ static void initializeCallbacksCr3bp(Callbacks* callbacks){
    callbacks->f = fCr3bp;
    callbacks->z = NULL;  
    callbacks->j = jCr3bp;
}

/**
 * @brief This function defines the dynamics model
 * 
 * @param f [output] output vector (dx/dt)
 * @param x current state
 * @param dx grid with in each dimension 
 */
__device__ static void fCr3bp(double* f, double* x, double* dx){    
    double r1 = pow(pow(x[0]+coef[0],2) + pow(x[1],2) + pow(x[2],2), 1.5);
    double r2 = pow(pow(x[0]-1+coef[0],2) + pow(x[1],2) + pow(x[2],2), 1.5);
    f[0] = x[3];
    f[1] = x[4];
    f[2] = x[5];
    f[3] = 2*x[4]+x[0]-(coef[0]*(x[0]-1+coef[0])/r2)-((1-coef[0])*(x[0]+coef[0])/r1);
    f[4] = -2*x[3]+x[1]-(coef[0]*x[1]/r2)-((1-coef[0])*x[1]/r1);
    f[5] = -(coef[0]*x[2]/r2)-((1-coef[0])*x[2]/r1);
}

/**
 * @brief Initial boundary function
 * @param x current state
 * @return J
 */
__device__ static double jCr3bp(double* x){    
    double r1 = pow(pow(x[0]+coef[0],2)+pow(x[1],2)+pow(x[2],2), 0.5);
    double r2 = pow(pow(x[0]-1+coef[0],2)+pow(x[1],2)+pow(x[2],2), 0.5);
    double J = pow(x[0], 2.0) + pow(x[1], 2.0) + (2*(1-coef[0])/r1) + (2*coef[0]/r2) + coef[0]*(1 - coef[0]) - (pow(x[3], 2.0) + pow(x[4], 2.0) + pow(x[5], 2.0));
    return J;
}

/**
 * @brief Ask to the model to define the grid configuration
 * 
 * @param grid [output] the grid definition object to configure
 * @param firstMeasurement the first measurement
 */
static void configureGridCr3bp(GridDefinition *grid, Measurement *firstMeasurement){    
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