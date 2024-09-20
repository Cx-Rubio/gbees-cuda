
#include "models.h"

/** --- Lorenz3D --- */

/** Private declarations */
static void fLorenz3D(double* f, double* x, double* dx, double* coef);
static void zLorenz3D(double* h, double* x, double* dx, double* coef);

/** Default configuration parameters for Lorenz3D */
char pDirLorenz3D[] = "./results";
char mDirLorenz3D[] = "./measurements";
char mFileLorenz3D[] = "measurement0.txt";
double trajectoryCoefficients[] = {4.0, 1.0, 48.0};

/** 
 * @brief Get Lorenz3D default configuration
 */
Model getLorenz3DConfig(){
    Model model;
    model.pDir = pDirLorenz3D;      // Saved PDFs path
    model.mDir = mDirLorenz3D;    // Measurement path
    model.mFile = mFileLorenz3D;  // Measurement file
    model.trajectory.coefficients = trajectoryCoefficients; // Trajectory coefficients
    model.f = &fLorenz3D;                       //  Dynamics model
    model.z = &zLorenz3D;                   // Measurement model
    model.numDistRecorded = 5;          // Number of distributions recorded per measurement
    model.numMeasurements = 2;       // Number of measurements
    model.deletePeriodSteps = 20;       // Number of steps per deletion procedure
    model.outputPeriodSteps = 20;      // Number of steps per output to terminal
    model.performOutput = true;         // Write info to terminal
    model.performRecord = true;         // Write PDFs to .txt file // REF- Convention over Configuration (CoC)
    model.performMeasure = true;      // Take discrete measurement updates
    model.useBounds = false;              // Add inadmissible regions to grid  
    return model;
}

/**
 * @brief This function defines the dynamics model
 * 
 * @param h [output] output vector (dx/dt)
 * @param x current state
 * @param dx grid with in each dimension
 * @param coef other constants defined for the trajectory  
 */
static void fLorenz3D(double* f, double* x, double* dx, double* coef){
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
 * @param coef other constants defined for the trajectory
 */
static void zLorenz3D(double* h, double* x, double* dx, double* coef){
    h[0] = x[2];
}

