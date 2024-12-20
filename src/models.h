// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.

#ifndef MODELS_H
#define MODELS_H

#include "grid.h"
#include "measurement.h"

/** Forward declaration*/
struct Measurement;
struct GridDefinition;

/** Model device callbacks */
typedef struct {
  void (*f)(double*, double*, double*); // Dynamics model function ptr
  void (*z)(double*, double*, double*); // Measurement model function ptr
  double (*j)(double*); // Boundary function (optional)  
}  Callbacks;

/** Model configuration */
typedef struct Model Model;
struct Model {
  char* pDir; // Saved PDFs path
  char* mDir; // Measurement path
  char* mFile; // Measurement file    
  void (*configureGrid)(GridDefinition*, Measurement*); // Ask to the model to define the grid configuration
  int mDim;       // Measurement model dimension (DIM_h)
  int numDistRecorded;        // Number of distributions recorded per measurement
  int recordDivider;    // Use a value greater than 1 to record only a fraction of the total distributions
  int recordSelected;   // Select which fraction of the total records are recorded
  int numMeasurements;     // Number of measurements
  int deletePeriodSteps;       // Number of steps per deletion procedure
  int outputPeriodSteps;      // Number of steps per output to terminal
  bool performOutput;         // Write info to terminal
  bool performRecord;         // Write PDFs to .txt file // REF- Convention over Configuration (CoC)
  bool performMeasure;      // Take discrete measurement updates
  bool useBounds;               // Add inadmissible regions to grid  
  Callbacks* callbacks;
};

/**
 * @brief Free model memory
 */
void freeModel(Model* model);

#endif