
#ifndef MODELS_H
#define MODELS_H

/** Trajectory information */
typedef struct  {
    double *coefficients;
} Trajectory;

/** Model configuration */
typedef struct {
  char* pDir; // Saved PDFs path
  char* mDir; // Measurement path
  char* mFile; // Measurement file
  Trajectory trajectory;        // Trajectory information
  void (*f)(double*, double*, double*, double*);   // Dynamics model function ptr
  void (*z)(double*, double*, double*, double*); // Measurement model function ptr
  int numDistRecorded;        // Number of distributions recorded per measurement
  int numMeasurements;     // Number of measurements
  int deletePeriodSteps;       // Number of steps per deletion procedure
  int outputPeriodSteps;      // Number of steps per output to terminal
  bool performOutput;         // Write info to terminal
  bool performRecord;         // Write PDFs to .txt file // REF- Convention over Configuration (CoC)
  bool performMeasure;      // Take discrete measurement updates
  bool useBounds;               // Add inadmissible regions to grid  
} Model;

/** 
 * @brief Get Lorenz3D configuration
 */
Model getLorenz3DConfig(void);

#endif