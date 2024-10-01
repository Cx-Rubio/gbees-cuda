// Copyright 2024 by Carlos Rubio (ULE) and Benjamin Hanson (UCSD), published under BSD 3-Clause License.
#include "config.h"
#include "kernel.h"
#include <stdio.h>
#include <cooperative_groups.h>
#include "maths.h"
#include <float.h>

namespace cg = cooperative_groups;

/** Initialize cells */
static __device__ void initializeCell(uint32_t usedIndex, GridDefinition* gridDefinition, Grid* grid, Model* model, Global* global);

/** Calculate gaussian probability at state x given mean and covariance */
static __device__ double gaussProbability(int32_t* key, GridDefinition* gridDefinition, Measurement* measurements);

/** Initialize advection values */
static __device__ void initializeAdv(GridDefinition* gridDefinition, Model* model, Cell* cell);

/** Initialize ik nodes */
static __device__ void initializeIkNodes(Grid* grid, Cell* cell, uint32_t usedIndex);

/** Initialize boundary value */
static __device__ void initializeBoundary(Cell* cell, Model* model);

/** Initialize Grid boundary */
static __device__ void initializeGridBoundary(int offsetIndex, int iterations, double* localArray, GridDefinition* gridDefinition, Grid* grid, Global* global);

/** Normalize probability distribution */
static __device__ void normalizeDistribution(int offsetIndex, int iterations, double* localArray, double* globalArray, Grid* grid);

/** Compute grid bounds */
static __device__ void gridBounds(double* output, double* localArray, double* globalArray, double boundaryValue, double(*fn)(double, double) );

/** Grow grid */
static __device__ void growGrid(int offsetIndex, int iterations, GridDefinition* gridDefinition, Grid* grid, Model* model);

/** Grow grid from one cell */
static __device__ void growGridFromCell(Cell* cell, GridDefinition* gridDefinition, Grid* grid, Model* model);

/** Grow grid from one cell in one dimension and direction */
static __device__ void growGridDireccional(Cell* cell, int dimension, enum Direction direction, GridDefinition* gridDefinition, Grid* grid, Model* model);

/** Create new cell in the grid */
static __device__ void createCell(int32_t* state, GridDefinition* gridDefinition, Grid* grid, Model* model);


/** Perform the Godunov scheme on the discretized PDF*/
static __global__ void godunov_method(int iterations, Grid* grid, GridDefinition* gridDef);

/** Compute the donor cell upwind value for each grid cell */
static __device__ void update_dcu(Cell* cell, Grid* grid, GridDefinition* gridDef);

/** Compute the corner transport upwind values in each direction */
static __device__ void update_ctu(Cell* cell, Grid* grid, GridDefinition* gridDef);

/** Compute flux from the left */
static __device__ double uPlus(double v);

/** Compute flux from the right */
static __device__ double uMinus(double v);

/** MC flux limiter */
static __device__ double fluxLimiter(double th);


 /** Check CFL condition (computes minimum) */
static __device__ void check_cfl_condition(int offsetIndex, int iterations, double* localArray, double* globalArray, Grid* grid, GridDefinition* gridDef)       

	
/** 
 * @brief This function performs the Godunov scheme on the discretized PDF, which is 2nd-order accurate
          and total variation diminishing
 * 
 * @param cell the cell object
 * @param grid the grid object
 * @param gridDefinition the grid definition
 */
static __global__ void godunov_method(int iterations, Grid* grid, GridDefinition* gridDef)
{
    int offsetIndex = threadIdx.x + blockIdx.x * blockDim.x * iterations;     
    int usedIndex;
    Cell* cell;
    // initialize cells
    for(int iter=0;iter<iterations;iter++){        
      usedIndex = (uint32_t)(offsetIndex + iter * blockDim.x); // index in the used list
      cell = getCell(usedIndex, grid); 
      update_dcu(cell, grid, gridDef);
      update_ctu(cell, grid, gridDef);

    }
}


static __device__ double uPlus(double v){
  return fmax(v, 0.0);
}

static __device__ double uMinus(double v){
  return fmin(v, 0.0);
}

static __device__ double computeFlux(Cell* cell, Cell* iCell, GridDefinition* G, int i)
{
  double F = G->dt*(cell->prob-iCell->prob)/(2*G->dx[i]);
  return F;
}

static __device__ double fluxLimiter(double th)
{
  double min1 = (1.0 + th)/2.0;
  min1 = fmin(min1, 2.0);
  min1 = fmax(min1, 2.0*th);
  return min1;
}

static __device__ void update_dcu(Cell* cell, Grid* grid, GridDefinition* gridDef)
{

  if (cell==NULL){
    return;
  }
  double vUpstream;
  double vDownstream;
  
  cell->dcu = 0;
  Cell* iCell; Cell* kCell;
    for(int i = 0; i < DIM; i++){
        cell->ctu[i] = 0.0;
        iCell = getCell(cell->iNodes[i], grid);
        kCell = getCell(cell->kNodes[i], grid);

        double dcu_p = 0;
        double dcu_m = 0;

	/* Original implementation */
	//vUpstream = iCell->v[i];
	//vDownstream = cell->v[i];

	/*Corrected implementation */
	// Valid only for equispaced meshes
	vUpstream = 0.5*(iCell->v[i] + cell->v[i]);
	vDownstream = 0.5*(cell->v[i] + kCell->v[i]);
	
	
        if(kCell != NULL){
          dcu_p = uPlus(vDownstream) * cell->prob + uMinus(vDownstream) * kCell->prob;
        }else{
          dcu_p = uPlus(vDownstream) * cell->prob;
        }
        if(iCell != NULL){
            dcu_m = uPlus(vUpstream) * iCell->prob + uMinus(vUpstream) * cell->prob;
        }
        cell->dcu -= (gridDef->dt/gridDef->dx[i])*(dcu_p-dcu_m);
    }

}

static __device__ void update_ctu(Cell* cell, Grid* grid, GridDefinition* gridDef)
{
  if (cell == NULL)
    {
      return;
    }
    Cell* iCell;
    Cell* jCell;
    Cell* pCell;
    Cell* iiCell;
    Cell* kCell;
    double th;
    double F;
    
    for(int i = 0; i < DIM; i++){
        iCell = getCell(cell->iNodes[i], grid);
        //TreeNode* i_node = r->i_nodes[i];
        //TreeNode* j_node; TreeNode* p_node;
        if(iCell!=NULL){
          F = computeFlux(cell, iCell, gridDef, i);
	/* Original implementation */
	//vUpstream = iCell->v[i];
	//vDownstream = cell->v[i];

	/*Corrected implementation */
	// Valid only for equispaced meshes
	vUpstream = 0.5*(iCell->v[i] + cell->v[i]);
	vDownstream = 0.5*(cell->v[i] + kCell->v[i]);

	  
            for(int j = 0; j < DIM; j++){

	      vUpstream_j = 0.5*(iCell->v[j] + cell->v[j]);
	      vDownstream = 0.5*(cell->v[i] + kCell->v[i]);

	      
                if (j!=i){
                  jCell = getCell(cell->iNodes[j], grid);
                  pCell = getCell(iCell->iNodes[j], grid);

                  cell->ctu[j]      -= uPlus(vUpstream) * uPlus(cell->v[j]) * F;
                  iCell->ctu[j] -= uMinus(vUpstream) * uMinus(iCell->v[j]) * F;

                  if(jCell!=NULL){
                        jCell->ctu[j] -= uPlus(vUpstream) * uMinus(jCell->v[j]) * F;
                    }
                    if(pCell!=NULL){
                        pCell->ctu[j] -= uMinus(vUpstream) * uMinus(pCell->v[j]) * F;
                    }
                }
            }


	    //High-Resolution Correction Terms

            if (vUpstream>0){
                iiCell = getCell(iCell->iNodes[i], grid);
		if(iiCell != NULL){
                    th = (iCell->prob-iiCell->prob)/(cell->prob-iCell->prob);
                }
		else
		{
                    th = (iCell->prob)/(cell->prob-iCell->prob);
                }
            }else{

	      kCell = getCell(cell->kNodes[i], grid);
                if(kCell != NULL){
                    th = (kCell->prob - cell->prob)/(cell->prob - iCell->prob);
                }else{
                    th = (-cell->prob)/(cell->prob - iCell->prob);
                }
            }

            iCell->ctu[i] += fabs(vUpstream)*(gridDef->dx[i]/gridDef->dt - fabs(vUpstream))*F*fluxLimiter(th);
    
            }
    }


}
	
/** 
 * @brief Initialization kernel function 
 * 
 * @param iterations number of cells that should process the same thread
 * @param gridDefinition the grid definition
 * @param grid the grid object
 * @param model the model
 * @param measurements the list of measurements
 */
__global__ void gbeesKernel(int iterations, GridDefinition gridDefinition, Grid grid, Model model, Global global){
    
    // shared memory for reduction processes
    __shared__ double localArray[THREADS_PER_BLOCK];   
    
    // get used list offset index
    int offsetIndex = threadIdx.x + blockIdx.x * blockDim.x * iterations;     
    
    // initialize cells
    for(int iter=0;iter<iterations;iter++){        
        int usedIndex = (uint32_t)(offsetIndex + iter * blockDim.x); // index in the used list                
        initializeCell(usedIndex, &gridDefinition, &grid, &model, &global); // initialize cell
    }    
    
    // set grid maximum and minimum bounds
    if(model.useBounds){ // TODO test use bounds
        initializeGridBoundary(offsetIndex, iterations, localArray, &gridDefinition, &grid, &global);   
        
        /* if(offsetIndex == 0){
            printf("Bounds min %e\n", gridDefinition.lo_bound);
            printf("Bounds max %e\n", gridDefinition.hi_bound);
        } */
    }
    
    // normalize distribution
    normalizeDistribution(offsetIndex, iterations, localArray, global.reductionArray, &grid);

    //if(key[0] == -3 && key[1] == -2 && key[2] == 5) printf("Probability %e\n", prob);
    //if(usedIndex == 100) printf("Probability of %d,%d,%d : %f\n", key[0], key[1], key[2], prob);
    
    /*if(key[0] == 6 && key[1] == 6 && key[2] == 0){
    if(usedIndex == 0){    
        printf("key %d, %d, %d\n",cell->state[0],cell->state[1],cell->state[2]);
        int dim = 0;
        uint32_t iNode = cell->iNodes[dim];
        uint32_t kNode = cell->kNodes[dim];
        
        if(iNode){
            int heapIndexI = (grid.usedList + (iNode-1))->heapIndex;
            Cell* cellI = &grid.heap[heapIndexI];
            printf("I node %d, %d, %d\n", cellI->state[0], cellI->state[1], cellI->state[2]);
        } else {
            printf("Mo iNode\n");
        }
        
        if(kNode){            
            int heapIndexK = (grid.usedList + (kNode-1))->heapIndex;
            Cell* cellK = &grid.heap[heapIndexK];
            printf("K node %d, %d, %d\n", cellK->state[0], cellK->state[1], cellK->state[2]);   
        } else {
            printf("Mo kNode\n");
        }            
    }*/
    // for each measurement
    for(int nm=0;nm<model.numMeasurements;nm++){
        // select active measurement
        Measurement* measurement = &global.measurements[nm];
        
        // propagate probability distribution until the next measurement
        double mt = 0.0; // time propagated from the last measurement
        //int stepCount = 1; // step count
        while(fabs(mt - measurement->T) > TOL) { 
            growGrid(offsetIndex, iterations, &gridDefinition, &grid, &model);
            
            /*
            check_cfl_condition();
            godunov_method();
            update_prob();
            normalize_tree();
            
            if (step_count % DEL_STEP == 0) { // deletion procedure
                prune_tree();
                normalize_tree(); 
            }
         
            stepCount++;
            */
            // FIXME take account of G.dt
            break; // FIXME remove
        }
        
        // perform Bayesian update for the next measurement
        if(nm < model.numMeasurements -1){
            /*
            meas_up_recursive();
            normalize_tree();
            prune_tree();
            normalize_tree(); 
            */
        }
        break; // FIXME remove
    }
    
    
}

/** Initialize cells */
static __device__ void initializeCell(uint32_t usedIndex, GridDefinition* gridDefinition, Grid* grid, Model* model, Global* global){
    // intialize cells    
    if(usedIndex < grid->usedSize){    
        double prob = 0.0;
        Cell* cell = NULL;
    
        // used list entry
        UsedListEntry* usedListEntry = grid->usedList + usedIndex;
        
        // obtain key (state coordinates)
        uint32_t hashtableIndex = usedListEntry->hashTableIndex;
        int32_t* key = grid->table[hashtableIndex].key;
        
        // compute initial probability    
        prob = gaussProbability(key, gridDefinition, global->measurements);
        
        // update cell          
        cell = getCell(usedIndex, grid);
        cell->new_f = 0;
        
        // compute state
        for(int i=0;i<DIM;i++){
            cell->state[i] = key[i]; // state coordinates
            cell->x[i] = gridDefinition->dx[i] * key[i] + gridDefinition->center[i]; // state value
        }
        
        cell->prob = prob; 
        initializeAdv(gridDefinition, model, cell);
        initializeIkNodes(grid, cell, usedIndex);    
        
        // initialize bounday value
        if(model->useBounds){
            initializeBoundary(cell, model);
        }
    }    
}

/** Calculate gaussian probability at state x given mean and covariance */
static __device__ double gaussProbability(int32_t* key, GridDefinition* gridDefinition, Measurement* measurements){    
    double mInvX[DIM];
    double diff[DIM];
    
    for(int i=0;i<DIM;i++){
        diff[i] = key[i] * gridDefinition->dx[i];
    }
    multiplyMatrixVector( (double*)measurements[0].covInv, diff, mInvX, DIM);
    double dotProduct = computeDotProduct(diff, mInvX, DIM);
    return exp(-0.5 * dotProduct);
}

/** Initialize advection values */
static __device__ void initializeAdv(GridDefinition* gridDefinition, Model* model, Cell* cell){        
    double xk[DIM];
    (*model->callbacks->f)(xk, cell->x, gridDefinition->dx); 

    double sum = 0;
    for(int i = 0; i < DIM; i++){
        cell->v[i] = xk[i];
        sum += fabs(cell->v[i]) / gridDefinition->dx[i];
    }
  
    cell->new_f = 1;
    cell->cfl_dt = 1.0/sum;
    
    /*if(cell->state[0]==0 && cell->state[1]==2 && cell->state[2]==2){
        printf("cell cfl_df %e, v[0] %e, v[1] %e, v[2] %e \n", cell->cfl_dt, cell->v[0], cell->v[1], cell->v[2] );
        } */
}

/**
 * Initialize ik nodes 
 * This function depends on an specific order to fill the usedList ( filled in function initializeHashtable() ).
 */
static __device__ void initializeIkNodes(Grid* grid, Cell* cell, uint32_t usedIndex){        
    uint32_t offset = 1;
    for(int i=DIM-1;;i--){        
        // if is not the first cell in the dimension i        
        if(cell->state[i] > -(int)grid->initialExtent[i]){
            uint32_t iIndex = usedIndex - offset;
            cell->iNodes[i] = iIndex + 1; // reserve 0 for no reference            
        } else {            
            cell->iNodes[i] = 0;
        }
        
        // if is not the last cell in the dimension i        
        if(cell->state[i] < (int)grid->initialExtent[i]){
            uint32_t kIndex = usedIndex + offset;        
            cell->kNodes[i] = kIndex + 1; // reserve 0 for no reference            
        }  else {            
            cell->kNodes[i] = 0;
        }
        
        if(i<=0) break;
        offset *= grid->initialExtent[i] * 2 + 1;
    }    
    cell->ik_f = 1;
}

static __device__ void initializeBoundary(Cell* cell, Model* model){
    double j = (*model->callbacks->j)(cell->x);
    cell->bound_val = j;
}

static __device__ void initializeGridBoundary(int offsetIndex, int iterations, double* localArray, GridDefinition* gridDefinition, Grid* grid, Global* global){
    double boundaryValue = -DBL_MAX;    
    for(int iter=0;iter<iterations;iter++){        
        // index in the used list
        uint32_t usedIndex = (uint32_t)(offsetIndex + iter * blockDim.x);   
        Cell* cell = getCell(usedIndex, grid);        
        if(cell != NULL && cell->bound_val > boundaryValue) boundaryValue = cell->bound_val;
    }
    gridBounds(&gridDefinition->hi_bound, localArray, global->reductionArray, boundaryValue, fmax);
    
    boundaryValue = DBL_MAX;   
    for(int iter=0;iter<iterations;iter++){        
        // index in the used list
        uint32_t usedIndex = (uint32_t)(offsetIndex + iter * blockDim.x);   
        Cell* cell = getCell(usedIndex, grid);        
        if(cell != NULL && cell->bound_val < boundaryValue) boundaryValue = cell->bound_val;
    }
    gridBounds(&gridDefinition->lo_bound, localArray, global->reductionArray, boundaryValue, fmin);
}


/** Normalize probability distribution */
static __device__ void normalizeDistribution(int offsetIndex, int iterations, double* localArray, double* globalArray, Grid* grid){        
    // grid synchronization
    cg::grid_group g = cg::this_grid();      
   
    // store the sum of the cells probability for all the iterations at the local reduction array
    localArray[threadIdx.x] = 0.0;
    for(int iter=0;iter<iterations;iter++){
        uint32_t usedIndex = (uint32_t)(offsetIndex + iter * blockDim.x);           
        Cell* cell = getCell(usedIndex, grid); 
        if(cell != NULL) localArray[threadIdx.x] += cell->prob;
    }
    
    __syncthreads();
    
    // reduction process in shared memory (sequencial addressing)
    for(int s=1;s<blockDim.x;s*=2){
        int indexDst = 2 * s * threadIdx.x;
        int indexSrc = indexDst + s;
        if(indexSrc < blockDim.x){
            localArray[indexDst] += localArray[indexSrc];                        
        }
        __syncthreads();
    }
         
    if(threadIdx.x == 0){        
        // store total sum to global array
        globalArray[blockIdx.x] = localArray[0];       
        
        g.sync();
        
        // reduction process in global memory (sequencial addressing)
        for(int s=1;s<gridDim.x;s*=2){
            int indexDst = 2 * s * blockIdx.x;
            int indexSrc = indexDst + s;
            if(indexSrc < gridDim.x){
                globalArray[indexDst] += globalArray[indexSrc];            
            }
            g.sync();
        }     
    }         
   
    /*if(threadIdx.x == 0){
        printf("prob block sum %e\n", localArray[0]);
        }*/
    
    // at the end, the sum of the probability its at globalArray[0]    
    if(threadIdx.x == 0 && blockIdx.x == 0){
            printf("prob sum %e\n", globalArray[0]);
    }
    
    // update the probability of the cells
    for(int iter=0;iter<iterations;iter++){
        uint32_t usedIndex = (uint32_t)(offsetIndex + iter * blockDim.x);   
        Cell* cell = getCell(usedIndex, grid); 
        if(cell != NULL) cell->prob /= globalArray[0];                    
    }    
}


/** Check CFL condition (computes minimum) */
static __device__ void check_cfl_condition(int offsetIndex, int iterations, double* localArray, double* globalArray, Grid* grid, GridDefinition* gridDef){        
    // grid synchronization
    cg::grid_group g = cg::this_grid();      
   
    // store the sum of the cells probability for all the iterations at the local reduction array
    localArray[threadIdx.x] = 9999.0;
    for(int iter=0;iter<iterations;iter++){
        uint32_t usedIndex = (uint32_t)(offsetIndex + iter * blockDim.x);           
        Cell* cell = getCell(usedIndex, grid); 
        if(cell != NULL) localArray[threadIdx.x] = fmin(localArray[threadIdx.x], cell->cfl_dt);
    }
    
    __syncthreads();
    
    // reduction process in shared memory (sequencial addressing)
    for(int s=1;s<blockDim.x;s*=2){
        int indexDst = 2 * s * threadIdx.x;
        int indexSrc = indexDst + s;
        if(indexSrc < blockDim.x){
	  localArray[indexDst] = fmin(localArray[indexDst],localArray[indexSrc]);                        
        }
        __syncthreads();
    }
         
    if(threadIdx.x == 0){        
        // store total sum to global array
        globalArray[blockIdx.x] = localArray[0];       
        
        g.sync();
        
        // reduction process in global memory (sequencial addressing)
        for(int s=1;s<gridDim.x;s*=2){
            int indexDst = 2 * s * blockIdx.x;
            int indexSrc = indexDst + s;
            if(indexSrc < gridDim.x){
	      globalArray[indexDst] = fmin(globalArray[indexDst],globalArray[indexSrc]);            
            }
            g.sync();
        }     
    }         
   
    /*if(threadIdx.x == 0){
        printf("prob block sum %e\n", localArray[0]);
        }*/
    
    // at the end, the sum of the probability its at globalArray[0]    
    if(threadIdx.x == 0 && blockIdx.x == 0){
            printf("prob sum %e\n", globalArray[0]);
    }

    gridDef->dt = fmin(gridDef->dt, globalArray[0]);

}




/** Set the grid definition bounds with the max and min boundary values of the initial grid cells */
static __device__ void gridBounds(double* output, double* localArray, double* globalArray, double boundaryValue, double(*fn)(double, double) ){
    // grid synchronization
    cg::grid_group g = cg::this_grid();      
    
    // store cell bounday value in the reduction array
    localArray[threadIdx.x] = boundaryValue;
    
    __syncthreads();
    
    // reduction process in shared memory (sequencial addressing)
    for(int s=1;s<blockDim.x;s*=2){
        int indexDst = 2 * s * threadIdx.x;
        int indexSrc = indexDst + s;
        if(indexSrc < blockDim.x){
            localArray[indexDst] = fn(localArray[indexSrc], localArray[indexDst]);                        
        }
        __syncthreads();
    }
        
    if(threadIdx.x == 0){        
        // store total sum to global array
        globalArray[blockIdx.x] = localArray[0];       
        
        g.sync();
        
        // reduction process in global memory (sequencial addressing)   
        for(int s=1;s<gridDim.x;s*=2){
            int indexDst = 2 * s * blockIdx.x;
            int indexSrc = indexDst + s;
            if(indexSrc < gridDim.x){
                globalArray[indexDst] = fn(globalArray[indexSrc], globalArray[indexDst]);            
            }
            g.sync();
        } 
        if(blockIdx.x == 0){
            *output = globalArray[0];
        }
        g.sync();
    }        
}

/** Grow grid */
static __device__ void growGrid(int offsetIndex, int iterations, GridDefinition* gridDefinition, Grid* grid, Model* model){
    for(int iter=0;iter<iterations;iter++){ 
        int usedIndex = (uint32_t)(offsetIndex + iter * blockDim.x); // index in the used list  
        if(usedIndex < grid->usedSize){ 
            Cell* cell = getCell(usedIndex, grid);
            if(cell->prob >= gridDefinition->threshold){
                growGridFromCell(cell, gridDefinition, grid, model);
            }
        }        
    }

    // TODO update ik nodes (at the end of the inserts)    
}

/** Grow grid from one cell */
static __device__ void growGridFromCell(Cell* cell, GridDefinition* gridDefinition, Grid* grid, Model* model){
    for(int dimension=0;dimension<DIM;dimension++){
        if(cell->v[dimension] > 0.0){
            growGridDireccional(cell, dimension, FORWARD, gridDefinition, grid, model);        
        } else if(cell->v[dimension] < 0.0){
            growGridDireccional(cell, dimension, BACKWARD, gridDefinition, grid, model);    
        }
    }
}

/** Grow grid from one cell in one dimension and direction */
static __device__ void growGridDireccional(Cell* cell, int dimension, enum Direction direction, GridDefinition* gridDefinition, Grid* grid, Model* model){
    // check if already exists next face
    uint32_t nextFaceIndex = 0; // initialized to null reference
    int32_t state[DIM]; // state indexes for the new cells
    if(direction == FORWARD) nextFaceIndex = cell->kNodes[dimension];
    else nextFaceIndex = cell->iNodes[dimension];
    
    // create next face if not exists
    if(!nextFaceIndex){
        // create new cell key[dimension] = cell->key[dimension]+direction
        copyKey(cell->state, state);
        state[dimension] += direction;
        createCell(state, gridDefinition, grid, model);
    }
    
    // check edges
    for (int j = 0; j < DIM; j++){
        if(j != dimension){
            if(cell->v[j] > 0.0){
                // create new cell key[dimension] = cell->key[dimension] = key[dimension]+direction & cell->key[j] = cell->key[j]+1
                copyKey(cell->state, state);
                state[dimension] += direction;
                state[j] +=1;
                createCell(state, gridDefinition, grid, model);
            } else if(cell->v[j] < 0.0){
                // create new cell key[dimension] = cell->key[dimension] = key[dimension]+direction & cell->key[j] = cell->key[j]-1
                copyKey(cell->state, state);
                state[dimension] += direction;
                state[j] -=1;
                createCell(state, gridDefinition, grid, model);
            }
        }
    }    
}

/** Create new cell in the grid */
static __device__ void createCell(int32_t* state, GridDefinition* gridDefinition, Grid* grid, Model* model){
    Cell cell;
    
    // compute state
    for(int i=0;i<DIM;i++){
        cell.state[i] = state[i]; // state coordinates
        cell.x[i] = gridDefinition->dx[i] * state[i] + gridDefinition->center[i]; // state value
    }
        
    cell.prob = 0.0; 
    cell.new_f = 0;
    cell.ik_f = 0;
    initializeAdv(gridDefinition, model, &cell);
    // TODO initialize ctu[] y dcu
    
    // FIXME shyncro, critical region
    insertCell(&cell, grid);
    
}
