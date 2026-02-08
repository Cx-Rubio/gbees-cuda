
# GBEES-CUDA

GBEES-CUDA is the CUDA implementation of the Grid-based Bayesian Estimation Exploiting Sparsity (GBEES) algorithm.\
GBEES is a method for propagating uncertainty in systems with nonlinear dynamics. It also allows Bayesian updates of the probability density function based on discrete measurements.

## Download the source code
In a computer with the git client installed, clone the repository by using:

```
git clone https://github.com/Cx-Rubio/gbees-cuda.git
```

Alternatively you can download the code in a zip file from the downloads page: 

https://github.com/Cx-Rubio/gbees-cuda/archive/refs/heads/main.zip

## Launch Configuration

The launch configuration is centralized in the file ***config.h***, where the following options can be set:
- **DIM**: The dimension of the state vector of the problem
- **BLOCKS**: Number of CUDA blocks to launch
- **THREADS_PER_BLOCK** Number of threads per block
- **CELLS_PER_THREAD** Number of cells that should process each thread
- **ENABLE_LOG**: Enable logs (comment out to disable logs)
- **HASH_TABLE_RATIO**: Size of the hashtable with respect the maximum number of cells
- **SINGLE_PRECISION_SNAPSHOTS** Left uncommented for single precision in the snapshots; comment it out for double precision

The GPU launch configuration is determined by three key parameters: the number of blocks, the number of threads per block, and the number of cells each thread processes.
The objective of the launch configuration is to maximize the GPU occupancy. Since the program uses a Cooperative Kernel, maximum occupancy is achieved by launching a total number of threads equal to the GPU’s maximum simultaneous threads.
If the maximum grid size exceeds this capacity, each thread must process multiple cells, requiring the parameter for cells processed per thread to be set to a value greater than one.
Therefore, the launch configuration strategy is to keep the number of cells processed by each thread as low as possible. If the model is sufficiently large, the product of the number of blocks and the number of threads per block should equal the GPU’s maximum thread capacity.

## Model configuration

The models (the specific problems to solve), are defined in the models folder. Each model should populate the model structure similarly to the CPU implementation documented [here](https://bhanson10.github.io/gbees.pdf).
Additionally, the *recordDivider* field allows saving only a fraction of the record distributions, while the *recordSelected* field specifies which fraction of the total records is recorded.

## Compile
The software is developed and tested only in Linux. Before compile it is needed to have installed the NVIDIA CUDA toolkit. In Ubuntu can be done by:
```
sudo apt install nvidia-cuda-toolkit
```
Edit the ***_makefile_*** and adjust the *CUDAFLAGS* according to the architecture level of the target GPU. The option *-maxrregcount = 32* should be included to limit the number of registers used by each thread.

Compile to generate the executable by using:

```
make clean; make all
```

## Execute

After running the `make all` command, the program can be executed with:
```
./GBEES
```

Currently, four example models are implemented. Users can select a model by editing the ***main.cu*** file, uncommenting the desired model configuration, and commenting out the others.

```
    configureLorenz3D(&model);
    //configurePcr3bp(&model);
    //configureCr3bp(&model);
    //configureLorenz6D(&model);
```
The above configuration solves the original 3D Lorenz model.\
Other models can added by the user in the models folder.

## Directory structure and file formats
The location of the measurement files and the results folder for each model can be configured by modifying the model definition in the source code. For the example models, the measurements folder is configured as ***./measurements/$modelName***, and the output is configured as the ***./results*** folder.

This section offers a brief overview of the measurement and output formats. For a comprehensive explanation, please consult the detailed documentation available [here](https://bhanson10.github.io/gbees.pdf).

### 1- Measurement files

The structure of a measurement `.txt` file must strictly adhere to the required format for GBEES to process it correctly. Measurements are zero-indexed:
- **Initial uncertainty** (a priori): `measurement_0.txt`
- **Subsequent measurements**: `measurement_1.txt`, `measurement_2.txt`, etc. 
- For example,  below is the structure of the initial measurement file (`measurement_0.txt` )for the 3D Lorenz example, located in `/gbees-cuda/measurements/Lorenz3D/measurement_0.txt`:

```
x (LU) y (LU) z (LU)
-11.50 -10.00 9.5000 

Covariance(x, y, z) 
1.000000000000000000 0.000000000000000000 0.000000000000000000
0.000000000000000000 1.000000000000000000 0.000000000000000000
0.000000000000000000 0.000000000000000000 1.000000000000000000
T (TU) 
1
```

In which:
1. **Line 1**: Labels for state coordinates (units in parentheses, e.g., `LU` for length units)
2. **Line 2**: Mean vector for the measurement (space-separated values)
3. **Line 3**: Skipped (empty line)
4. **Line 4**: Covariance matrix label, specifying variables (e.g., `Covariance(x, y, z)`)
5. **Lines 5-7**: Covariance matrix (d × d, space-separated values)
6. **Line 8**: Skipped (empty line)
7. **Line 9**: Period label, specifying the time interval to the next measurement (e.g., `T (TU)` where `TU` represents time units)
8. **Line 10**: Period value

- Labels (Lines 1, 4, and 9) are flexible but must exist. 
- Skipped lines (Lines 3 and 8) must remain empty.
- All values are **space-separated**. 
- For the initial measurement (`measurement_0.txt`):
	- The mean vector and covariance matrix must match the dimensionality of the dynamics model \( f \). 
- For subsequent measurements (`measurement_1.txt`, etc.): 
	- The mean vector and covariance matrix must match the dimensionality of the measurement model \( h \). 
- The dimensionality of the mean vector and covariance matrix can vary as long as the format is consistent.


### 2- Output files (PDF snapshots)

GBEES-CUDA outputs non-Gaussian Probability Density Functions (PDFs) as `.txt` files, just like it reads measurements in `.txt` format. The PDFs are separated by measurement updates. PDFs are stored in the specified output folder with the name ***PX_pdf_Y.pdf***, where X is the number of mesurement (starting with zero) and Y is the number of snapshot for this measurement (also starting with zero).

From the 3D Lorenz example, the first PDF file saved in `/gbees-cuda/results/` might look like:
```
0.000000
7.2596185287e-06 -1.4500000000e+01 -1.2000000000e+01 8.5000000000e+00
1.0562682633e-05 -1.4500000000e+01 -1.2000000000e+01 9.0000000000e+00

```
in which:
- **First line**: Simulation time of the PDF.
- **Other lines**: 
	- **First column**: Probability value.
	- **Remaining columns**: Grid cell state values.

in the example:
- **Simulation time**: `0.000000`
- **First cell**: 
	- Probability: `7.2596185287e-06` 
	- State values (x, y, z): `-14.5, -12.0, 8.5`
- **Second cell**: 
	- Probability: `1.0562682633e-05`
	- State values (x, y, z): `-14.5, -12.0, 9.0`

### References
 A detailed description of the GBEES algorithm is available in:\
*Efficient Grid-Based Bayesian Estimation of Nonlinear Low-Dimensional Systems with Sparse, Non-Gaussian PDFs," Automatica 48 (7) (2012) 1286–1290, by T. R. Bewley, et al.* https://doi.org/10.1016/j.automatica.2012.02.039 \

 The rationale of the CUDA implementation can be found in:\
*GBEES-GPU: A High-Dimensional Efficient Parallel GPU Algorithm for Nonlinear Uncertainty Propagation, by B. Hanson, et al.*
