# High-Dimensional Output Gaussian Processes

## Description
This repository implements high-dimensional output Gaussian Process emulators using [GPyTorch](https://gpytorch.ai/) and [RobustGaSP](https://git.rwth-aachen.de/mbd/psimpy), Gaussian Process frameworks based on PyTorch and R, respectively. GPyTorch provides flexibility and GPU acceleration, while RobustGaSP offers robust optimization and massive output dimensions in geostatistical applications. One 
advantage using GPytorch is that, compared to RobustGaSP, it has a lot more degrees of flexibility to develop the Gaussian Process model and can easily be extended to 
integrate with STOA deep learning approach.

More comparison between RobustGaSP and GPytorch are listed as the table below:
| Feature                     | PPGaSP (RobustGaSP)                                                                                 | GPyTorch                                                                       |
|-----------------------------|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| **Language**                | R (backend in C++)                                                                                  | Python (PyTorch-based)                                                          |
| **Hardware support**        | CPU only, optimized for single-threaded performance via compiled C++ (via Rcpp and RcppArmadillo)  | Fully integrated with PyTorch, supports CUDA for GPU acceleration              |
| **Gaussian Process Type**   | Robust & Partial Independent Multi-output Gaussian Process                                          | Full, Approximate, and Multi-task Gaussian Process models                      |
| **Optimization Method**     | L-BFGS, Nelder-Mead, Brent (via `nloptr` in R, fast second-order optimization)                     | Adam, L-BFGS, Cosine Annealing, and other gradient-based optimizers            |
| **Matrix computation**      | Uses Cholesky decomposition and avoids direct matrix inversion for efficiency                       | Uses linear operators (Kronecker, Toeplitz, or KeOps-based) to optimize matrix operations |
| **Inference Speed**         | Extremely fast due to reduced matrix inversion and composite likelihood approximations              | Slower for exact GP, but scalable using variational methods and GPU support    |
| **Multi-output GP support** | Assumes partial independence between outputs with shared covariance structure, reducing complexity | Fully flexible multitask GPs (LMC, Independent, Variational Multi-Output GPs)   |
| **Target User**             | Well-suited for geostatistics with high-dimensional outputs                                         | Highly flexible, can be easily combined with SOTA deep learning research       |


## Getting started

### 1. Clone the repository:
```bash
git clone --recurse-submodules https://github.com/gary8564/high_dim_out_gp.git
```

### 2. Installation
Using conda:
```bash
conda env create -f environment.yaml
conda activate high_dim_out_gp
```

### 3. Repository Structure
```text
.
├── docs/
│   └── data.md                      # Documentation for data sources and descriptions
├── demo/                            # (empty) Demo notebooks and case studies
├── scripts/
│   └── fetch_data.sh               # Script to fetch required datasets
├── src/
│   ├── gpytorch_emulator/          # Main GPyTorch-based emulator implementations
│   │   ├── __init__.py
│   │   ├── ppgasp.py              # GPyTorch implementation of PPGaSP concept
│   │   └── svgp_lmc.py            # Sparse Variational GP with Linear Model of Coregionalization
│   ├── pca_psimpy/                # PCA + PSIMPy integration for dimension reduction
│   │   ├── src/psimpy/            # PSIMPy library with emulators, samplers, etc.
│   │   ├── docs/                  # Documentation and examples
│   │   └── tests/                 # PSIMPy-specific tests
│   └── utils/                     # Utility functions
│       ├── __init__.py
│       ├── error_metrics.py       # Performance evaluation metrics
│       ├── plot.py               # Plotting utilities
│       └── preprocess.py         # Data preprocessing functions
├── tests/                         # Unit & integration tests for main emulators
│   ├── test_ppgasp.py            # Tests for Multi-output GP models
│   ├── test_svgp_lmc.py          # Tests for Sparse Variational GP models
│   └── test_pca_ppgasp.py        # Tests for PCA + PPGaSP models
├── environment.yaml               # Conda environment specification
├── pyproject.toml                # Python project configuration
├── .gitignore
└── README.md                        
```

## Data
The detailed description and the source of the data used here are described [here](./docs/data.md).

## Available Models

This repository implements several Gaussian Process models for high-dimensional output problems:

### 1. Multi-output Gaussian Processes (`ppgasp.py`)
- **MoGP_GPytorch**: Multi-output GP with independent outputs or shared hyperparameters
- **PCA_MoGP_GPytorch**: PCA-based dimension reduction + Multi-output GP
- **BatchIndependentMultioutputGPModel**: Batch processing for multiple independent GPs

### 2. Sparse Variational Gaussian Processes (`svgp_lmc.py`)
- **SVGP_LMC**: Sparse Variational GP with Linear Model of Coregionalization
- **MultiTask_GP**: Adaptive multi-task GP with automatic parameter selection

### 3. PCA + PPGaSP Integration (`pca_psimpy/`)
- **PCAPPGaSP**: PCA dimension reduction + Projected Predictive Gaussian Stochastic Process
- **PCAScalarGaSP**: PCA dimension reduction for scalar output problems

## Usage

