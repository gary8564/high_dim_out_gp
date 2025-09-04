# gpytorch

## Description
[GPytorch](https://gpytorch.ai/), a Gaussian Process framework based on Pytorch. One advantage is that, compared to RobustGaSP, it has a lot more degrees of flexibility to develop the Gaussian Process model. For instance, it can easily be extended to integrate with STOA deep learning approach. And since it is fully integrated with PyTorch, CUDA for GPU acceleration is inheritently supported.

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
git clone git@git-ce.rwth-aachen.de:mbd/eccr/gpytorch.git
```

### 2. Installation
Using conda:
```bash
conda env create -f environment.yaml
conda activate gp_emulator
```

### 3. Repository Structure
```text
.
├── data/
│   ├── high_dim_input_prob/         ← case studies dataset for high-dimensional input problem
│   └── high_dim_output_prob/        ← case studies dataset for high-dimensional output problem
├── literatures/                     ← relevant papers
├── docs/
│   └── data.md
├── demo/           
│   ├── high_dim_input_prob.py       ← demo of high-dimensional input problem
│   └── high_dim_output_prob.py      ← demo of high-dimensional output problem
├── src/
│   ├── ppgasp.py                    ← GPytorch implementation of the concept of PPGaSP
│   ├── dkl.py                       ← Deep kernel learning
│   └── utils.py                     ← Utility functions
├── tests/                           ← unit & integration tests
├── environment.yaml
├── .gitignore
└── README.md                        
```

## Data
The detailed description and the source of the data used here are described [here](./docs/data.md).

## Usage
1. High-dimensional input problem
```bash
python demo/high_dim_input_prob.py --case-study "tsunami_tokushima" --model "gpytorch" --dim-reduction
```
2. High-dimensional output problem
```bash
python demo/high_dim_output_prob.py --case-study "acheron" --model "ppgasp" --dim-reduction
```

