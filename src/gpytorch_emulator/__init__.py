from importlib.metadata import version
__version__ = version("high_dim_out_gp")  

<<<<<<< HEAD
from gpytorch_emulator.ppgasp import MoGP_GPytorch, PCA_MoGP_GPytorch
from gpytorch_emulator.svgp_lmc import SVGP_LMC, MultiTask_GP
=======
from gpytorch_emulator.ppgasp import BiGP, PCA_BiGP
from gpytorch_emulator.svgp_lmc import SVGP_LMC, MultiTaskGP
from gpytorch_emulator.lbfgs import LBFGS, FullBatchLBFGS
>>>>>>> c066404 (Pulled down update to pca_psimpy)
