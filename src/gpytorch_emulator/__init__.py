from importlib.metadata import version
__version__ = version("high_dim_out_gp")  

from gpytorch_emulator.ppgasp import MoGP_GPytorch, PCA_MoGP_GPytorch
from gpytorch_emulator.svgp_lmc import SVGP_LMC, MultiTask_GP
