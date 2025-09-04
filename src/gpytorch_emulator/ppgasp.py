import torch
import time
import math
import numpy as np
import gpytorch
import logging
from sklearn.preprocessing import StandardScaler  
from tqdm import tqdm
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.kernels import ScaleKernel, MultitaskKernel
from gpytorch.kernels.keops import RBFKernel, MaternKernel
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from gpytorch.constraints import GreaterThan


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchIndependentMultioutputGPModel(ExactGP):
    """
    This model implements a Batch Independent Multioutput GP,
    where the outputs are treated as independent (in batch) but share
    the same hyperparameters. This mimics the PPGaSP approach (shared kernel,
    fixed identity task covariance) and greatly reduces computational cost.
    """
    def __init__(self, train_x, train_y, likelihood, kernel_type='matern_5_2', mimic_ppgasp=False):
        """
        Args:
            train_x: training inputs of shape (n, d)
            train_y: training outputs of shape (n, num_tasks)
            likelihood: a MultitaskGaussianLikelihood instance
            kernel_type: one of 'matern_5_2', 'matern_3_2', or 'rbf'
            mimic_ppgasp: whether to mimic the conceptual idea of PPGaSP. By default, False.
            
        """
        super().__init__(train_x, train_y, likelihood)
        self.register_buffer('train_x_buf', train_x)  
        self.register_buffer('train_y_buf', train_y)
        
        self.num_tasks = train_y.shape[-1]
        self.input_dim = train_x.shape[-1]
        self.mimic_ppgasp = mimic_ppgasp
        # Create batched mean (num_tasks,)
        # self.mean_module = ConstantMean(batch_shape=torch.Size([self.num_tasks]),
        #                                 constant_prior=NormalPrior(0.0, 1.0)) if self.mimic_ppgasp else \
        #                                     ConstantMean(batch_shape=torch.Size([self.num_tasks]))
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=self.num_tasks) if self.mimic_ppgasp else \
                                            ConstantMean(batch_shape=torch.Size([self.num_tasks]))

        
        empirical_scale = (train_x.max(dim=0).values - train_x.min(dim=0).values).mean().item()
        shape_param = 2.5  # Slightly lower shape parameter for better regularization
        rate_param = 5.0 / empirical_scale  # Scale based on feature range

        if kernel_type == 'matern_5_2':
            base_kernel = MaternKernel(nu=2.5, ard_num_dims=self.input_dim,
                                           lengthscale_prior=GammaPrior(shape_param, rate_param)) if self.mimic_ppgasp else \
                                               MaternKernel(nu=2.5, ard_num_dims=self.input_dim, 
                                                            batch_shape=torch.Size([self.num_tasks]), 
                                                            lengthscale_constraint=gpytorch.constraints.GreaterThan(1e-4))
        elif kernel_type == 'matern_3_2':
            base_kernel = MaternKernel(nu=1.5, ard_num_dims=self.input_dim,
                                           lengthscale_prior=GammaPrior(shape_param, rate_param)) if self.mimic_ppgasp else \
                                               MaternKernel(nu=1.5, ard_num_dims=self.input_dim, 
                                                            batch_shape=torch.Size([self.num_tasks]), 
                                                            lengthscale_constraint=gpytorch.constraints.GreaterThan(1e-4))
        elif kernel_type == 'rbf':
            base_kernel = RBFKernel(ard_num_dims=self.input_dim, lengthscale_prior=GammaPrior(shape_param, rate_param)) if self.mimic_ppgasp else \
                RBFKernel(ard_num_dims=self.input_dim, batch_shape=torch.Size([self.num_tasks]), lengthscale_constraint=gpytorch.constraints.GreaterThan(1e-4))
        else:
            raise ValueError("Unsupported kernel_type: {}".format(kernel_type))

        # Wrap the base kernel with a ScaleKernel to include an output-scale parameter
        # shared_kernel = ScaleKernel(base_kernel, outputscale_prior=GammaPrior(2.0, 0.15))
        coregionalized_kernel = MultitaskKernel(base_kernel, num_tasks=self.num_tasks, rank=0)
        # Create the batched kernel (num_tasks,)
        # self.covar_module = ScaleKernel(
        #     shared_kernel, 
        #     batch_shape=torch.Size([self.num_tasks])
        # ) if self.mimic_ppgasp else ScaleKernel(
        #     base_kernel,
        #     batch_shape=torch.Size([self.num_tasks])
        # )
        self.covar_module = ScaleKernel(coregionalized_kernel) if self.mimic_ppgasp else ScaleKernel(base_kernel,batch_shape=torch.Size([self.num_tasks]))
        
        # Initialize lengthscales with dimension-aware values
        self._initialize_lengthscales(train_x, train_y)
        
        if self.mimic_ppgasp:
            # Tie the parameters across tasks so that each output uses the same hyperparameters.
            self.tie_parameters_across_tasks()

    def _get_inner_kernel(self, kernel):
        """
        Peel off any ScaleKernel wrappers, then if it's a MultitaskKernel
        dive into its .data_covar_module, and repeat until we hit
        the actual RBF/Matern kernel.
        """
        while True:
            if isinstance(kernel, ScaleKernel):
                kernel = kernel.base_kernel
            elif isinstance(kernel, MultitaskKernel):
                kernel = kernel.data_covar_module
            else:
                break
        return kernel
        
    def _initialize_lengthscales(self, train_x, train_y):
        """Initialize lengthscales based on input data characteristics"""
        # Calculate feature ranges for better initialization
        feature_ranges = train_x.max(dim=0).values - train_x.min(dim=0).values
        
        # Find the actual MaternKernel or RBFKernel you wrapped above
        inner = self._get_inner_kernel(self.covar_module)

        if self.mimic_ppgasp:
            n_samples = train_x.shape[0]
            # scaling_factors = feature_ranges / math.sqrt(n)
            input_dim = float(self.input_dim)
            scaling_factors = feature_ranges / (n_samples ** (1.0 / input_dim))
            # Initialize the base kernel's lengthscale with dimension-specific values
            init_lengthscales = 50.0 * scaling_factors
            inner.initialize(lengthscale=init_lengthscales.unsqueeze(0))
            # Register dimension-specific lower bounds
            # lower_bounds = -torch.log(torch.tensor(0.1)) / feature_ranges
            lower_bounds = -torch.log(torch.tensor(0.1)) / (feature_ranges * input_dim)
            raw_lb = torch.log(torch.exp(lower_bounds) - 1.0)
            inner.register_constraint("raw_lengthscale", GreaterThan(raw_lb))
        else:    
            median_range = feature_ranges.median().item()
            init_lengthscale = math.sqrt(self.input_dim) * median_range
            inner.lengthscale = init_lengthscale
            # Initialize outputscale based on data variance
            init_outputscale = train_y.var(dim=0).mean().sqrt()
            self.covar_module.outputscale = init_outputscale

    def tie_parameters_across_tasks(self):
        """
        Force the kernel parameters to be shared across all tasks.
        This mimics the PPGaSP assumption that the task covariance is the identity.
        """
        # Tie the lengthscale
        inner_kernel = self._get_inner_kernel(self.covar_module)
        base_lengthscale = inner_kernel.lengthscale.detach().clone()
        # common_lengthscale = base_lengthscale.mean(dim=0)
        # inner_kernel.lengthscale.data = common_lengthscale.unsqueeze(0).expand(self.num_tasks, 1, common_lengthscale.size(0)).clone()
        common_lengthscale = base_lengthscale.mean(dim=0, keepdim=True)
        inner_kernel.lengthscale.data = common_lengthscale.expand_as(base_lengthscale)
        
        # Tie the outputscale
        base_outputscale = self.covar_module.outputscale.detach().clone()
        common_outputscale = base_outputscale.mean()
        self.covar_module.outputscale.data = common_outputscale.expand_as(self.covar_module.outputscale).clone()

        # # Tie the mean constant
        # base_mean = self.mean_module.constant.detach().clone()
        # common_mean = base_mean.mean()
        # self.mean_module.constant.data = common_mean.expand_as(self.mean_module.constant).clone()
          
    def forward(self, x):
        # x has shape (n, d)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        if self.mimic_ppgasp:
            # For mimic_ppgasp=True: Uses MultitaskMean and MultitaskKernel 
            # This should work directly with MultitaskMultivariateNormal
            return MultitaskMultivariateNormal(mean_x, covar_x)
        else:
            # For mimic_ppgasp=False: Uses batched mean and batched kernels
            # Need to construct from batch of MVNs
            return MultitaskMultivariateNormal.from_batch_mvn(
                MultivariateNormal(mean_x, covar_x)
            )

class MoGP_GPytorch:
    def __init__(self, device: str, kernel_type: str = 'matern_5_2', mimic_ppgasp: bool = True):
        self.device = torch.device(device)
        self.kernel_type = kernel_type
        self.mimic_ppgasp = mimic_ppgasp
    
    def train(self,
              train_X: np.ndarray | torch.Tensor, train_Y: np.ndarray | torch.Tensor, 
              num_epochs: int = 200, lr: float = 0.1, 
              optim: str = "adam", enable_scheduler: bool = False):
        # 1) LBFGS/mixed not allowed for MoGP_GPytorch
        if optim in ("lbfgs","mixed") and not isinstance(self, PCA_MoGP_GPytorch):
           raise ValueError(
               f"`{self.__class__.__name__}.train` only supports 'adam' or 'adamw'.\n"
               "If you want to use 'lbfgs' or 'mixed', please use PCA_MoGP_GPytorch."
               )
        # Training setup
        if isinstance(train_X, np.ndarray):
            self.train_X = torch.from_numpy(train_X).to(torch.float32)
        else:
            self.train_X = train_X
        if isinstance(train_Y, np.ndarray):
            self.train_Y = torch.from_numpy(train_Y).to(torch.float32)
        else:
            self.train_Y = train_Y
        self.train_X, self.train_Y = self.train_X.to(self.device), self.train_Y.to(self.device)
        # Define the multitask likelihood.
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=self.train_Y.shape[1], rank=0).to(self.device)
        # Instantiate the improved batched model.
        self.model = BatchIndependentMultioutputGPModel(self.train_X, self.train_Y, self.likelihood,
                                                kernel_type=self.kernel_type, mimic_ppgasp=self.mimic_ppgasp).to(self.device)
        self.model.train()
        self.likelihood.train()
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        # Optimizer
        fine_tune_optimizer = None
        if optim == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optim == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        elif optim == "lbfgs":
            optimizer = torch.optim.LBFGS(self.model.parameters(), 
                                          lr=lr,
                                          max_iter=30,
                                          history_size=10,
                                          tolerance_grad=1e-6,
                                          tolerance_change=1e-8,
                                          line_search_fn='strong_wolfe')
            def closure():
               optimizer.zero_grad()
               output = self.model(self.train_X)
               # The loss includes both the negative log likelihood and the negative log prior terms.
               loss = -mll(output, self.train_Y)
               loss.backward()
               return loss
        elif optim == "mixed":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            fine_tune_optimizer = torch.optim.LBFGS(self.model.parameters(), lr=0.1, max_iter=100)
            def fine_tune_closure():
               fine_tune_optimizer.zero_grad()
               output = self.model(self.train_X)
               loss = -mll(output, self.train_Y)
               loss.backward()
               return loss
        else:
            raise ValueError("Optimizer are only supported with `adam`, `adamw`, `lbfgs`, or `mixed`(i.e. Adam + LBFGS).")
        
        # Scheduler
        scheduler = None
        if enable_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
        
        # Training
        num_epochs = num_epochs
        start_time = time.time()
        for epoch in tqdm(range(num_epochs), desc="training..."):
            if optim == "lbfgs":     
                optimizer.step(closure)
                if epoch  % 10 == 0:
                   current_loss = closure().item()
                   print(f"Epoch {epoch}/{num_epochs}, Loss: {current_loss:.3f}")
            else:    
                optimizer.zero_grad()
                output = self.model(self.train_X)
                loss = -mll(output, self.train_Y)
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.3f}")
                loss.backward()
                optimizer.step()
                
            if scheduler is not None:
                scheduler.step(loss.item())
            
        if fine_tune_optimizer is not None:
            # LBFGS fine-tuning after training with ADAM
            if enable_scheduler:
                fine_tune_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(fine_tune_optimizer, mode='min', factor=0.5, patience=5)
            num_finetune_epochs = 20
            for epoch in tqdm(range(num_finetune_epochs), desc="LBFGS fine-tuning"):
                fine_tune_optimizer.step(fine_tune_closure)
                fine_tune_scheduler.step(loss.item())
                current_loss = fine_tune_closure().item()
                print(f"Fine-tuning Epoch {epoch+1}/{num_finetune_epochs}, Loss: {current_loss:.3f}")
                if fine_tune_optimizer.param_groups[0]['lr'] < 1e-6:
                    print("Learning rate too small, stopping fine-tuning")
                    break
        
        training_time = time.time() - start_time
        print(f"Training GPytorch takes {training_time:.3f} s")
        return training_time
    
    def predict(self, test_X: np.ndarray | torch.Tensor):
        if isinstance(test_X, np.ndarray):
            self.test_X = torch.from_numpy(test_X).to(torch.float32)
        else:
            self.test_X = test_X
        self.test_X = self.test_X.to(self.device)
        self.model.eval()
        self.likelihood.eval()
        start_time = time.time()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(self.test_X))
        infer_time = time.time() - start_time    
        mean = predictions.mean.cpu().detach().numpy()
        std = predictions.stddev.cpu().detach().numpy()
        lower, upper = predictions.confidence_region()
        lower = lower.cpu().detach().numpy()
        upper = upper.cpu().detach().numpy()
        return mean, std, lower, upper, infer_time
    
class PCA_MoGP_GPytorch(MoGP_GPytorch):
    def __init__(self, 
                 output_dim_reducer,
                 device: str, 
                 kernel_type: str = 'matern_5_2', 
                 mimic_ppgasp: bool = False):
        super().__init__(device, kernel_type, mimic_ppgasp)
        self.output_dim_reducer = output_dim_reducer
        self.scaler = StandardScaler()
    
    def preprocess_dim_reduction(self, 
                                 train_X: np.ndarray,
                                 train_Y: np.ndarray, 
                                 test_X: np.ndarray, 
                                 test_Y: np.ndarray):
        # Store original dimensions
        self.original_input_dim = train_X.shape[1]
        self.original_output_dim = train_Y.shape[1]
        
        # Data standardization
        training_dataset = np.hstack((train_X, train_Y))
        training_dataset_scaled = self.scaler.fit_transform(training_dataset)
        test_dataset = np.hstack((test_X, test_Y))
        test_dataset_scaled = self.scaler.transform(test_dataset)
        train_X_scaled = training_dataset_scaled[:, :self.original_input_dim]
        train_Y_scaled = training_dataset_scaled[:, self.original_input_dim:]
        test_X_scaled = test_dataset_scaled[:, :self.original_input_dim]
        test_Y_scaled = test_dataset_scaled[:, self.original_input_dim:]
        
        # Apply output PCA 
        train_Y_scaled_reduced = self.output_dim_reducer.fit_transform(train_Y_scaled)
        # Verify reduced dimensions
        assert train_Y_scaled_reduced.shape[1] == self.output_dim_reducer.reducer.n_components, \
            f"Output PCA reduced to {train_Y_scaled_reduced.shape[1]} components, expected {self.output_dim_reducer.reducer.n_components}"
        print(f"Reduced output dimension to {train_Y_scaled_reduced.shape[1]}.")
        test_Y_scaled_reduced = self.output_dim_reducer.transform(test_Y_scaled)
        return train_X_scaled, train_Y_scaled_reduced, test_X_scaled, test_Y_scaled_reduced
    
    def postprocess_invert_back(self, predictions_mean: np.ndarray):
        # Transform reduced dimension back to original dimension 
        reconstructed_bands = self.output_dim_reducer.inverse_transform(predictions_mean)
        print(f"Inverse transform back to original output dimension: {reconstructed_bands.shape[1]}.")

        # Unnormalize data
        mu = self.scaler.mean_[self.original_input_dim:]
        sigma = self.scaler.scale_[self.original_input_dim:]
        predictions = reconstructed_bands * sigma + mu
        return predictions
        
        
        

            