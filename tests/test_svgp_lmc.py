import pytest
import numpy as np
import torch
import logging
from gpytorch_emulator.svgp_lmc import MultiTask_GP, SVGP_LMC

# Suppress logging during tests
logging.getLogger().setLevel(logging.WARNING)


def generate_synthetic_high_dim_output_data(n_samples=100, input_dim=5, output_dim=50, noise_level=0.1, random_seed=42):
    """
    Generate synthetic data for high-dimensional output problems.
    
    Parameters:
    -----------
    n_samples : int
        Number of training samples
    input_dim : int  
        Input dimensionality
    output_dim : int
        High output dimensionality  
    noise_level : float
        Noise level in the output
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    X_train, y_train, X_test, y_test : np.ndarray
        Training and testing data
    """
    np.random.seed(random_seed)
    
    # Generate input data
    X_train = np.random.uniform(-1.5, 1.5, (n_samples, input_dim))
    X_test = np.random.uniform(-1.5, 1.5, (n_samples // 2, input_dim))
    
    # Create a complex multi-output function with clear correlation structure
    # This is important for LMC to work well
    latent_dim = min(12, output_dim // 3)  # True latent dimensionality
    
    def complex_multioutput_function(X):
        # Generate latent functions with clear patterns
        latent_outputs = np.zeros((X.shape[0], latent_dim))
        
        for i in range(latent_dim):
            if i == 0:
                # Smooth periodic function
                latent_outputs[:, i] = 2.0 * np.sin(2 * np.pi * X[:, 0]) + 0.5 * X[:, min(1, input_dim-1)] ** 2
            elif i == 1 and input_dim > 1:
                # Smooth cosine function
                latent_outputs[:, i] = 1.5 * np.cos(np.pi * X[:, 1]) + 0.3 * X[:, 0]
            elif i == 2 and input_dim > 2:
                # Gaussian-like function
                latent_outputs[:, i] = 2.0 * np.exp(-0.5 * (X[:, 2] ** 2)) + 0.2 * X[:, 0] * X[:, 1]
            elif i == 3 and input_dim > 3:
                # Hyperbolic tangent
                latent_outputs[:, i] = 1.8 * np.tanh(X[:, 3]) + 0.1 * np.sum(X[:, :2], axis=1)
            elif i == 4 and input_dim > 4:
                # Polynomial function
                latent_outputs[:, i] = X[:, 4] ** 3 + 0.4 * np.sin(X[:, 0] + X[:, 1])
            else:
                # Additional smooth latent functions
                idx1 = i % input_dim
                idx2 = (i + 1) % input_dim
                freq = 0.5 + 0.2 * i  # Varying frequency
                latent_outputs[:, i] = np.sin(freq * np.pi * X[:, idx1]) + 0.3 * np.cos(freq * np.pi * X[:, idx2])
        
        # Create structured loading matrix with groups of related outputs
        np.random.seed(random_seed + 1)  # Different seed for loading matrix
        loading_matrix = np.zeros((output_dim, latent_dim))
        
        # Create groups of outputs that share similar latent functions
        group_size = max(1, output_dim // latent_dim)  # Ensure group_size is at least 1
        for i in range(latent_dim):
            start_idx = i * group_size
            end_idx = min((i + 1) * group_size, output_dim)
            
            # Ensure we don't have empty groups
            if end_idx > start_idx:
                # Strong loading on primary latent function
                loading_matrix[start_idx:end_idx, i] = 1.0 + 0.3 * np.random.randn(end_idx - start_idx)
            
                # Weaker loadings on other latent functions
                for j in range(latent_dim):
                    if j != i:
                        loading_matrix[start_idx:end_idx, j] = 0.2 * np.random.randn(end_idx - start_idx)
        
        # Normalize loadings to control output variance
        loading_matrix = loading_matrix / np.sqrt(latent_dim)
        
        # Map latent outputs to observed outputs
        y = latent_outputs @ loading_matrix.T
        
        return y
    
    # Generate outputs
    y_train = complex_multioutput_function(X_train) + noise_level * np.random.randn(n_samples, output_dim)
    y_test = complex_multioutput_function(X_test) + noise_level * np.random.randn(n_samples // 2, output_dim)
    
    return X_train, y_train, X_test, y_test


class TestSVGP_LMC:
    """Test the SVGP_LMC model standalone"""
    
    def test_svgp_lmc_initialization(self):
        """Test SVGP_LMC initialization"""
        input_dim = 4  
        num_tasks = 20  
        rank = 4       
        num_inducing = 20  
        use_pca_init = True
        
        model = SVGP_LMC(
            input_dim=input_dim,
            num_tasks=num_tasks,
            rank=rank,
            num_inducing=num_inducing,
            use_pca_init=use_pca_init
        )
        
        assert model.rank == rank
        assert model.num_tasks == num_tasks
        assert model.use_pca_init == use_pca_init
        
        # Check variational strategy
        assert hasattr(model, 'variational_strategy')
        assert model.variational_strategy.num_latents == rank
        assert model.variational_strategy.num_tasks == num_tasks
    
    def test_svgp_lmc_forward(self):
        """Test SVGP_LMC forward pass"""
        input_dim = 3  
        num_tasks = 8  
        rank = 4       
        num_inducing = 16 
        batch_size = 10  
        
        model = SVGP_LMC(
            input_dim=input_dim,
            num_tasks=num_tasks,
            rank=rank,
            num_inducing=num_inducing
        )
        
        x = torch.randn(batch_size, input_dim)
        output = model(x)
        
        assert hasattr(output, 'mean')
        assert hasattr(output, 'covariance_matrix')
        assert output.mean.shape == (batch_size, num_tasks)
    
    def test_svgp_lmc_pca_initialization(self):
        """Test PCA initialization of mixing matrix"""
        input_dim = 3 
        num_tasks = 12 
        rank = 5      
        n_samples = 30 
        
        # Create synthetic training data
        train_Y = torch.randn(n_samples, num_tasks)
        
        model = SVGP_LMC(
            input_dim=input_dim,
            num_tasks=num_tasks,
            rank=rank,
            num_inducing=16,  
            use_pca_init=True
        )
        
        # Initialize mixing matrix with PCA
        model._init_mixing_matrix_from_pca(train_Y)
        
        # Check that mixing matrix was set
        assert model.variational_strategy.lmc_coefficients.shape == (rank, num_tasks)
        assert not torch.isnan(model.variational_strategy.lmc_coefficients).any()


class TestMultiTask_GP:
    """Test the MultiTask_GP model with high-dimensional output data"""
    
    def test_multitask_gp(self):
        """Test MultiTask_GP training and prediction"""
        X_train, y_train, X_test, y_test = generate_synthetic_high_dim_output_data(
            n_samples=50, input_dim=3, output_dim=15  
        )
        
        # Initialize model
        multitask_gp = MultiTask_GP(
            base_inducing=20,   
            min_rank=4,       
            use_pca_init=True, 
            device="cpu"      
        )
        
        # Train model
        training_time = multitask_gp.train(
            X_train, y_train,
            epochs=80,  
            lr=0.05,
            enable_scheduler=True
        )
        
        assert training_time > 0
        assert hasattr(multitask_gp, 'model')
        assert hasattr(multitask_gp, 'likelihood')
        assert multitask_gp.rank is not None
        assert multitask_gp.num_inducing is not None
        assert multitask_gp.rank >= 4
        
        # Make predictions
        pred_mean, pred_std, lower, upper, infer_time = multitask_gp.predict(X_test)
        
        assert infer_time > 0
        assert pred_mean.shape == (X_test.shape[0], y_train.shape[1])
        assert pred_std.shape == (X_test.shape[0], y_train.shape[1])
        assert lower.shape == (X_test.shape[0], y_train.shape[1])
        assert upper.shape == (X_test.shape[0], y_train.shape[1])
        
        # Check that predictions are finite
        assert not np.isnan(pred_mean).any()
        assert not np.isnan(pred_std).any()
        assert not np.isinf(pred_mean).any()
        assert not np.isinf(pred_std).any()
        
        # Check that confidence intervals are ordered correctly
        assert np.all(lower <= pred_mean)
        assert np.all(pred_mean <= upper)
        assert np.all(pred_std > 0)


class TestAdaptiveParameterSelection:
    """Test adaptive parameter selection in MultiTask_GP"""
    
    def test_parameter_determination(self):
        """Test automatic parameter determination"""
        X_train, y_train, X_test, y_test = generate_synthetic_high_dim_output_data(
            n_samples=100, input_dim=6, output_dim=40
        )
        
        multitask_gp = MultiTask_GP(base_inducing=48, min_rank=8)
        
        # Convert to torch tensors
        multitask_gp.train_X = torch.from_numpy(X_train).to(torch.float32)
        multitask_gp.train_Y = torch.from_numpy(y_train).to(torch.float32)
        
        # Normalize data first
        multitask_gp._normalize_data(multitask_gp.train_X, multitask_gp.train_Y)
        
        # Test parameter determination
        multitask_gp._determine_model_parameters(
            multitask_gp.train_Y, input_dim=6, output_dim=40
        )
        
        assert multitask_gp.rank is not None
        assert multitask_gp.num_inducing is not None
        assert multitask_gp.rank >= 8  # min_rank
        assert multitask_gp.rank <= 40  # Should not exceed output_dim
        assert multitask_gp.num_inducing >= 48  # Should be at least base_inducing
        assert multitask_gp.num_inducing <= 512  # Reasonable upper bound
    
    def test_inducing_point_initialization(self):
        """Test inducing point initialization"""
        X_train, y_train, X_test, y_test = generate_synthetic_high_dim_output_data(
            n_samples=70, input_dim=5, output_dim=25
        )
        
        multitask_gp = MultiTask_GP(base_inducing=32, min_rank=6)
        
        # Setup tensors and normalize
        multitask_gp.train_X = torch.from_numpy(X_train).to(torch.float32)
        multitask_gp.train_Y = torch.from_numpy(y_train).to(torch.float32)
        multitask_gp._normalize_data(multitask_gp.train_X, multitask_gp.train_Y)
        
        # Determine parameters
        multitask_gp._determine_model_parameters(
            multitask_gp.train_Y, input_dim=5, output_dim=25
        )
        
        # Initialize inducing points
        inducing_points = multitask_gp._initialize_inducing_points()
        
        expected_shape = (multitask_gp.rank, multitask_gp.num_inducing, 5)
        assert inducing_points.shape == expected_shape
        assert not torch.isnan(inducing_points).any()
        assert not torch.isinf(inducing_points).any()


class TestTrainingComponents:
    """Test individual training components"""
    
    def test_beta_scheduler(self):
        """Test beta scheduling for variational training"""
        multitask_gp = MultiTask_GP()
        
        # Test beta progression
        epoch_0 = multitask_gp._get_beta(0)
        epoch_25 = multitask_gp._get_beta(25)
        epoch_100 = multitask_gp._get_beta(100)
        epoch_300 = multitask_gp._get_beta(300)
        
        assert epoch_0 == 0.001  # Very small initial beta
        assert epoch_25 < epoch_100  # Should increase
        assert epoch_100 < 1.0  # Should still be ramping
        assert epoch_300 == 1.0  # Should reach 1.0
    
    def test_lr_scheduler(self):
        """Test learning rate scheduling"""
        multitask_gp = MultiTask_GP()
        total_epochs = 200
        
        # Test learning rate progression
        lr_0 = multitask_gp._get_lr(0, total_epochs)
        lr_10 = multitask_gp._get_lr(10, total_epochs)
        lr_20 = multitask_gp._get_lr(20, total_epochs)
        lr_100 = multitask_gp._get_lr(100, total_epochs)
        lr_200 = multitask_gp._get_lr(200, total_epochs)
        
        assert 0 <= lr_0 <= 1
        assert lr_10 > lr_0  # Warmup phase
        assert lr_20 == 1.0  # After warmup
        assert lr_100 < 1.0  # Cosine annealing
        assert lr_200 < lr_100  # Continuing to decrease
    
    def test_initialization_methods(self):
        """Test various initialization methods"""
        X_train, y_train, _, _ = generate_synthetic_high_dim_output_data(
            n_samples=60, input_dim=4, output_dim=20
        )
        
        multitask_gp = MultiTask_GP(base_inducing=32, min_rank=5)
        
        # Setup and train partially to get model
        multitask_gp.train_X = torch.from_numpy(X_train).to(torch.float32)
        multitask_gp.train_Y = torch.from_numpy(y_train).to(torch.float32)
        multitask_gp._normalize_data(multitask_gp.train_X, multitask_gp.train_Y)
        
        multitask_gp._determine_model_parameters(
            multitask_gp.train_Y, input_dim=4, output_dim=20
        )
        
        inducing_points = multitask_gp._initialize_inducing_points()
        multitask_gp._setup_model(4, 20, inducing_points)
        
        # Test kernel parameter initialization
        multitask_gp._initialize_kernel_parameters(4)
        lengthscales = multitask_gp.model.covar_module.base_kernel.lengthscale
        assert not torch.isnan(lengthscales).any()
        assert torch.all(lengthscales > 0)
        
        # Test outputscale initialization
        multitask_gp._initialize_outputscale()
        outputscale = multitask_gp.model.covar_module.outputscale
        assert not torch.isnan(outputscale).any()
        assert torch.all(outputscale > 0)
        
        # Test mean initialization
        multitask_gp._initialize_mean()
        mean_constant = multitask_gp.model.mean_module.constant
        assert not torch.isnan(mean_constant).any()
        
        # Test likelihood initialization
        multitask_gp._initialize_likelihood(20)
        assert hasattr(multitask_gp, 'likelihood')
        assert multitask_gp.likelihood.num_tasks == 20

class TestErrorHandling:
    """Test error handling"""
    
    def test_invalid_parameters(self):
        """Test handling of invalid parameters"""
        # Test that model parameters must be determined before initialization
        multitask_gp = MultiTask_GP()
        
        with pytest.raises(RuntimeError, match="Model parameters .* must be determined"):
            multitask_gp._validate_parameters()
    
    def test_small_dataset_handling(self):
        """Test handling of very small datasets"""
        X_train, y_train, X_test, y_test = generate_synthetic_high_dim_output_data(
            n_samples=15, input_dim=3, output_dim=10  # Very small dataset
        )
        
        multitask_gp = MultiTask_GP(base_inducing=20, min_rank=3)  # More inducing than samples
        
        # Should handle gracefully
        training_time = multitask_gp.train(
            X_train, y_train,
            epochs=50,
            lr=0.1
        )
        
        assert training_time > 0
        
        # Should still make predictions
        pred_mean, pred_std, _, _, infer_time = multitask_gp.predict(X_test)
        assert pred_mean.shape == (X_test.shape[0], 10)
    
    def test_empty_data(self):
        """Test handling of empty data"""
        multitask_gp = MultiTask_GP()
        
        with pytest.raises((ValueError, RuntimeError)):
            multitask_gp.train(np.array([]).reshape(0, 1), np.array([]).reshape(0, 1))
    
    def test_mismatched_dimensions(self):
        """Test handling of mismatched input/output dimensions"""
        X_train = np.random.randn(50, 4)
        y_train = np.random.randn(40, 15)  # Wrong number of samples
        
        multitask_gp = MultiTask_GP()
        
        with pytest.raises((ValueError, RuntimeError)):
            multitask_gp.train(X_train, y_train)

if __name__ == "__main__":
    pytest.main([__file__])
