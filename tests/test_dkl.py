import pytest
import numpy as np
import torch
import gpytorch
from gpytorch_emulator.dkl import ExactGP, DKL_GP, FeatureExtractor, ExactGPModel, DKL_GPRegressor


def generate_synthetic_high_dim_input_data(n_samples=100, input_dim=50, output_dim=1, noise_level=0.1, random_seed=42):
    """
    Generate synthetic data for high-dimensional input problems.
    
    Parameters:
    -----------
    n_samples : int
        Number of training samples
    input_dim : int  
        High input dimensionality
    output_dim : int
        Output dimensionality (usually 1 for scalar output)
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
    
    # Generate high-dimensional input data
    X_train = np.random.randn(n_samples, input_dim)
    X_test = np.random.randn(n_samples // 2, input_dim)
    
    # Create a complex non-linear function with dimension reduction effect
    # Use only first few dimensions as "active" dimensions
    active_dims = min(5, input_dim // 5)
    
    def complex_function(X):
        # Non-linear function that depends mainly on first few dimensions
        active_X = X[:, :active_dims]
        y = (np.sin(2 * np.pi * active_X[:, 0]) + 
             np.cos(np.pi * active_X[:, 1]) if active_dims > 1 else 0 +
             0.5 * active_X[:, 2] ** 2 if active_dims > 2 else 0 +
             np.exp(-0.5 * active_X[:, 3]) if active_dims > 3 else 0 +
             0.2 * active_X[:, 4] if active_dims > 4 else 0)
        
        # Add some interaction with other dimensions (but weaker)
        if input_dim > active_dims:
            remaining_dims = X[:, active_dims:]
            y += 0.1 * np.sum(remaining_dims ** 2, axis=1) / remaining_dims.shape[1]
        
        return y.reshape(-1, output_dim)
    
    # Generate outputs
    y_train = complex_function(X_train) + noise_level * np.random.randn(n_samples, output_dim)
    y_test = complex_function(X_test) + noise_level * np.random.randn(n_samples // 2, output_dim)
    
    return X_train, y_train, X_test, y_test


class TestExactGP:
    """Test the ExactGP model with high-dimensional input data"""
    
    def test_exact_gp_train_predict(self):
        """Test ExactGP training and prediction"""
        X_train, y_train, X_test, y_test = generate_synthetic_high_dim_input_data(
            n_samples=30, input_dim=10, output_dim=1  
        )
        
        # Initialize model
        exact_gp = ExactGP(device="cpu", kernel_type="matern_5_2")
        
        # Train model
        training_time = exact_gp.train(
            X_train, y_train.ravel(), 
            num_epochs=50,  
            lr=0.1, 
            optim="adam",
            enable_scheduler=False
        )
        
        assert training_time > 0
        assert hasattr(exact_gp, 'model')
        assert hasattr(exact_gp, 'likelihood')
        
        # Make predictions
        mean, std, lower, upper, infer_time = exact_gp.predict(X_test)
        
        assert infer_time > 0
        assert mean.shape == (X_test.shape[0],)
        assert std.shape == (X_test.shape[0],)
        assert lower.shape == (X_test.shape[0],)
        assert upper.shape == (X_test.shape[0],)
        
        # Check that confidence intervals are ordered correctly
        assert np.all(lower <= mean)
        assert np.all(mean <= upper)
        assert np.all(std > 0)


class TestDKL_GP:
    """Test the DKL_GP model with high-dimensional input data"""
    
    def test_dkl_gp_train_predict(self):
        """Test DKL_GP training and prediction"""
        X_train, y_train, X_test, y_test = generate_synthetic_high_dim_input_data(
            n_samples=40, input_dim=15, output_dim=1  
        )
        
        # Initialize model
        dkl_gp = DKL_GP(reduced_dim=3, device="cpu", kernel_type="matern_5_2")
        
        # Train model
        training_time = dkl_gp.train(
            X_train, y_train.ravel(),
            num_epochs=80,  
            lr=0.01,
            optim="adam",
            enable_scheduler=False
        )
        
        assert training_time > 0
        assert hasattr(dkl_gp, 'model')
        assert hasattr(dkl_gp, 'likelihood')
        assert isinstance(dkl_gp.model, DKL_GPRegressor)
        
        # Make predictions
        mean, std, lower, upper, infer_time = dkl_gp.predict(X_test)
        
        assert infer_time > 0
        assert mean.shape == (X_test.shape[0],)
        assert std.shape == (X_test.shape[0],)
        assert lower.shape == (X_test.shape[0],)
        assert upper.shape == (X_test.shape[0],)
        
        # Check that confidence intervals are ordered correctly
        assert np.all(lower <= mean)
        assert np.all(mean <= upper)
        assert np.all(std > 0)


class TestFeatureExtractor:
    """Test the FeatureExtractor neural network"""
    
    def test_feature_extractor_forward(self):
        """Test FeatureExtractor forward pass"""
        data_dim = 25
        latent_dim = 3
        batch_size = 10
        
        feature_extractor = FeatureExtractor(data_dim=data_dim, latent_dim=latent_dim)
        
        # Test forward pass
        x = torch.randn(batch_size, data_dim)
        output = feature_extractor(x)
        
        assert output.shape == (batch_size, latent_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestExactGPModel:
    """Test the ExactGPModel standalone"""
    
    def test_exact_gp_model_forward(self):
        """Test ExactGPModel forward pass"""
        n_samples = 15 
        input_dim = 8  
        
        X = torch.randn(n_samples, input_dim)
        y = torch.randn(n_samples)
        
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(X, y, likelihood, kernel_type="matern_5_2") 
        
        # Test forward pass
        output = model(X)
        
        assert hasattr(output, 'mean')
        assert hasattr(output, 'covariance_matrix')
        assert output.mean.shape == (n_samples,)


class TestDKL_GPRegressor:
    """Test the DKL_GPRegressor standalone"""
    
    def test_dkl_gp_regressor_forward(self):
        """Test DKL_GPRegressor forward pass"""
        n_samples = 15 
        input_dim = 10 
        reduced_dim = 3 
        
        X = torch.randn(n_samples, input_dim)
        y = torch.randn(n_samples)
        
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = DKL_GPRegressor(X, y, likelihood, reduced_dim=reduced_dim, kernel_type="matern_5_2")
        
        # Test forward pass
        output = model(X)
        
        assert hasattr(output, 'mean')
        assert hasattr(output, 'covariance_matrix')
        assert output.mean.shape == (n_samples,)
        
        # Test that feature extractor reduces dimensionality correctly
        projected_x = model.feature_extractor(X)
        assert projected_x.shape == (n_samples, reduced_dim)

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_kernel_type(self):
        """Test handling of invalid kernel types"""
        X_train, y_train, _, _ = generate_synthetic_high_dim_input_data(n_samples=20, input_dim=10)
        
        exact_gp = ExactGP(device="cpu", kernel_type="invalid_kernel")
        with pytest.raises(ValueError, match="Unsupported kernel_type"):
            exact_gp.train(X_train, y_train.ravel())
    
    def test_invalid_optimizer(self):
        """Test handling of invalid optimizers"""
        X_train, y_train, _, _ = generate_synthetic_high_dim_input_data(n_samples=20, input_dim=10)
        
        exact_gp = ExactGP(device="cpu")
        with pytest.raises(ValueError, match="Optimizer are only supported"):
            exact_gp.train(X_train, y_train.ravel(), optim="invalid_optimizer")
    
    def test_empty_data(self):
        """Test handling of empty data"""
        exact_gp = ExactGP(device="cpu")
        
        with pytest.raises((ValueError, RuntimeError)):
            exact_gp.train(np.array([]), np.array([]))


if __name__ == "__main__":
    pytest.main([__file__])
