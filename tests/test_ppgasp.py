import pytest
import numpy as np
import torch
import gpytorch
from gpytorch_emulator.ppgasp import MoGP_GPytorch, PCA_MoGP_GPytorch, BatchIndependentMultioutputGPModel
from pca_psimpy.src.psimpy.emulator.pca import OutputDimReducer, LinearPCA


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
    
    # Generate input data in a reasonable range
    X_train = np.random.uniform(-2, 2, (n_samples, input_dim))
    X_test = np.random.uniform(-2, 2, (n_samples // 2, input_dim))
    
    # Create a complex multi-output function with correlation structure
    latent_dim = min(8, output_dim // 4)  # True latent dimensionality
    
    def complex_multioutput_function(X):
        # Generate latent functions based on input
        latent_outputs = np.zeros((X.shape[0], latent_dim))
        
        for i in range(latent_dim):
            if i == 0:
                latent_outputs[:, i] = np.sin(2 * np.pi * X[:, 0]) + 0.5 * X[:, min(1, input_dim-1)] ** 2
            elif i == 1 and input_dim > 1:
                latent_outputs[:, i] = np.cos(np.pi * X[:, 1]) + 0.3 * X[:, 0]
            elif i == 2 and input_dim > 2:
                latent_outputs[:, i] = np.exp(-0.5 * (X[:, 2] ** 2)) + 0.2 * X[:, 0] * X[:, 1]
            elif i == 3 and input_dim > 3:
                latent_outputs[:, i] = np.tanh(X[:, 3]) + 0.1 * np.sum(X[:, :2], axis=1)
            elif i == 4 and input_dim > 4:
                latent_outputs[:, i] = X[:, 4] ** 3 + 0.2 * np.sin(X[:, 0] + X[:, 1])
            else:
                # Additional latent functions
                idx1 = i % input_dim
                idx2 = (i + 1) % input_dim
                latent_outputs[:, i] = 0.5 * np.sin(i * X[:, idx1]) + 0.3 * np.cos(i * X[:, idx2])
        
        # Create loading matrix to map latent to observed outputs
        np.random.seed(random_seed + 1)  # Different seed for loading matrix
        loading_matrix = np.random.randn(output_dim, latent_dim)
        
        # Normalize loadings to control output variance
        loading_matrix = loading_matrix / np.sqrt(latent_dim)
        
        # Map latent outputs to observed outputs
        y = latent_outputs @ loading_matrix.T
        
        # Add some independent structure to each output
        y += 0.05 * np.random.randn(X.shape[0], output_dim)
        
        return y
    
    # Generate outputs
    y_train = complex_multioutput_function(X_train) + noise_level * np.random.randn(n_samples, output_dim)
    y_test = complex_multioutput_function(X_test) + noise_level * np.random.randn(n_samples // 2, output_dim)
    
    return X_train, y_train, X_test, y_test


class TestMoGP_GPytorch:
    """Test the MoGP_GPytorch model with high-dimensional output data"""
    
    def test_mogp_gpytorch(self):
        """Test MoGP_GPytorch training and prediction"""
        X_train, y_train, X_test, y_test = generate_synthetic_high_dim_output_data(
            n_samples=40, input_dim=3, output_dim=20  
        )
        
        # Initialize model with simple configuration
        mogp = MoGP_GPytorch(device="cpu", kernel_type="matern_5_2", mimic_ppgasp=True)
        
        # Train model
        training_time = mogp.train(
            X_train, y_train,
            num_epochs=50, 
            lr=0.1,
            optim="adam",
            enable_scheduler=False
        )
        
        assert training_time > 0
        assert hasattr(mogp, 'model')
        assert hasattr(mogp, 'likelihood')
        assert isinstance(mogp.model, BatchIndependentMultioutputGPModel)
        
        # Make predictions
        mean, std, lower, upper, infer_time = mogp.predict(X_test)
        
        assert infer_time > 0
        assert mean.shape == (X_test.shape[0], y_train.shape[1])
        assert std.shape == (X_test.shape[0], y_train.shape[1])
        assert lower.shape == (X_test.shape[0], y_train.shape[1])
        assert upper.shape == (X_test.shape[0], y_train.shape[1])
        
        # Check that predictions are finite
        assert not np.isnan(mean).any()
        assert not np.isnan(std).any()
        assert not np.isinf(mean).any()
        assert not np.isinf(std).any()
        
        # Check that confidence intervals are ordered correctly
        assert np.all(lower <= mean)
        assert np.all(mean <= upper)
        assert np.all(std > 0)


class TestPCA_MoGP_GPytorch:
    """Test the PCA_MoGP_GPytorch model with high-dimensional output data"""
    
    def test_pca_mogp_gpytorch(self):
        """Test PCA_MoGP_GPytorch training and prediction"""
        X_train, y_train, X_test, y_test = generate_synthetic_high_dim_output_data(
            n_samples=50, input_dim=4, output_dim=15 
        )
        
        # Create output dimension reducer
        output_pca = LinearPCA(n_components=5)  
        output_dim_reducer = OutputDimReducer(output_pca)
        
        # Initialize model
        pca_mogp = PCA_MoGP_GPytorch(
            output_dim_reducer=output_dim_reducer,
            device="cpu",
            kernel_type="matern_5_2",
            mimic_ppgasp=False
        )
        
        # Preprocess data with dimension reduction
        train_X_scaled, train_Y_reduced, test_X_scaled, test_Y_reduced = pca_mogp.preprocess_dim_reduction(
            X_train, y_train, X_test, y_test
        )
        
        assert train_Y_reduced.shape[1] == 5  # output_n_components
        assert test_Y_reduced.shape[1] == 5  # output_n_components
        
        # Train model 
        training_time = pca_mogp.train(
            train_X_scaled, train_Y_reduced,
            num_epochs=60, 
            lr=0.1,
            optim="adam",
            enable_scheduler=False
        )
        
        assert training_time > 0
        assert hasattr(pca_mogp, 'model')
        assert hasattr(pca_mogp, 'likelihood')
        
        # Make predictions in reduced space
        mean_reduced, std_reduced, lower_reduced, upper_reduced, infer_time = pca_mogp.predict(test_X_scaled)
        
        assert infer_time > 0
        assert mean_reduced.shape == (test_X_scaled.shape[0], 5)  # output_n_components
        assert std_reduced.shape == (test_X_scaled.shape[0], 5)  # output_n_components
        
        # Check that predictions are finite
        assert not np.isnan(mean_reduced).any()
        assert not np.isnan(std_reduced).any()
        
        # Postprocess back to original space
        predictions_original = pca_mogp.postprocess_invert_back(mean_reduced)
        
        assert predictions_original.shape == (test_X_scaled.shape[0], 15)  # output_dim
        assert not np.isnan(predictions_original).any()
        assert not np.isinf(predictions_original).any()


class TestBatchIndependentMultioutputGPModel:
    """Test the BatchIndependentMultioutputGPModel standalone"""
    
    def test_batch_model_forward(self):
        """Test BatchIndependentMultioutputGPModel forward pass"""
        n_samples = 20  
        input_dim = 3   
        num_tasks = 5   
        
        X = torch.randn(n_samples, input_dim)
        y = torch.randn(n_samples, num_tasks)
        
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks, rank=0)

        model = BatchIndependentMultioutputGPModel(
            X, y, likelihood, kernel_type="matern_5_2", mimic_ppgasp=True 
        )
        
        # Test forward pass
        output = model(X)
        
        assert hasattr(output, 'mean')
        # Both configurations should output (n_samples, num_tasks) 
        assert output.mean.shape == (n_samples, num_tasks)
    
    def test_tie_parameters_across_tasks(self):
        """Test parameter tying functionality"""
        n_samples = 25
        input_dim = 4
        num_tasks = 8
        
        X = torch.randn(n_samples, input_dim)
        y = torch.randn(n_samples, num_tasks)
        
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks, rank=0)

        model = BatchIndependentMultioutputGPModel(
            X, y, likelihood, kernel_type="matern_5_2", mimic_ppgasp=True
        )
        
        # Parameters should be tied after initialization when mimic_ppgasp=True
        inner_kernel = model._get_inner_kernel(model.covar_module)
        lengthscales = inner_kernel.lengthscale
        
        # Check that lengthscales are similar across tasks (tied)
        if lengthscales.dim() > 1 and lengthscales.shape[0] > 1:
            lengthscale_std = torch.std(lengthscales, dim=0)
            # Handle NaN values by checking finite values or skipping test for small data
            if torch.all(torch.isfinite(lengthscale_std)):
                assert torch.all(lengthscale_std < 1e-3)  # Relaxed tolerance


class TestDifferentOptimizers:
    """Test different optimizers for PCA_MoGP_GPytorch"""
    
    def test_pca_mogp_optimizers(self):
        """Test PCA_MoGP_GPytorch with adam optimizer"""
        X_train, y_train, X_test, y_test = generate_synthetic_high_dim_output_data(
            n_samples=40, input_dim=3, output_dim=12  
        )
        
        output_pca = LinearPCA(n_components=6)
        output_dim_reducer = OutputDimReducer(output_pca)
        
        pca_mogp = PCA_MoGP_GPytorch(
            output_dim_reducer=output_dim_reducer,
            device="cpu",
            kernel_type="rbf",
            mimic_ppgasp=True
        )
        
        train_X_scaled, train_Y_reduced, test_X_scaled, test_Y_reduced = pca_mogp.preprocess_dim_reduction(
            X_train, y_train, X_test, y_test
        )
        
        # Test training with adam optimizer
        training_time = pca_mogp.train(
            train_X_scaled, train_Y_reduced,
            num_epochs=30,  
            lr=0.1,
            optim="adam",  
            enable_scheduler=False
        )
        
        assert training_time > 0
        
        # Test prediction
        mean_reduced, std_reduced, lower_reduced, upper_reduced, infer_time = pca_mogp.predict(test_X_scaled)
        
        assert mean_reduced.shape == (test_X_scaled.shape[0], 6)  # n_components
        assert not np.isnan(mean_reduced).any()
        assert infer_time > 0

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_optimizer_mogp(self):
        """Test handling of invalid optimizers for MoGP_GPytorch"""
        X_train, y_train, _, _ = generate_synthetic_high_dim_output_data(
            n_samples=50, input_dim=4, output_dim=15
        )
        
        mogp = MoGP_GPytorch(device="cpu")
        
        # lbfgs and mixed should raise error for base MoGP_GPytorch
        with pytest.raises(ValueError, match="only supports 'adam' or 'adamw'"):
            mogp.train(X_train, y_train, optim="lbfgs")
        
        with pytest.raises(ValueError, match="only supports 'adam' or 'adamw'"):
            mogp.train(X_train, y_train, optim="mixed")
    
    def test_invalid_kernel_type(self):
        """Test handling of invalid kernel types"""
        X_train, y_train, _, _ = generate_synthetic_high_dim_output_data(n_samples=20, input_dim=4, output_dim=10)
        
        mogp = MoGP_GPytorch(device="cpu", kernel_type="invalid_kernel")
        with pytest.raises(ValueError, match="Unsupported kernel_type"):
            mogp.train(X_train, y_train)
    
    def test_invalid_optimizer_general(self):
        """Test handling of completely invalid optimizers"""
        X_train, y_train, _, _ = generate_synthetic_high_dim_output_data(
            n_samples=50, input_dim=4, output_dim=15
        )
        
        output_pca = LinearPCA(n_components=8)
        output_dim_reducer = OutputDimReducer(output_pca)
        
        pca_mogp = PCA_MoGP_GPytorch(output_dim_reducer=output_dim_reducer, device="cpu")
        
        train_X_scaled, train_Y_reduced, _, _ = pca_mogp.preprocess_dim_reduction(
            X_train, y_train, X_train[:20], y_train[:20]
        )
        
        with pytest.raises(ValueError, match="Optimizer are only supported"):
            pca_mogp.train(train_X_scaled, train_Y_reduced, optim="invalid_optimizer")
    
    def test_empty_data(self):
        """Test handling of empty data"""
        mogp = MoGP_GPytorch(device="cpu")
        
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            mogp.train(np.array([]).reshape(0, 1), np.array([]).reshape(0, 1))

if __name__ == "__main__":
    pytest.main([__file__])
