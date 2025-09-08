import pytest
import numpy as np
from pca_psimpy.src.psimpy.emulator.pca_robustgasp import PCAPPGaSP, PCAScalarGaSP
from pca_psimpy.src.psimpy.emulator.pca import InputDimReducer, OutputDimReducer, LinearPCA


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
    X_train = np.random.uniform(-2, 2, (n_samples, input_dim))
    X_test = np.random.uniform(-2, 2, (n_samples // 2, input_dim))
    
    # Create a complex multi-output function with latent structure
    # The outputs should have correlation structure that PCA can exploit
    latent_dim = min(10, output_dim // 3)  # True latent dimensionality
    
    def complex_multioutput_function(X):
        # Generate latent functions
        latent_outputs = np.zeros((X.shape[0], latent_dim))
        
        for i in range(latent_dim):
            if i == 0:
                latent_outputs[:, i] = np.sin(2 * np.pi * X[:, 0]) + 0.5 * X[:, 1] ** 2
            elif i == 1 and input_dim > 1:
                latent_outputs[:, i] = np.cos(np.pi * X[:, 1]) + 0.3 * X[:, 0]
            elif i == 2 and input_dim > 2:
                latent_outputs[:, i] = np.exp(-0.5 * (X[:, 2] ** 2)) + 0.2 * X[:, 0] * X[:, 1]
            elif i == 3 and input_dim > 3:
                latent_outputs[:, i] = np.tanh(X[:, 3]) + 0.1 * np.sum(X[:, :2], axis=1)
            else:
                # Additional latent functions
                latent_outputs[:, i] = 0.5 * np.sin(i * X[:, 0]) + 0.3 * np.cos(i * X[:, min(1, input_dim-1)])
        
        # Create loading matrix to map latent to observed outputs
        np.random.seed(random_seed + 1)  # Different seed for loading matrix
        loading_matrix = np.random.randn(output_dim, latent_dim)
        
        # Normalize loadings to control output variance
        loading_matrix = loading_matrix / np.sqrt(latent_dim)
        
        # Map latent outputs to observed outputs
        y = latent_outputs @ loading_matrix.T
        
        # Add some independent noise to each output
        y += 0.1 * np.random.randn(X.shape[0], output_dim)
        
        return y
    
    # Generate outputs
    y_train = complex_multioutput_function(X_train) + noise_level * np.random.randn(n_samples, output_dim)
    y_test = complex_multioutput_function(X_test) + noise_level * np.random.randn(n_samples // 2, output_dim)
    
    return X_train, y_train, X_test, y_test


def create_dim_reducers(input_dim, output_dim, input_n_components=None, output_n_components=None):
    """Create input and output dimension reducers for testing"""
    
    # Input dimension reducer (if input_dim is high enough)
    input_dim_reducer = None
    if input_n_components is not None and input_n_components < input_dim:
        input_pca = LinearPCA(n_components=input_n_components)
        input_dim_reducer = InputDimReducer(input_pca)
    
    # Output dimension reducer
    output_dim_reducer = None
    if output_n_components is not None and output_n_components < output_dim:
        output_pca = LinearPCA(n_components=output_n_components)
        output_dim_reducer = OutputDimReducer(output_pca)
    
    return input_dim_reducer, output_dim_reducer


class TestPCAPPGaSP:
    """Test the PCAPPGaSP model with high-dimensional output data"""
    
    def test_pcappgasp(self):
        """Test PCAPPGaSP training and prediction"""
        X_train, y_train, X_test, y_test = generate_synthetic_high_dim_output_data(
            n_samples=40, input_dim=4, output_dim=15  
        )
        
        # Create dimension reducers
        input_dim_reducer, output_dim_reducer = create_dim_reducers(
            input_dim=4, output_dim=15, input_n_components=None, output_n_components=8
        )
        
        # Determine ndim (input dimension after reduction)
        ndim = 4 
        
        # Initialize model
        pcappgasp = PCAPPGaSP(
            ndim=ndim,
            input_dim_reducer=input_dim_reducer,
            output_dim_reducer=output_dim_reducer,
            nugget_est=True,
            method='post_mode'
        )
        
        # Train model
        training_time = pcappgasp.train(X_train, y_train)
        
        assert training_time > 0
        assert hasattr(pcappgasp, 'emulator')
        assert pcappgasp.original_input_dim == 4  # input_dim
        assert pcappgasp.original_output_dim == 15  # output_dim
        
        # Make predictions
        predictions_latent, predictions_mean_orig, infer_time, _ = pcappgasp.predict(X_test)
        
        assert infer_time > 0
        
        # Check latent predictions shape
        expected_latent_dim = 8  # output_n_components
        assert predictions_latent.shape == (X_test.shape[0], expected_latent_dim, 4)
        
        # Check original space predictions shape
        assert predictions_mean_orig.shape == (X_test.shape[0], 15)  # output_dim
        
        # Check that predictions are finite
        assert not np.isnan(predictions_latent).any()
        assert not np.isnan(predictions_mean_orig).any()
        assert not np.isinf(predictions_latent).any()
        assert not np.isinf(predictions_mean_orig).any()
    
    def test_pcappgasp_sample(self):
        """Test PCAPPGaSP sampling functionality"""
        X_train, y_train, X_test, y_test = generate_synthetic_high_dim_output_data(
            n_samples=30, input_dim=3, output_dim=12 
        )
        
        # Create dimension reducers
        input_dim_reducer, output_dim_reducer = create_dim_reducers(
            input_dim=3, output_dim=12, input_n_components=None, output_n_components=6
        )
        
        ndim = 3  
        
        pcappgasp = PCAPPGaSP(
            ndim=ndim,
            input_dim_reducer=input_dim_reducer,
            output_dim_reducer=output_dim_reducer
        )
        
        pcappgasp.train(X_train, y_train)
        
        # Test sampling
        nsamples = 5
        samples = pcappgasp.sample(X_test[:10], nsamples=nsamples)
        
        if output_dim_reducer is not None:
            assert samples.shape == (10, 12, nsamples)  # output_dim
        else:
            expected_output_dim = 6  # output_n_components
            assert samples.shape == (10, expected_output_dim, nsamples)
        
        assert not np.isnan(samples).any()
        assert not np.isinf(samples).any()


class TestPCAScalarGaSP:
    """Test the PCAScalarGaSP model with high-dimensional input data"""
    
    def test_pcascalargasp(self):
        """Test PCAScalarGaSP training and prediction"""
        # Generate scalar output data
        X_train, y_train_multi, X_test, _ = generate_synthetic_high_dim_output_data(
            n_samples=40, input_dim=8, output_dim=1  
        )
        y_train = y_train_multi.ravel()  
        
        input_dim_reducer = None
        input_n_components = 5  
        if input_n_components is not None:
            input_pca = LinearPCA(n_components=input_n_components)
            input_dim_reducer = InputDimReducer(input_pca)
        
        ndim = input_n_components 
        
        # Initialize model
        pcascalargasp = PCAScalarGaSP(
            ndim=ndim,
            input_dim_reducer=input_dim_reducer,
            nugget_est=True,
            method='post_mode'
        )
        
        # Train model
        training_time = pcascalargasp.train(X_train, y_train)
        
        assert training_time > 0
        assert hasattr(pcascalargasp, 'emulator')
        
        # Make predictions
        predictions, infer_time = pcascalargasp.predict(X_test)
        
        assert infer_time > 0
        assert predictions.shape == (X_test.shape[0], 4)  # mean, lower, upper, sd
        
        # Check that predictions are finite
        assert not np.isnan(predictions).any()
        assert not np.isinf(predictions).any()
        
        # Check that confidence intervals are ordered correctly
        mean = predictions[:, 0]
        lower = predictions[:, 1]
        upper = predictions[:, 2]
        sd = predictions[:, 3]
        
        assert np.all(lower <= mean)
        assert np.all(mean <= upper)
        assert np.all(sd > 0)
    
    def test_pcascalargasp_sample(self):
        """Test PCAScalarGaSP sampling functionality"""
        X_train, y_train_multi, X_test, y_test_multi = generate_synthetic_high_dim_output_data(
            n_samples=30, input_dim=6, output_dim=1  
        )
        y_train = y_train_multi.ravel()
        
        input_dim_reducer = None
        input_n_components = 4  
        if input_n_components is not None:
            input_pca = LinearPCA(n_components=input_n_components)
            input_dim_reducer = InputDimReducer(input_pca)
        
        ndim = input_n_components  
        
        pcascalargasp = PCAScalarGaSP(
            ndim=ndim,
            input_dim_reducer=input_dim_reducer
        )
        
        pcascalargasp.train(X_train, y_train)
        
        # Test sampling
        nsamples = 8
        samples = pcascalargasp.sample(X_test[:10], nsamples=nsamples)
        
        assert samples.shape == (10, nsamples)
        assert not np.isnan(samples).any()
        assert not np.isinf(samples).any()

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_dimension_mismatch(self):
        """Test handling of dimension mismatches"""
        X_train, y_train, X_test, y_test = generate_synthetic_high_dim_output_data(
            n_samples=50, input_dim=5, output_dim=20
        )
        
        # Create PCAPPGaSP with wrong ndim
        pcappgasp = PCAPPGaSP(ndim=3)  # Wrong: should be 5
        
        with pytest.raises(AssertionError):
            pcappgasp.train(X_train, y_train)
    
    def test_invalid_test_input_dimension(self):
        """Test handling of invalid test input dimensions"""
        X_train, y_train, X_test, y_test = generate_synthetic_high_dim_output_data(
            n_samples=50, input_dim=5, output_dim=15
        )
        
        pcappgasp = PCAPPGaSP(ndim=5)
        pcappgasp.train(X_train, y_train)
        
        # Wrong test input dimension
        X_test_wrong = np.random.randn(10, 3)  # Should be 5
        
        with pytest.raises(ValueError, match="Expected input dimension"):
            pcappgasp.predict(X_test_wrong)
    
    def test_zero_mean_with_output_pca_error(self):
        """Test that zero_mean='Yes' with output PCA raises error"""
        output_pca = LinearPCA(n_components=5)
        output_dim_reducer = OutputDimReducer(output_pca)
        
        with pytest.raises(ValueError, match="Cannot use zero_mean='Yes' when PCA is used for output"):
            PCAPPGaSP(
                ndim=3,
                output_dim_reducer=output_dim_reducer,
                zero_mean="Yes"
            )
    
    def test_pcascalargasp_multioutput_error(self):
        """Test that PCAScalarGaSP raises error for multi-output data"""
        X_train, y_train, _, _ = generate_synthetic_high_dim_output_data(
            n_samples=50, input_dim=5, output_dim=3
        )
        
        pcascalargasp = PCAScalarGaSP(ndim=5)
        
        with pytest.raises(ValueError, match="PCAScalarGaSP only works for scalar-output model"):
            pcascalargasp.train(X_train, y_train)

if __name__ == "__main__":
    pytest.main([__file__])
