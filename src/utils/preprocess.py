import numpy as np
import rasterio

def zero_truncated_data(filepath, threshold, valid_cols=None):
        """ Preprocess the dataset to filter out the zeros so that GP emulators can be trained
        
        Args:
            filepath (str): File path of the dataset to be preprocessed.
            threshold (int, float): Threshold value to define valid cells from simulations.
            valid_cols (list, optional): column numbers to extract. Defaults to None.
        
        Raises:
            TypeError: threshold must be a number
            ValueError: threshold cannot be negative
        
        Returns:
            training (np.ndarray): A data frame consisting of the vector outputs from simulations
            valid_cols (np.ndarray): An array consisting of the valid column names
        """
        if not isinstance(threshold, (int, float)):
            raise TypeError('threshold must be a number')
        if threshold < 0:
            raise ValueError('threshold cannot be negative')
    
        with rasterio.open(filepath) as src:
            rows = src.height
            cols = src.width
            size = src.count
        
            unstacked = np.zeros((size, rows * cols))
        
            for sim in range(size):
                unstacked[sim, :] = src.read(sim + 1).reshape(1, rows * cols)
        
        if valid_cols is None:
            valid_cols = np.where(unstacked >= threshold, 1, 0).sum(axis=0)
        indices = np.flatnonzero(valid_cols)
        nz_out = unstacked[:, indices]
        return nz_out, valid_cols, rows, cols
    