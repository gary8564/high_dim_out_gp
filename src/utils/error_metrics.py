import numpy as np
import sys
from warnings import warn
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error, root_mean_squared_error


class ErrorMetrics:
    @staticmethod
    def MAPE(predictions: np.ndarray, observations: np.ndarray) -> float:
        """Mean Absolute Percentage Error (MAPE)

        Args:
            predictions (np.ndarray): predictions
            measurements (np.ndarray): measurements

        Raises:
            ValueError: if sizes do not match

        Returns:
            float: error [%]
        """
        if predictions.size != observations.size:
            raise ValueError("predictions and measurements must be of equal size")

        return 100.0 * mean_absolute_percentage_error(
            y_true=observations, y_pred=predictions
        )

    @staticmethod
    def RMSE(predictions: np.ndarray, observations: np.ndarray) -> float:
        """Normalized Root-Mean-Square Error (NRMSE)

        Args:
            predictions (np.ndarray): predictions
            observations (np.ndarray): observations

        Raises:
            ValueError: if sizes do not match
            UserWarning: if mean of observations is zero

        Returns:
            float: NRMSD [1]
        """
        if predictions.size != observations.size:
            raise ValueError("predictions and measurements must be of equal size")

        RMSE = root_mean_squared_error(y_true=observations, y_pred=predictions)

        return RMSE

    @staticmethod
    def NRMSE(predictions: np.ndarray, observations: np.ndarray) -> float:
        """Normalized Root-Mean-Square Error (NRMSE)

        Args:
            predictions (np.ndarray): predictions
            observations (np.ndarray): observations

        Raises:
            ValueError: if sizes do not match
            UserWarning: if mean of observations is zero

        Returns:
            float: NRMSD [1]
        """
        if predictions.size != observations.size:
            raise ValueError("predictions and measurements must be of equal size")

        RMSE = np.sqrt(mean_squared_error(y_true=observations, y_pred=predictions))

        mean = np.mean(observations)
        if abs(mean) < sys.float_info.epsilon:
            warn("Mean of observations is zero (or very close to it)")

        return RMSE / mean

    @staticmethod
    def R2(predictions: np.ndarray, observations: np.ndarray) -> float:
        """R^2 Coefficient of Determination

        Args:
            predictions (np.ndarray): predictions
            observations (np.ndarray): observations

        Raises:
            ValueError: if sizes do not match

        Returns:
            float: R^2 error [1]
        """
        if predictions.size != observations.size:
            raise ValueError("predictions and measurements must be of equal size")

        return r2_score(y_true=observations, y_pred=predictions)

    @staticmethod
    def CoverageProbability(predictions_mean: np.ndarray, predictions_lower95: np.ndarray, predictions_upper95: np.ndarray, observations: np.ndarray) -> float:
        """Coverage Probability Calculation

        Args:
            predictions_mean (np.ndarray): maen values of predictions 
            predictions_lower95 (np.ndarray): lower 95% confidence interval of predictions
            predicttions_upper95 (np.ndarray): upper 95% confidence interval of predictions
            observations (np.ndarray): observations

        Raises:
            ValueError: if sizes do not match

        Returns:
            float: coverage probability [1]
        """
        
        if predictions_mean.size != observations.size:
            raise ValueError("predictions and measurements must be of equal size")

        return np.mean((predictions_upper95 >= observations) & (observations >= predictions_lower95))