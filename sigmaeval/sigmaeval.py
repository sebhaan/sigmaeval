"""
Evaluation Metrics for Probabilistic Predictions.

This module provides specialized evaluation metrics for scoring the accuracy of uncertainty predictions 
and posterior distributions, rather than point-prediction accuracy. Unlike standard metrics from 
libraries like `sklearn.metrics`, which assess prediction correctness, these functions focus on quantifying the 
calibration and reliability of probabilistic models.

This module includes metrics like the Continuous Ranked Probability Score (CRPS) and Brier Score, which assess 
the quality of probabilistic forecasts by comparing predicted distributions to actual outcomes.

Supported Evaluation Functions:
- crps_quantile: Compute CRPS for regression predictions based on quantile estimates.
- crps_gaussian: CRPS closed-form solution for Gaussian predictive distributions.
- crps_ensemble_gaussian: Compute CRPS for ensemble predictions by fitting Gaussian distributions.
- brier_score_multiclass: Compute the Brier score for multi-class classification predictions.
- brier_score_components: Decompose the Brier score into reliability, resolution, and uncertainty.
- expected_calibration_error: Compute the Expected Calibration Error (ECE) for classification models.

These metrics are particularly useful in Bayesian machine learning, ensemble models, and 
probabilistic forecasting, where predictions are distributions rather than single values. They are 
especially relevant for **TabPFN**, a probabilistic transformer-based model for tabular classification, 
as it produces full posterior predictive distributions. By using CRPS and Brier score-based metrics, 
this module helps quantify how well these distributions reflect true uncertainties, ensuring more 
calibrated and reliable probabilistic predictions.

All functions are designed to work with NumPy arrays and support vectorized computation.

Author: Sebastian Haan
"""
import numpy as np
from typing import Union, Optional, Tuple
from numpy.typing import NDArray
from math import erf

def crps_quantile(predicted_quantiles: np.ndarray, 
                  quantile_levels: np.ndarray, 
                  true_values: np.ndarray, 
                  normalization = True) -> float:
    """
    Compute the Continuous Ranked Probability Score (CRPS) for regression predictions based on quantile estimates.
    
    The CRPS measures the accuracy of probabilistic forecasts by comparing the predicted cumulative distribution 
    function (CDF) with the empirical CDF of the observations. It is a proper scoring rule that generalizes the 
    mean absolute error (MAE) to probabilistic forecasts. Lower CRPS values indicate better forecasts, with 0 
    being a perfect forecast.

    The CRPS ranges from 0 to +Inf (not normalized), where 0 indicates a perfect forecast. 
    
    When normalized (default), the CRPS is divided by the standard deviation of the observations, making it 
    scale-independent. Normalized CRPS typically ranges from 0 to 1, where:
    - Excellent: < 0.2
    - Good: 0.2 - 0.4
    - Fair: 0.4 - 0.6
    - Poor: > 0.6
    
    Parameters:
    - predicted_quantiles: np.ndarray of shape (n_quantiles, n_samples)
        The predicted quantiles for each sample.
    - quantile_levels: np.ndarray of shape (n_quantiles,)
        The corresponding quantile levels (e.g., [0.1, 0.25, 0.5, 0.75, 0.9]).
    - true_values: np.ndarray of shape (n_samples,)
        The actual observed values for each sample.
    - normalization: bool (default: True)
    
    Returns:
    - crps: float
        The average CRPS over all samples.
    """

    # cross-check in case inputs are lists instead of numpy arrays
    if isinstance(predicted_quantiles, list):
        predicted_quantiles = np.array(predicted_quantiles)
        print("Warning: predicted_quantiles is a list. Converting to numpy array.")
    if isinstance(quantile_levels, list):
        quantile_levels = np.array(quantile_levels)

    n_quantiles, n_samples = predicted_quantiles.shape
    
    # Input validation
    if len(quantile_levels) != n_quantiles:
        raise ValueError(f"Quantile levels length ({len(quantile_levels)}) must match number of predicted quantiles ({n_quantiles})")
    if len(true_values) != n_samples:
        raise ValueError(f"True values length ({len(true_values)}) must match number of samples ({n_samples})")
    if not np.all(np.diff(quantile_levels) > 0):
        raise ValueError("Quantile levels must be strictly increasing")
    if not (0 <= quantile_levels).all() and (quantile_levels <= 1).all():
        raise ValueError("Quantile levels must be between 0 and 1")
    
    # Ensure arrays are float type for numerical stability
    predicted_quantiles = np.asarray(predicted_quantiles, dtype=np.float64)
    true_values = np.asarray(true_values, dtype=np.float64)
    
    # Sort quantiles for each sample to ensure monotonicity
    predicted_quantiles = np.sort(predicted_quantiles, axis=0)
    
    # Vectorized computation
    crps_sum = 0.0
    for i in range(n_samples):
        y_true = true_values[i]
        quantiles = predicted_quantiles[:, i]
        
        # Handle potential NaN/Inf values
        if np.any(~np.isfinite(quantiles)) or not np.isfinite(y_true):
            continue
            
        # Compute the empirical CDF step function
        indicator = (y_true <= quantiles).astype(np.float64)
        
        # Add small epsilon to avoid numerical instability in diff
        quantiles_with_zero = np.concatenate(([quantiles[0] - 1e-8], quantiles))
        
        # Compute CRPS for this sample
        crps_i = np.sum((quantile_levels - indicator) ** 2 * np.diff(quantiles_with_zero))
        crps_sum += crps_i
    
    # Handle case where all samples were invalid
    n_valid_samples = n_samples - np.isnan(crps_sum)
    if n_valid_samples == 0:
        return np.nan
    
    # Normalize by number of samples
    if normalization:
        norm = np.std(true_values)
        crps_sum = crps_sum / norm if norm > 0 else crps_sum
        
    return crps_sum / n_samples


### CRPS for Gaussian Distributions

def normal_cdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the cumulative distribution function of the standard normal distribution.
    
    Uses the error function (erf) which is directly related to the normal CDF:
    Φ(x) = 0.5 * [1 + erf(x/√2)]
    """
    return 0.5 * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))


def normal_pdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute the probability density function of the standard normal distribution.
    
    φ(x) = 1/√(2π) * exp(-x²/2)
    """
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)


def crps_gaussian(
    y_true: NDArray[np.float64],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    aggregate: bool = True,
    sample_weight: Optional[NDArray[np.float64]] = None,
    min_sigma: float = 1e-7
) -> Union[float, NDArray[np.float64]]:
    """
    Compute the Continuous Ranked Probability Score (CRPS) for Gaussian predictive distributions.
    
    The CRPS is a proper scoring rule for distributional forecasts, specialized here
    for Gaussian distributions. For a normal distribution N(μ, σ²), the CRPS has a 
    closed-form solution.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Observed true values.
    mu : array-like of shape (n_samples,)
        Predicted means of the Gaussian distributions.
    sigma : array-like of shape (n_samples,)
        Predicted standard deviations of the Gaussian distributions.
    aggregate : bool, default=True
        If True, returns mean CRPS. If False, returns per-sample CRPS.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If None, samples are equally weighted.
    min_sigma : float, default=1e-7
        Minimum allowed value for sigma to prevent numerical instability.
        
    Returns
    -------
    Union[float, NDArray[np.float64]]
        CRPS score (float if aggregate=True, array of shape (n_samples,) if False).
        Lower values indicate better predictions, with 0 being perfect.
    """
    # Input validation and conversion
    y_true = np.asarray(y_true, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Shape validation
    if y_true.shape != mu.shape or y_true.shape != sigma.shape:
        raise ValueError(
            f"Shapes must match: y_true {y_true.shape}, mu {mu.shape}, "
            f"sigma {sigma.shape}"
        )
    
    # Validate and process sample weights
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if sample_weight.shape != y_true.shape:
            raise ValueError(
                f"Sample weight shape {sample_weight.shape} does not match "
                f"y_true shape {y_true.shape}"
            )
        if np.any(sample_weight < 0):
            raise ValueError("Sample weights must be non-negative")
    
    # Handle numerical stability for sigma
    if np.any(sigma <= 0):
        raise ValueError(
            f"Standard deviations must be positive. Min value: {np.min(sigma)}"
        )
    sigma = np.maximum(sigma, min_sigma)
    
    # Compute standardized residuals
    z = (y_true - mu) / sigma
    
    # Compute CDF and PDF using numpy implementations
    Phi_z = normal_cdf(z)
    phi_z = normal_pdf(z)
    
    # Compute CRPS using vectorized operations
    crps_values = sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1.0 / np.sqrt(np.pi))
    
    # Ensure non-negativity (handle numerical errors)
    crps_values = np.maximum(crps_values, 0.0)
    
    if not aggregate:
        return crps_values
    
    # Compute weighted mean if sample weights provided
    if sample_weight is not None:
        return np.average(crps_values, weights=sample_weight)
    
    return np.mean(crps_values)


def crps_ensemble_gaussian(
    y_true: NDArray[np.float64],
    ensemble_preds: NDArray[np.float64],
    aggregate: bool = True
) -> Union[float, NDArray[np.float64]]:
    """
    Compute CRPS for ensemble predictions by fitting Gaussian distributions.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Observed true values.
    ensemble_preds : array-like of shape (n_samples, n_ensemble_members)
        Ensemble predictions.
    aggregate : bool, default=True
        If True, returns mean CRPS. If False, returns per-sample CRPS.
    """
    # Compute ensemble statistics
    mu = np.mean(ensemble_preds, axis=1)
    sigma = np.std(ensemble_preds, axis=1, ddof=1)  # Use n-1 for sample std
    
    return crps_gaussian(y_true, mu, sigma, aggregate=aggregate)


def crps_decomposition(
    y_true: NDArray[np.float64],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    n_bins: int = 10
) -> Tuple[float, float, float]:
    """
    EXPERIMENTAL: USE WITH CAUTION

    Decompose CRPS into reliability, resolution, and uncertainty components.
    
    The decomposition follows the traditional Murphy (1973) decomposition of proper
    scoring rules into REL - RES + UNC, where:
    - REL (reliability): measures the conditional bias of predictions
    - RES (resolution): measures the ability to discriminate between different events
    - UNC (uncertainty): represents the inherent variability of the observations
    
    Parameters
    ----------
    y_true : NDArray[np.float64]
        Observed true values, shape (n_samples,)
    mu : NDArray[np.float64]
        Predicted means, shape (n_samples,)
    sigma : NDArray[np.float64]
        Predicted standard deviations, shape (n_samples,)
    n_bins : int, optional (default=10)
        Number of bins for computing reliability and resolution
        
    Returns
    -------
    Tuple[float, float, float]
        (reliability, resolution, uncertainty) components of the CRPS
        
    Notes
    -----
    The resolution term is computed by binning predictions and measuring how much
    the conditional means differ from the overall mean. This ensures resolution
    is non-negative and properly measures forecast discrimination ability.
    """
    if not (len(y_true) == len(mu) == len(sigma)):
        raise ValueError("All inputs must have the same length")
        
    n_samples = len(y_true)
    
    # Compute uncertainty (variance of observations)
    # This represents the inherent uncertainty in the data
    uncertainty = np.std(y_true) / np.sqrt(2 * np.pi)
    
    # Compute standardized residuals
    z = (y_true - mu) / sigma
    
    # Compute reliability (calibration)
    expected_cdf = normal_cdf(z)
    empirical_cdf = (z <= 0).astype(np.float64)
    reliability = np.mean((expected_cdf - empirical_cdf) ** 2)
    
    # Compute resolution using binning approach
    bin_edges = np.linspace(np.min(mu), np.max(mu), n_bins + 1)
    bin_indices = np.digitize(mu, bin_edges)
    
    resolution = 0.0
    climatological_mean = np.mean(y_true)
    
    # Calculate resolution for each bin
    for bin_idx in range(1, n_bins + 1):
        bin_mask = (bin_indices == bin_idx)
        bin_size = np.sum(bin_mask)
        
        if bin_size > 0:  # Only process non-empty bins
            # Compute mean observation in this bin
            bin_mean = np.mean(y_true[bin_mask])
            # Add contribution to resolution
            resolution += bin_size * (bin_mean - climatological_mean) ** 2
    
    # Normalize resolution by sample size
    resolution = resolution / n_samples
    
    return reliability, resolution, uncertainty


### Brier Score

def brier_score_multiclass(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.float64],
    aggregate: bool = True,
    sample_weight: Optional[NDArray[np.float64]] = None,
    normalize: bool = True
) -> Union[float, NDArray[np.float64]]:
    """
    Computes the Brier score for multi-class classification predictions.
    
    The Brier score measures the accuracy of probabilistic predictions by computing
    the mean squared error between predicted probabilities and actual outcomes.
    A perfect score is 0, and the worst score is 1 (if normalize=True) or 2 
    (if normalize=False).

    Parameters:
    -----------
    y_true : NDArray[np.int_]
        True class labels (integer values from 0 to n_classes-1)
        Shape: (n_samples,)
    y_pred : NDArray[np.float64]
        Predicted probabilities for each class
        Shape: (n_samples, n_classes)
    aggregate : bool, default=True
        If True returns mean Brier score, otherwise returns Brier score per sample
    sample_weight : Optional[NDArray[np.float64]], default=None
        Sample weights. If given, shape must be (n_samples,)
    normalize : bool, default=True 
        If True, normalizes the score by number of classes for better interpretability
        across different classification problems

    Returns:
    --------
    Union[float, NDArray[np.float64]]
        Brier score (float if aggregate=True, else array of shape (n_samples,))

    Raises:
    -------
    ValueError
        If input shapes are inconsistent or values are invalid
    

    Examples
    --------
    >>> y_true = np.array([0, 1, 2])
    >>> y_pred = np.array([[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
    >>> brier_score_multiclass(y_true, y_pred)
    """
    # Input validation
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1-dimensional, got shape {y_true.shape}")
    
    if y_pred.ndim != 2:
        raise ValueError(f"y_pred must be 2-dimensional, got shape {y_pred.shape}")
        
    n_samples, n_classes = y_pred.shape
    
    if len(y_true) != n_samples:
        raise ValueError(
            f"Length of y_true ({len(y_true)}) does not match number of samples "
            f"in y_pred ({n_samples})"
        )
    
    if not np.all(np.logical_and(y_true >= 0, y_true < n_classes)):
        raise ValueError(f"y_true values must be in range [0, {n_classes-1}]")
        
    # Validate probabilities sum to 1 (within numerical precision)
    prob_sums = np.sum(y_pred, axis=1)
    if not np.allclose(prob_sums, 1.0, rtol=1e-5, atol=1e-8):
        raise ValueError("Predicted probabilities must sum to 1 for each sample")
        
    # Handle sample weights
    if sample_weight is not None:
        if sample_weight.shape != (n_samples,):
            raise ValueError(
                f"sample_weight shape {sample_weight.shape} does not match "
                f"number of samples ({n_samples})"
            )
        if not np.all(sample_weight >= 0):
            raise ValueError("sample_weight cannot contain negative values")
            
    # Compute Brier score efficiently without explicit one-hot encoding
    # This reduces memory usage for large datasets
    row_indices = np.arange(n_samples)
    true_probs = y_pred[row_indices, y_true]
    squared_errors = np.square(y_pred)
    
    # Add contribution from true class
    squared_errors[row_indices, y_true] = np.square(true_probs - 1)
    
    # Sum across classes for each sample
    brier_per_sample = np.sum(squared_errors, axis=1)
    
    # Normalize if requested
    if normalize:
        brier_per_sample /= n_classes
        
    if not aggregate:
        return brier_per_sample
        
    # Compute weighted average if sample weights provided
    if sample_weight is not None:
        return np.average(brier_per_sample, weights=sample_weight)
        
    return np.mean(brier_per_sample)


def brier_score_components(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.float64],
    n_bins: int = 10
) -> Tuple[float, float, float]:
    """
    Decompose the Brier score into reliability, resolution, and uncertainty components.

    The Brier score can be decomposed into three meaningful components that provide insights into different aspects of the probabilistic predictions. 
    Here's how each component works:

    1. RELIABILITY (Also called Calibration):
    - Measures how well the predicted probabilities match observed frequencies
    - For each predicted probability level, compares the actual frequency of events
    - Lower values are better (0 is perfect)
    - Calculated by binning predictions and comparing bin averages to actual outcomes

    2. RESOLUTION:
    - Measures how much the predictions vary from the overall mean frequency 
    - Indicates ability to separate different cases
    - Higher values are better
    - Shows if model can identify situations with different probabilities of occurrence

    3. UNCERTAINTY:
    - Represents inherent uncertainty in the dataset
    - Depends only on the variability of observations
    - Independent of the predictions
    - Formula: mean(observation) * (1 - mean(observation))

    The relationship between components is:

    Brier Score = Reliability - Resolution + Uncertainty

    Where:
    - Reliability should be minimized (closer to 0)
    - Resolution should be maximized
    - Uncertainty is fixed for a given dataset

    This decomposition helps diagnose prediction issues:
    - High reliability → Poor calibration
    - Low resolution → Predictions don't separate cases well
    - High uncertainty → Inherently difficult prediction task


    Algorithm key steps are:

    1. Data Preparation:
    - Convert true labels to one-hot encoding for multi-class handling
    - Calculate overall mean frequency per class

    2. For Each Class:
    - Bin the predicted probabilities
    - Calculate bin statistics:
        * Average predicted probability
        * Average observed frequency
        * Number of samples

    3. Component Calculation:
    - Reliability: Squared difference between predicted and observed frequencies
    - Resolution: Squared difference between bin frequency and overall mean
    - Uncertainty: Based on overall class frequencies

    4. Important Details:
    - Proper normalization by total number of predictions
    - Handling empty bins
    - Per-class calculations for multi-class problems


    
    Parameters:
    -----------
    y_true : NDArray[np.int_]
        True class labels
        Shape: (n_samples,)
    y_pred : NDArray[np.float64]
        Predicted probabilities
        Shape: (n_samples, n_classes)
    n_bins : int, default=10
        Number of bins for probability calibration assessment
        
    Returns:
    --------
    Tuple[float, float, float]
        reliability: Measures how well calibrated the predictions are
        resolution: Measures how well the model separates different classes
        uncertainty: Inherent uncertainty in the dataset
    """
    n_samples, n_classes = y_pred.shape
    
    # Convert to one-hot encoding for component calculation
    y_true_one_hot = np.zeros((n_samples, n_classes))
    y_true_one_hot[np.arange(n_samples), y_true] = 1
    
    # Calculate mean observed frequency
    mean_observed = np.mean(y_true_one_hot, axis=0)
    
    # Initialize components
    reliability = 0.0
    resolution = 0.0
    
    # Calculate components per class
    for c in range(n_classes):
        # Create bins for predicted probabilities
        pred_probs = y_pred[:, c]
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(pred_probs, bins) - 1
        
        # Calculate statistics for each bin
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if not np.any(mask):
                continue
                
            bin_size = np.sum(mask)
            bin_pred_mean = np.mean(pred_probs[mask])
            bin_true_mean = np.mean(y_true_one_hot[mask, c])
            
            # Update components
            reliability += bin_size * (bin_pred_mean - bin_true_mean) ** 2
            resolution += bin_size * (bin_true_mean - mean_observed[c]) ** 2
    
    # Normalize components
    reliability /= (n_samples * n_classes)
    resolution /= (n_samples * n_classes)
    
    # Calculate uncertainty
    uncertainty = np.mean(mean_observed * (1 - mean_observed))
    
    return reliability, resolution, uncertainty


## ECE Metric

def expected_calibration_error_simple(y_true: np.ndarray, y_pred_probs: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes the Expected Calibration Error (ECE) for a classification model.

    Parameters:
    - y_true: np.ndarray of shape (n_samples,), true class labels
    - y_pred_probs: np.ndarray of shape (n_samples, n_classes), predicted probabilities for each class
    - n_bins: int, number of bins to use for confidence intervals

    Returns:
    - ECE score (float)
    """
    # Get the predicted class and confidence (max probability)
    y_pred = np.argmax(y_pred_probs, axis=1)
    confidences = np.max(y_pred_probs, axis=1)  # Highest predicted probability

    # Define bins
    bin_edges = np.linspace(0, 1, n_bins + 1)  # Bin edges from 0 to 1
    bin_indices = np.digitize(confidences, bin_edges, right=True) - 1  # Assign each confidence to a bin

    ece = 0.0  # Initialize ECE

    for m in range(n_bins):
        bin_mask = bin_indices == m  # Select samples in bin m
        bin_size = np.sum(bin_mask)

        if bin_size > 0:
            acc_bin = np.mean(y_pred[bin_mask] == y_true[bin_mask])  # Accuracy in bin
            conf_bin = np.mean(confidences[bin_mask])  # Mean confidence in bin
            ece += (bin_size / len(y_true)) * abs(acc_bin - conf_bin)  # Weighted error

    return ece


def expected_calibration_error(
    y_true: NDArray[np.int64],
    y_pred_probs: NDArray[np.float64],
    n_bins: int = 10,
    sample_weight: Optional[NDArray[np.float64]] = None,
    return_details: bool = False
) -> Union[float, Tuple[float, dict]]:
    """
    Compute the Expected Calibration Error (ECE) for a classification model.
    
    ECE measures the difference between predicted probabilities (confidence)
    and actual accuracy, weighted by the frequency of predictions in each 
    confidence bin.
    
    Parameters
    ----------
    y_true : NDArray[np.int64]
        True class labels of shape (n_samples,)
    y_pred_probs : NDArray[np.float64]
        Predicted probabilities for each class, shape (n_samples, n_classes)
    n_bins : int, optional (default=10)
        Number of bins for confidence intervals
    sample_weight : NDArray[np.float64], optional (default=None)
        Sample weights of shape (n_samples,)
    return_details : bool, optional (default=False)
        If True, returns additional diagnostics
        
    Returns
    -------
    Union[float, Tuple[float, dict]]
        If return_details=False:
            float: ECE score
        If return_details=True:
            Tuple containing:
            - float: ECE score
            - dict: Additional metrics including:
                - 'reliability_diagram': Dict with 'accuracies', 'confidences', 
                   'counts' per bin
                - 'max_calibration_error': Maximum calibration error across bins
                - 'overconfident_bins': Number of overconfident bins
                - 'underconfident_bins': Number of underconfident bins
    
    Notes
    -----
    ECE is computed as:
    ECE = Σ(bin_size/n) * |accuracy_in_bin - confidence_in_bin|
    
    References
    ----------
    Guo, C., et al. "On Calibration of Modern Neural Networks."
    ICML 2017.
    
    Examples
    --------
    >>> y_true = np.array([0, 1, 2, 1])
    >>> y_pred_probs = np.array([[0.8, 0.1, 0.1],
    ...                         [0.2, 0.7, 0.1],
    ...                         [0.1, 0.2, 0.7],
    ...                         [0.3, 0.6, 0.1]])
    >>> ece = expected_calibration_error(y_true, y_pred_probs)
    """
    # Input validation
    y_true = np.asarray(y_true)
    y_pred_probs = np.asarray(y_pred_probs, dtype=np.float64)
    
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1-dimensional, got shape {y_true.shape}")
    
    if y_pred_probs.ndim != 2:
        raise ValueError(
            f"y_pred_probs must be 2-dimensional, got shape {y_pred_probs.shape}"
        )
        
    if len(y_true) != len(y_pred_probs):
        raise ValueError(
            f"Length mismatch: {len(y_true)} labels but {len(y_pred_probs)} "
            "prediction sets"
        )
    
    if not np.all(np.isclose(np.sum(y_pred_probs, axis=1), 1.0)):
        raise ValueError("Predicted probabilities must sum to 1 for each sample")
        
    if n_bins < 1:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if sample_weight.shape != y_true.shape:
            raise ValueError(
                f"Sample weight shape {sample_weight.shape} does not match "
                f"y_true shape {y_true.shape}"
            )
        if np.any(sample_weight < 0):
            raise ValueError("Sample weights must be non-negative")
    
    # Get predicted classes and confidences
    y_pred = np.argmax(y_pred_probs, axis=1)
    confidences = np.max(y_pred_probs, axis=1)
    
    # Create bins and assign predictions to bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges, right=True) - 1
    
    # Initialize metrics
    ece = 0.0
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    max_cal_error = 0.0
    overconfident_bins = 0
    underconfident_bins = 0
    
    # Calculate metrics for each bin
    total_weight = (
        np.sum(sample_weight) if sample_weight is not None 
        else len(y_true)
    )
    
    for m in range(n_bins):
        bin_mask = bin_indices == m
        bin_size = np.sum(bin_mask)
        
        if bin_size > 0:
            # Calculate weighted accuracy and confidence
            if sample_weight is not None:
                bin_weights = sample_weight[bin_mask]
                bin_weight_sum = np.sum(bin_weights)
                acc_bin = np.sum(
                    bin_weights * (y_pred[bin_mask] == y_true[bin_mask])
                ) / bin_weight_sum
                conf_bin = np.sum(
                    bin_weights * confidences[bin_mask]
                ) / bin_weight_sum
                weight_ratio = bin_weight_sum / total_weight
            else:
                acc_bin = np.mean(y_pred[bin_mask] == y_true[bin_mask])
                conf_bin = np.mean(confidences[bin_mask])
                weight_ratio = bin_size / len(y_true)
            
            # Update ECE and tracking metrics
            cal_error = abs(acc_bin - conf_bin)
            ece += weight_ratio * cal_error
            max_cal_error = max(max_cal_error, cal_error)
            
            # Track calibration direction
            if conf_bin > acc_bin:
                overconfident_bins += 1
            elif conf_bin < acc_bin:
                underconfident_bins += 1
            
            # Store bin statistics
            bin_accuracies[m] = acc_bin
            bin_confidences[m] = conf_bin
            bin_counts[m] = bin_size
    
    if not return_details:
        return ece
    
    details = {
        'reliability_diagram': {
            'accuracies': bin_accuracies,
            'confidences': bin_confidences,
            'counts': bin_counts,
            'edges': bin_edges
        },
        'max_calibration_error': max_cal_error,
        'overconfident_bins': overconfident_bins,
        'underconfident_bins': underconfident_bins
    }
    
    return ece, details
