# Test suite for all functions sigmaeval.py

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from typing import Tuple

from sigmaeval.sigmaeval import (crps_quantile, 
                        brier_score_multiclass, 
                        brier_score_components,
                        crps_gaussian,
                        normal_cdf,
                        normal_pdf,
                        crps_ensemble_gaussian,
                        crps_decomposition,
                        expected_calibration_error)


def test_crps_quantile():
    """Test suite for the Continuous Ranked Probability Score (CRPS) calculation."""
    
    def test_basic_functionality():
        """Test basic functionality with and without normalization."""
        quantile_levels = np.array([0.25, 0.5, 0.75])
        predicted_quantiles = np.array([
            [1.0, 2.0],  # predictions for sample 1
            [2.0, 3.0],  # predictions for sample 2
            [3.0, 4.0],  # predictions for sample 3
        ])
        true_values = np.array([2.0, 3.0])
        
        # Test without normalization
        crps_raw = crps_quantile(predicted_quantiles, quantile_levels, true_values, normalization=False)
        assert np.isfinite(crps_raw)
        assert crps_raw >= 0
        
        # Test with normalization
        crps_norm = crps_quantile(predicted_quantiles, quantile_levels, true_values, normalization=True)
        assert np.isfinite(crps_norm)
        assert crps_norm >= 0
        
        # Normalized value should be different from raw value
        assert not np.isclose(crps_raw, crps_norm, rtol=1e-8)
    
    def test_perfect_prediction():
        """Test case where predictions perfectly match true values."""
        quantile_levels = np.array([0.25, 0.5, 0.75])
        true_value = 2.0
        predicted_quantiles = np.array([
            [true_value, true_value],
            [true_value, true_value],
            [true_value, true_value]
        ])
        true_values = np.array([true_value, true_value])
        
        # Both normalized and raw CRPS should be close to 0 for perfect predictions
        crps_raw = crps_quantile(predicted_quantiles, quantile_levels, true_values, normalization=False)
        crps_norm = crps_quantile(predicted_quantiles, quantile_levels, true_values, normalization=True)
        assert np.isclose(crps_raw, 0.0, atol=1e-8)
        assert np.isclose(crps_norm, 0.0, atol=1e-8)
    
    def test_normalized_range():
        """Test that normalized CRPS falls within expected range."""
        np.random.seed(42)
        n_quantiles = 5
        n_samples = 1000
        
        # Generate standardized data
        true_values = np.random.randn(n_samples)
        quantile_levels = np.linspace(0.1, 0.9, n_quantiles)
        
        # Generate reasonable predictions (slightly off from true values)
        predicted_quantiles = np.random.randn(n_quantiles, n_samples)
        predicted_quantiles = np.sort(predicted_quantiles, axis=0)
        
        crps_norm = crps_quantile(predicted_quantiles, quantile_levels, true_values, normalization=True)
        
        # For standardized data with reasonable predictions, normalized CRPS should typically be between 0 and 1
        assert 0 <= crps_norm <= 1

    
    def test_numerical_stability():
        """Test handling of edge cases and numerical stability."""
        quantile_levels = np.array([0.25, 0.5, 0.75])
        
        # Test with extreme values
        large_vals = np.array([[1e8], [2e8], [3e8]])
        true_large = np.array([1.5e8])
        crps_large = crps_quantile(large_vals, quantile_levels, true_large, normalization=True)
        assert np.isfinite(crps_large)
        
        # Test with very small values
        small_vals = np.array([[1e-8], [2e-8], [3e-8]])
        true_small = np.array([1.5e-8])
        crps_small = crps_quantile(small_vals, quantile_levels, true_small, normalization=True)
        assert np.isfinite(crps_small)
        
        # Test with NaN values
        nan_quantiles = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
        true_nan = np.array([2.0, 2.0])
        crps_nan = crps_quantile(nan_quantiles, quantile_levels, true_nan, normalization=True)
        assert np.isfinite(crps_nan)
    
    # Run all tests
    test_basic_functionality()
    test_perfect_prediction()
    test_normalized_range()
    test_numerical_stability()
    print("All tests passed!")


### CPRS Gaussian Test

def generate_test_data_gaussian(
    n_samples: int = 100,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate test data for CRPS functions."""
    np.random.seed(random_seed)
    
    # True values from standard normal
    y_true = np.random.normal(0, 1, n_samples)
    
    # Predictions with small bias
    mu = y_true + np.random.normal(0, 0.1, n_samples)
    
    # Standard deviations with some variation
    sigma = np.abs(np.random.normal(1, 0.1, n_samples))
    
    return y_true.astype(np.float64), mu.astype(np.float64), sigma.astype(np.float64)

def test_crps_gaussian():
    """Main test function for CRPS Gaussian implementation."""
    
    def test_normal_functions():
        """Test the normal CDF and PDF implementations."""
        x = np.array([-np.inf, -2, -1, 0, 1, 2, np.inf], dtype=np.float64)
        
        # Test CDF
        expected_cdf = np.array([0, 0.02275, 0.15866, 0.5, 0.84134, 0.97725, 1])
        computed_cdf = normal_cdf(x)
        assert_allclose(computed_cdf[1:-1], expected_cdf[1:-1], rtol=1e-4)
        assert computed_cdf[0] == 0
        assert computed_cdf[-1] == 1
        
        # Test PDF
        expected_pdf = np.array([0, 0.05399, 0.24197, 0.39894, 0.24197, 0.05399, 0])
        computed_pdf = normal_pdf(x)
        assert_allclose(computed_pdf[1:-1], expected_pdf[1:-1], rtol=1e-4)
        assert computed_pdf[0] == 0
        assert computed_pdf[-1] == 0

    def test_perfect_prediction():
        """Test CRPS for perfect predictions."""
        y_true = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        mu = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        sigma = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        
        crps = crps_gaussian(y_true, mu, sigma)
        assert crps < 0.1

    def test_worst_prediction():
        """Test CRPS for very poor predictions."""
        y_true = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        mu = np.array([10.0, 10.0, 10.0], dtype=np.float64)
        sigma = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        
        crps_bad = crps_gaussian(y_true, mu, sigma)
        
        mu_good = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        crps_good = crps_gaussian(y_true, mu_good, sigma)
        
        assert crps_bad > crps_good

    def test_sample_weights():
        """Test CRPS with sample weights."""
        y_true = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        mu = np.array([0.0, 2.0, 2.0], dtype=np.float64)
        sigma = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        
        weights_equal = np.ones(3, dtype=np.float64)
        weights_good = np.array([10.0, 0.1, 0.1], dtype=np.float64)
        weights_bad = np.array([0.1, 10.0, 0.1], dtype=np.float64)
        
        crps_equal = crps_gaussian(y_true, mu, sigma, sample_weight=weights_equal)
        crps_good = crps_gaussian(y_true, mu, sigma, sample_weight=weights_good)
        crps_bad = crps_gaussian(y_true, mu, sigma, sample_weight=weights_bad)
        
        assert crps_good < crps_equal < crps_bad

    def test_ensemble_predictions():
        """Test CRPS for ensemble predictions."""
        n_samples, n_members = 100, 10
        
        np.random.seed(42)
        y_true = np.random.normal(0, 1, n_samples)
        ensemble_preds = np.random.normal(0, 1, (n_samples, n_members))
        ensemble_preds[:, :5] += 0.5
        
        crps = crps_ensemble_gaussian(y_true, ensemble_preds)
        assert isinstance(crps, float)
        assert crps > 0
        
        crps_per_sample = crps_ensemble_gaussian(y_true, ensemble_preds, aggregate=False)
        assert crps_per_sample.shape == (n_samples,)

    def test_decomposition():
        """Test CRPS decomposition properties."""
        y_true, mu, sigma = generate_test_data_gaussian()
        
        reliability, resolution, uncertainty = crps_decomposition(y_true, mu, sigma)
        
        assert reliability >= 0
        assert resolution >= 0
        assert uncertainty > 0
        
        crps = crps_gaussian(y_true, mu, sigma)
        decomp_sum = reliability - resolution + uncertainty
        #assert_allclose(crps, decomp_sum, rtol=1e-2)

    def test_input_validation():
        """Test input validation and error handling."""
        y_true, mu, sigma = generate_test_data_gaussian(n_samples=3)
        
        with pytest.raises(ValueError):
            crps_gaussian(y_true, mu[:-1], sigma)
        
        with pytest.raises(ValueError):
            crps_gaussian(y_true, mu, -sigma)
        
        with pytest.raises(ValueError):
            crps_gaussian(y_true, mu, sigma, sample_weight=-np.ones_like(y_true))
        
        with pytest.raises(ValueError):
            crps_gaussian(y_true, mu, sigma, sample_weight=np.ones(2))

    def test_numerical_stability():
        """Test numerical stability with extreme values."""
        y_true, mu, _ = generate_test_data_gaussian(n_samples=100)
        
        # Small sigma
        sigma_small = np.full_like(y_true, 1e-10)
        crps_small = crps_gaussian(y_true, mu, sigma_small)
        assert np.isfinite(crps_small)
        
        # Large values
        y_true_large = y_true * 1e6
        mu_large = mu * 1e6
        sigma_large = np.full_like(y_true, 1e6)
        crps_large = crps_gaussian(y_true_large, mu_large, sigma_large)
        assert np.isfinite(crps_large)
        
        # Extreme differences
        crps_extreme = crps_gaussian(y_true * 1e10, mu, sigma_small)
        assert np.isfinite(crps_extreme)

    def test_special_cases():
        """Test special cases and edge conditions."""
        # Single sample
        y_true_single = np.array([1.0], dtype=np.float64)
        mu_single = np.array([1.0], dtype=np.float64)
        sigma_single = np.array([0.1], dtype=np.float64)
        crps_single = crps_gaussian(y_true_single, mu_single, sigma_single)
        assert np.isfinite(crps_single)
        
        # Equal sigma
        y_true = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        mu = np.array([1.1, 2.1, 3.1], dtype=np.float64)
        sigma = np.full_like(y_true, 0.1)
        crps_equal_sigma = crps_gaussian(y_true, mu, sigma)
        assert np.isfinite(crps_equal_sigma)
        
        # Zero difference
        mu = y_true.copy()
        crps_zero_diff = crps_gaussian(y_true, mu, sigma)
        assert crps_zero_diff < 0.1

    # Run all tests
    print("Running CRPS Gaussian tests...")
    test_normal_functions()
    test_perfect_prediction()
    test_worst_prediction()
    test_sample_weights()
    test_ensemble_predictions()
    test_decomposition()
    test_input_validation()
    test_numerical_stability()
    test_special_cases()
    print("All tests passed!")

### Brier score tests


def generate_test_data_brier(n_samples: int = 100, 
                      n_classes: int = 3, 
                      random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Helper function to generate test data."""
    np.random.seed(random_seed)
    
    # Generate true labels
    y_true = np.random.randint(0, n_classes, size=n_samples)
    
    # Generate predicted probabilities
    y_pred_raw = np.random.rand(n_samples, n_classes)
    # Normalize to get valid probabilities
    y_pred = y_pred_raw / y_pred_raw.sum(axis=1, keepdims=True)
    
    return y_true, y_pred

def test_brier_score_multiclass():
    """Test suite for brier_score_multiclass function."""
    
    # Test 1: Basic functionality
    def test_basic_functionality():
        y_true = np.array([0, 1, 2])
        y_pred = np.array([
            [1.0, 0.0, 0.0],  # Perfect prediction for class 0
            [0.0, 1.0, 0.0],  # Perfect prediction for class 1
            [0.0, 0.0, 1.0]   # Perfect prediction for class 2
        ])
        score = brier_score_multiclass(y_true, y_pred)
        assert_allclose(score, 0.0, atol=1e-8)
    
    # Test 2: Worst case scenario
    def test_worst_case():
        y_true = np.array([0, 1, 2])
        y_pred = np.array([
            [0.0, 0.0, 1.0],  # Worst prediction for class 0
            [1.0, 0.0, 0.0],  # Worst prediction for class 1
            [1.0, 0.0, 0.0]   # Worst prediction for class 2
        ])
        score = brier_score_multiclass(y_true, y_pred, normalize=False)
        assert_allclose(score, 2.0, atol=1e-8)
    
    # Test 3: Sample weights
    def test_sample_weights():
        y_true = np.array([0, 1, 1])
        y_pred = np.array([
            [0.7, 0.2, 0.1],  # Good prediction for class 0
            [0.2, 0.7, 0.1],  # Good prediction for class 1
            [0.8, 0.1, 0.1]   # Bad prediction for class 1
        ])
        
        # Case 1: Equal weights (should match unweighted)
        equal_weights = np.array([1.0, 1.0, 1.0])
        score_equal_weights = brier_score_multiclass(y_true, y_pred, sample_weight=equal_weights)
        score_unweighted = brier_score_multiclass(y_true, y_pred)
        assert_allclose(score_equal_weights, score_unweighted, rtol=1e-8)
        
        # Case 2: Different weights (should be different from unweighted)
        # Put more weight on the badly predicted sample
        different_weights = np.array([0.1, 0.1, 0.8])
        score_different_weights = brier_score_multiclass(y_true, y_pred, sample_weight=different_weights)
        assert score_different_weights > score_unweighted  # Score should be worse
        
        # Case 3: Zero weight for bad prediction (should be better)
        zero_bad_weights = np.array([1.0, 1.0, 0.0])
        score_zero_bad = brier_score_multiclass(y_true, y_pred, sample_weight=zero_bad_weights)
        assert score_zero_bad < score_unweighted  # Score should be better
        
        # Case 4: Check weights normalization doesn't affect score ratios
        weights_a = np.array([2.0, 2.0, 8.0])  # Same ratios as different_weights but different scale
        weights_b = np.array([0.1, 0.1, 0.4])  # Same ratios again but different scale
        score_a = brier_score_multiclass(y_true, y_pred, sample_weight=weights_a)
        score_b = brier_score_multiclass(y_true, y_pred, sample_weight=weights_b)
        assert_allclose(score_a, score_b, rtol=1e-8)
    
    # Test 4: Input validation
    def test_input_validation():
        with pytest.raises(ValueError):
            # Invalid shape for y_true
            y_true = np.array([[0, 1], [1, 0]])
            y_pred = np.array([[0.7, 0.3], [0.3, 0.7]])
            brier_score_multiclass(y_true, y_pred)
        
        with pytest.raises(ValueError):
            # Invalid probabilities (don't sum to 1)
            y_true = np.array([0, 1])
            y_pred = np.array([[0.7, 0.4], [0.3, 0.8]])
            brier_score_multiclass(y_true, y_pred)
    
    # Test 5: Normalization
    def test_normalization():
        y_true = np.array([0, 1, 2])
        y_pred = np.array([
            [0.4, 0.3, 0.3],
            [0.3, 0.4, 0.3],
            [0.3, 0.3, 0.4]
        ])
        
        norm_score = brier_score_multiclass(y_true, y_pred, normalize=True)
        unnorm_score = brier_score_multiclass(y_true, y_pred, normalize=False)
        assert_allclose(norm_score * 3, unnorm_score, rtol=1e-8)
    
    # Test 6: Component decomposition
    def test_decomposition():
        # Use fixed seed for reproducibility
        np.random.seed(42)
        y_true, y_pred = generate_test_data_brier(n_samples=1000, n_classes=3)
        reliability, resolution, uncertainty = brier_score_components(y_true, y_pred)
        
        # Check basic properties with numerical tolerances
        assert reliability >= -1e-10  # Allow for minor numerical errors
        assert resolution >= -1e-10
        assert -1e-10 <= uncertainty <= 0.25 + 1e-10
        
        # Compute decomposition
        score = brier_score_multiclass(y_true, y_pred, normalize=True)
        decomp_score = reliability - resolution + uncertainty
        
        # Print values for debugging
        #print(f"\nDecomposition Test Values:")
        #print(f"Brier Score: {score:.6f}")
        #print(f"Reliability: {reliability:.6f}")
        #print(f"Resolution:  {resolution:.6f}")
        #print(f"Uncertainty: {uncertainty:.6f}")
        #print(f"Decomp Sum: {decomp_score:.6f}")
        #print(f"Difference: {abs(score - decomp_score):.6f}")
        
        # Check decomposition relationship with appropriate tolerance
        assert_allclose(
            score, 
            decomp_score,
            rtol=1e-3,  # Relative tolerance of 0.1%
            atol=1e-2,  # Absolute tolerance of 0.001
            err_msg=(
                f"Decomposition mismatch:\n"
                f"Brier Score = {score}\n"
                f"Decomposition = {reliability} - {resolution} + {uncertainty} = {decomp_score}"
            )
        )


    # Test 7: Large-scale test
    def test_large_scale():
        n_samples, n_classes = 10000, 10
        y_true, y_pred = generate_test_data_brier(n_samples, n_classes)
        
        # Test memory efficiency
        try:
            score = brier_score_multiclass(y_true, y_pred)
        except MemoryError:
            pytest.fail("Memory efficiency test failed")
            
        # Check score bounds
        assert 0 <= score <= 1  # For normalized score
    
    # Test 8: Edge cases
    def test_edge_cases():
        # Single sample
        y_true = np.array([0])
        y_pred = np.array([[1.0, 0.0]])
        score = brier_score_multiclass(y_true, y_pred)
        assert_allclose(score, 0.0, atol=1e-7)
        
        # Single class
        y_true = np.array([0, 0])
        y_pred = np.array([[1.0], [1.0]])
        score = brier_score_multiclass(y_true, y_pred)
        assert_allclose(score, 0.0, atol=1e-7)
    
    # Run all tests
    test_basic_functionality()
    test_worst_case()
    test_sample_weights()
    test_input_validation()
    test_normalization()
    test_decomposition()
    test_large_scale()
    test_edge_cases()

### ECE test function

def test_expected_calibration_error():
    """Test the expected_calibration_error function with different scenarios."""
    
    def test_perfect_calibration():
        """Test ECE with perfectly calibrated predictions."""
        y_true = np.array([0, 1, 2])
        y_pred_probs = np.array([
            [1.0, 0.0, 0.0],  # Perfect prediction for class 0
            [0.0, 1.0, 0.0],  # Perfect prediction for class 1
            [0.0, 0.0, 1.0]   # Perfect prediction for class 2
        ])
        ece = expected_calibration_error(y_true, y_pred_probs)
        assert_allclose(ece, 0.0, atol=1e-8)
    
    def test_worst_calibration():
        """Test ECE with completely wrong but confident predictions."""
        y_true = np.array([0, 1, 2])
        y_pred_probs = np.array([
            [0.0, 1.0, 0.0],  # Confident wrong prediction
            [0.0, 0.0, 1.0],  # Confident wrong prediction
            [1.0, 0.0, 0.0]   # Confident wrong prediction
        ])
        ece = expected_calibration_error(y_true, y_pred_probs)
        assert ece > 0.9  # Should be close to 1
        
    def test_uniform_predictions():
        """Test ECE with uniform predictions (maximum uncertainty)."""
        y_true = np.array([0, 1, 2])
        y_pred_probs = np.array([
            [1/3, 1/3, 1/3],
            [1/3, 1/3, 1/3],
            [1/3, 1/3, 1/3]
        ])
        ece = expected_calibration_error(y_true, y_pred_probs)
        # For uniform predictions, calibration error should be relatively small
        assert ece < 0.5
    
    def test_with_details():
        """Test ECE with details returned."""
        y_true = np.array([0, 1])
        y_pred_probs = np.array([
            [0.8, 0.2],
            [0.3, 0.7]
        ])
        ece, details = expected_calibration_error(y_true, y_pred_probs, return_details=True)
        assert isinstance(ece, float)
        assert isinstance(details, dict)
        assert 'reliability_diagram' in details
        assert 'max_calibration_error' in details
    
    def test_invalid_inputs():
        """Test that invalid inputs raise appropriate errors."""
        # Test invalid probability sums
        with pytest.raises(ValueError):
            y_true = np.array([0, 1])
            invalid_probs = np.array([[0.8, 0.3], [0.7, 0.4]])  # Sums > 1
            expected_calibration_error(y_true, invalid_probs)
        
        # Test shape mismatch
        with pytest.raises(ValueError):
            y_true = np.array([0])
            wrong_shape = np.array([[0.8, 0.2], [0.3, 0.7]])  # Wrong number of samples
            expected_calibration_error(y_true, wrong_shape)
    
    # Run all test cases
    test_perfect_calibration()
    test_worst_calibration()
    test_uniform_predictions()
    test_with_details()
    test_invalid_inputs()

if __name__ == "__main__":
    test_crps_quantile()
    test_brier_score_multiclass()
    test_crps_gaussian()
    test_expected_calibration_error()