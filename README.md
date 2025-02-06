# Sigmaeval: Scoring Uncertainty and Quantile Predictions

 
**Sigmaeval** provides a collection of evaluation metrics specifically designed for scoring probabilistic predictions, uncertainty estimates, and quantile predictions. These metrics are particularly useful for models that output probability distributions or quantile estimates instead of point predictions. It can be used for both **probabilistic regression** and **classification models**, making it versatile for a range of machine learning tasks.  Unlike metrics from `scikit-learn`, which focus on point prediction accuracy, **Sigmaeval** evaluates the quality of uncertainty estimates and probabilistic forecasts without replacing them.

While there are some libraries, such as [scoringrules](https://github.com/frazane/scoringrules?utm_source=chatgpt.com) and [UNIQUE](https://github.com/Novartis/UNIQUE?utm_source=chatgpt.com), that provide metrics for probabilistic forecasts, Sigmaeval fills an important gap by offering specialized evaluation tools specifically designed for assessing the quality of uncertainty and quantile predictions.


## Features

- **CRPS (Continuous Ranked Probability Score)**: Measures the accuracy of probabilistic predictions by comparing predicted cumulative distributions to the actual outcomes.
- **Brier Score**: Evaluates the calibration of probability predictions for classification problems.
- **Quantile Prediction Metrics**: Designed to assess the quality of quantile predictions.
- **Uncertainty Decomposition**: Metrics like Brier score components help break down prediction uncertainties into reliability, resolution, and uncertainty components.
  
These functions are especially relevant in fields such as Bayesian machine learning, ensemble methods, and probabilistic forecasting, where understanding the uncertainty in model predictions is crucial. They are also relevant for probabilistic transformer-based models such as **TabPFN**, as it produces full posterior predictive distributions. By using Continuous Ranked Probability Score (CRPS) and Brier score-based metrics, this package helps quantifying how well these distributions reflect true uncertainties.

## Installation

To install **Sigmaeval**, you can use pip:

```bash
pip install sigmaeval
```

## How-To Example with TabPFN

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from tabpfn import TabPFNRegressor
from sigmaeval.sigmaeval import crps_quantile

# Load dataset (regression)
X, y = datasets.fetch_california_housing(return_X_y=True)
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 500, test_size=500, random_state=42)

# Train TabPFN model
model_pfn = TabPFNRegressor()
model_pfn.fit(X_train, y_train)

# define prediction quantiles 
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
pred_quantiles = model_pfn.predict(X_test, output_type="quantiles", quantiles=quantiles)
pred_quantiles = np.asarray(pred_quantiles)

# Calculate average CRPS
crps = crps_quantile(pred_quantiles, quantiles, y_test)
```

For more use-case examples see notebook `sigmaeval_examples.ipynb`.



## CRPS Computation

The **Continuous Ranked Probability Score (CRPS)** is a metric for evaluating the accuracy of probabilistic forecasts, particularly in regression tasks. It compares the predicted cumulative distribution function (CDF) with the empirical CDF of the observations.

### CRPS Formula

The CRPS is defined as the squared difference between the predicted CDF and the empirical CDF, integrated over all possible outcomes. For a given sample $x$ with predicted quantiles $q(\alpha)$ for each quantile level $\alpha$, the CRPS for a single sample can be written as:

$$
\text{CRPS}(x) = \int_{-\infty}^{\infty} \left[ F_{\text{pred}}(u) - F_{\text{emp}}(u) \right]^2 du
$$

Where:
* $F_{\text{pred}}(u)$ is the predicted CDF of the forecasted distribution, derived from quantiles
* $F_{\text{emp}}(u)$ is the empirical CDF based on the true observed value $x$

Since directly computing the integral is often not feasible, the CRPS is approximated by summing the squared differences between the predicted CDF values at discrete quantiles and the indicator function of the true value (the empirical CDF), as detailed below:

### Computation Method

Given the **predicted quantiles** and the **true value**, the CRPS is computed as follows:

1. **Empirical CDF**: The empirical CDF of the observed true value $y_{\text{true}}$ is represented as an indicator function. This function is 0 for values less than $y_{\text{true}}$ and 1 for values greater than or equal to $y_{\text{true}}$.

$$
F_{\text{emp}}(y_{\text{true}}) = \begin{aligned}
&0 && \text{if } y < y_{\text{true}} \\
&1 && \text{if } y \geq y_{\text{true}}
\end{aligned}
$$

2. **Predicted CDF**: The predicted CDF at each quantile level $\alpha$ is obtained by sorting the predicted quantiles for each sample. For each quantile level $\alpha$, the CDF value is computed based on the predicted quantiles.

3. **CRPS Computation**: The CRPS is then calculated by summing the squared difference between the empirical CDF and the predicted CDF across all quantile levels, weighted by the difference between consecutive quantiles:

$$
\text{CRPS}_i = \sum_{\alpha} (F_{\text{pred}}(\alpha) - F_{\text{emp}}(\alpha))^2 \Delta q(\alpha)
$$


Where:
* $F_{\text{pred}}(\alpha)$ is the predicted CDF at quantile level $\alpha$
* $F_{\text{emp}}(\alpha)$ is the empirical CDF at the true value $y_{\text{true}}$
* $\Delta q(\alpha) = q(\alpha_{i+1}) - q(\alpha_i)$ is the difference between consecutive quantile levels

4. **Normalization**: To make the CRPS scale-independent, the score is often normalized by dividing by the standard deviation of the true values. This normalization ensures that the CRPS is not affected by the scale of the data, making it comparable across datasets.

5. **Final CRPS Score**: The CRPS score for all samples is computed by averaging the individual CRPS values across all data points:

$$
\text{CRPS} = \frac{1}{n} \sum_{i=1}^{n} \text{CRPS}_i
$$

   Where $n$ is the number of samples.

Normalized vs. Non-Normalized CRPS:

- **Normalized CRPS**: If the `normalization` parameter is set to **True**, the CRPS score is divided by the standard deviation of the true values. This helps to scale the CRPS score between 0 and 1, where smaller values indicate better forecasts.
  
- **Non-Normalized CRPS**: If normalization is disabled, the CRPS is computed directly, and its value ranges from 0 to $( +\infty )$, with 0 representing a perfect forecast.
 


### **Advantages of CRPS for Quantile-Based Predictions**

✅ **Handles Uncertainty Properly:** Unlike MSE, which only considers point predictions, CRPS evaluates the full distribution.  
✅ **Comparable Across Different Models:** Since CRPS measures the entire distribution’s performance, it is useful for comparing probabilistic models.  
✅ **Works with Prediction Intervals:** Even if the model only predicts quantiles, CRPS can approximate how well those quantiles match the actual distribution.  

## Computation of Brier Score

The **Brier Score** is a proper scoring rule that measures the accuracy of probabilistic predictions. For **multi-class classification**, the generalized **multi-class Brier Score** is defined as:

$$
\text{Brier Score} = \frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} (p_{ik} - y_{ik})^2
$$

where:
* $N$ is the number of samples
* $K$ is the number of classes
* $p_{ik}$ is the predicted probability for class $k$ in sample $i$
* $y_{ik}$ is a one-hot encoded ground-truth indicator (1 if true class, 0 otherwise)

A perfect prediction gives a Brier score of 0.

