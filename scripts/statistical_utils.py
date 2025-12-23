"""
Statistical Analysis Utilities for NIST WUI MH IAQ Analysis

This module provides common statistical functions used across multiple
analysis scripts in the NIST WUI MH IAQ repository.

Functions:
    - exponential_decay: Exponential decay model function
    - fit_exponential_curve: Fit exponential decay curve to data
    - perform_linear_fit: Linear regression with statistics
    - perform_polynomial_fit: Polynomial regression
    - select_best_fit: Automatic best fit selection using AIC
    - perform_z_test_comparison: Z-test for comparing two means
    - create_fitted_curve: Create smooth interpolated curves using splines

Author: Nathan Lima
Date: 2024-2025
"""

import numpy as np
import pandas as pd
from scipy import optimize, interpolate, stats


def exponential_decay(x, a, b):
    """
    Exponential decay model function.

    Model: y = a * exp(-b * x)

    Parameters:
    -----------
    x : array-like
        Independent variable (typically time)
    a : float
        Initial amplitude (y-intercept at x=0)
    b : float
        Decay rate constant

    Returns:
    --------
    array-like
        Exponential decay values

    Examples:
    ---------
    >>> x = np.array([0, 1, 2, 3])
    >>> y = exponential_decay(x, a=100, b=0.5)
    >>> y
    array([100.        ,  60.65306597,  36.78794412,  22.31301601])

    Notes:
    ------
    - Commonly used for modeling decay rates in air quality measurements
    - The parameter 'b' represents the decay rate (higher = faster decay)
    - The parameter 'a' represents the initial concentration
    """
    return a * np.exp(-b * x)


def fit_exponential_curve(x_data, y_data, initial_guess=None):
    """
    Fit exponential decay curve to data using non-linear least squares.

    Fits the model: y = a * exp(-b * x)

    Parameters:
    -----------
    x_data : array-like
        Independent variable values (typically time)
    y_data : array-like
        Dependent variable values (typically concentration)
    initial_guess : tuple of float, optional
        Initial guess for parameters (a, b)
        Default is (1, 1)

    Returns:
    --------
    tuple
        (popt, y_fit, std_err) where:
        - popt: Optimal parameters [a, b]
        - y_fit: Fitted y values
        - std_err: Standard errors of parameters

    Examples:
    ---------
    >>> x = np.array([0, 1, 2, 3, 4])
    >>> y = 100 * np.exp(-0.5 * x) + np.random.normal(0, 1, 5)
    >>> params, y_fit, errors = fit_exponential_curve(x, y)
    >>> params[0]  # Should be close to 100
    >>> params[1]  # Should be close to 0.5

    Notes:
    ------
    - Automatically removes NaN and infinite values
    - Removes non-positive y values (required for exponential fit)
    - Returns None, None, None if fitting fails
    - Standard errors calculated from covariance matrix diagonal
    """
    if initial_guess is None:
        initial_guess = (1, 1)

    # Remove NaN and infinite values
    mask = np.isfinite(x_data) & np.isfinite(y_data) & (y_data > 0)
    x_clean = np.array(x_data)[mask]
    y_clean = np.array(y_data)[mask]

    if len(x_clean) < 2:
        print("Not enough valid data points for curve fitting")
        return None, None, None

    try:
        # Fit the curve
        popt, pcov = optimize.curve_fit(
            exponential_decay, x_clean, y_clean, p0=initial_guess
        )

        # Calculate standard errors
        std_err = np.sqrt(np.diag(pcov))

        # Generate fitted values
        y_fit = exponential_decay(x_clean, *popt)

        return popt, y_fit, std_err

    except (RuntimeError, ValueError, optimize.OptimizeWarning) as e:
        print(f"Curve fitting failed: {str(e)}")
        return None, None, None


def perform_linear_fit(x_data, y_data):
    """
    Perform linear regression and calculate statistics.

    Fits the model: y = m * x + b

    Parameters:
    -----------
    x_data : array-like
        Independent variable values
    y_data : array-like
        Dependent variable values

    Returns:
    --------
    dict
        Dictionary containing:
        - 'slope': Slope of the line
        - 'intercept': Y-intercept
        - 'r_squared': Coefficient of determination
        - 'p_value': P-value for slope
        - 'std_err': Standard error of slope
        - 'y_fit': Fitted y values
        - 'aic': Akaike Information Criterion

    Examples:
    ---------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = 2 * x + 1 + np.random.normal(0, 0.1, 5)
    >>> result = perform_linear_fit(x, y)
    >>> result['slope']  # Should be close to 2
    >>> result['intercept']  # Should be close to 1

    Notes:
    ------
    - Removes NaN and infinite values automatically
    - Returns None if fitting fails or insufficient data
    - AIC calculated assuming normally distributed errors
    """
    # Remove NaN and infinite values
    mask = np.isfinite(x_data) & np.isfinite(y_data)
    x_clean = np.array(x_data)[mask]
    y_clean = np.array(y_data)[mask]

    if len(x_clean) < 2:
        return None

    try:
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x_clean, y_clean
        )

        # Calculate fitted values
        y_fit = slope * x_clean + intercept

        # Calculate residuals and AIC
        residuals = y_clean - y_fit
        rss = np.sum(residuals**2)
        n = len(y_clean)
        k = 2  # number of parameters (slope, intercept)
        aic = n * np.log(rss / n) + 2 * k

        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err,
            'y_fit': y_fit,
            'aic': aic,
        }

    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"Linear fit failed: {str(e)}")
        return None


def perform_polynomial_fit(x_data, y_data, degree=2):
    """
    Perform polynomial regression.

    Parameters:
    -----------
    x_data : array-like
        Independent variable values
    y_data : array-like
        Dependent variable values
    degree : int, optional
        Degree of polynomial
        Default is 2 (quadratic)

    Returns:
    --------
    dict
        Dictionary containing:
        - 'coefficients': Polynomial coefficients (highest degree first)
        - 'y_fit': Fitted y values
        - 'r_squared': Coefficient of determination
        - 'aic': Akaike Information Criterion

    Examples:
    ---------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = x**2 + 2*x + 1
    >>> result = perform_polynomial_fit(x, y, degree=2)
    >>> result['coefficients']  # Should be close to [1, 2, 1]

    Notes:
    ------
    - Removes NaN and infinite values automatically
    - Higher degree polynomials may overfit
    - AIC helps compare models with different degrees
    """
    # Remove NaN and infinite values
    mask = np.isfinite(x_data) & np.isfinite(y_data)
    x_clean = np.array(x_data)[mask]
    y_clean = np.array(y_data)[mask]

    if len(x_clean) < degree + 1:
        return None

    try:
        # Fit polynomial
        coefficients = np.polyfit(x_clean, y_clean, degree)

        # Calculate fitted values
        y_fit = np.polyval(coefficients, x_clean)

        # Calculate R-squared
        ss_res = np.sum((y_clean - y_fit) ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Calculate AIC
        n = len(y_clean)
        k = degree + 1  # number of parameters
        aic = n * np.log(ss_res / n) + 2 * k

        return {
            'coefficients': coefficients,
            'y_fit': y_fit,
            'r_squared': r_squared,
            'aic': aic,
        }

    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"Polynomial fit failed: {str(e)}")
        return None


def select_best_fit(x_data, y_data, models=None):
    """
    Automatically select best fitting model using AIC.

    Compares linear, quadratic, and cubic polynomial fits and returns
    the model with the lowest AIC (best fit while avoiding overfitting).

    Parameters:
    -----------
    x_data : array-like
        Independent variable values
    y_data : array-like
        Dependent variable values
    models : list of str, optional
        List of models to try: 'linear', 'quadratic', 'cubic'
        Default is ['linear', 'quadratic', 'cubic']

    Returns:
    --------
    dict
        Dictionary containing:
        - 'model_type': Best fitting model type
        - 'fit_result': Result dictionary from the fitting function
        - 'aic': AIC value of the best model

    Examples:
    ---------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = 2 * x + 1
    >>> result = select_best_fit(x, y)
    >>> result['model_type']  # Should be 'linear'

    Notes:
    ------
    - Lower AIC indicates better model
    - Automatically balances fit quality vs. model complexity
    """
    if models is None:
        models = ['linear', 'quadratic', 'cubic']

    results = {}

    if 'linear' in models:
        linear_result = perform_linear_fit(x_data, y_data)
        if linear_result:
            results['linear'] = linear_result

    if 'quadratic' in models:
        quad_result = perform_polynomial_fit(x_data, y_data, degree=2)
        if quad_result:
            results['quadratic'] = quad_result

    if 'cubic' in models:
        cubic_result = perform_polynomial_fit(x_data, y_data, degree=3)
        if cubic_result:
            results['cubic'] = cubic_result

    if not results:
        return None

    # Find model with lowest AIC
    best_model = min(results.items(), key=lambda x: x[1]['aic'])

    return {
        'model_type': best_model[0],
        'fit_result': best_model[1],
        'aic': best_model[1]['aic'],
    }


def perform_z_test_comparison(mu_a, sd_a, n_a, mu_b, sd_b, n_b, label_a='A', label_b='B'):
    """
    Perform z-test to compare two means.

    Parameters:
    -----------
    mu_a : float
        Mean of group A
    sd_a : float
        Standard deviation of group A
    n_a : int
        Sample size of group A
    mu_b : float
        Mean of group B
    sd_b : float
        Standard deviation of group B
    n_b : int
        Sample size of group B
    label_a : str, optional
        Label for group A (default 'A')
    label_b : str, optional
        Label for group B (default 'B')

    Returns:
    --------
    dict
        Dictionary containing:
        - 'z_statistic': Z-test statistic
        - 'p_value': Two-tailed p-value
        - 'significant': True if p < 0.05
        - 'comparison': String describing the result

    Examples:
    ---------
    >>> result = perform_z_test_comparison(10, 2, 30, 12, 2, 30, 'Control', 'Treatment')
    >>> result['significant']
    >>> result['comparison']

    Notes:
    ------
    - Uses two-tailed test
    - Assumes normal distribution
    - Significance level is 0.05
    """
    # Calculate standard error of the difference
    se_diff = np.sqrt((sd_a**2 / n_a) + (sd_b**2 / n_b))

    # Calculate z-statistic
    z_stat = (mu_a - mu_b) / se_diff

    # Calculate two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Determine significance
    significant = p_value < 0.05

    # Create comparison string
    if significant:
        if mu_a > mu_b:
            comparison = f"{label_a} is significantly higher than {label_b}"
        else:
            comparison = f"{label_a} is significantly lower than {label_b}"
    else:
        comparison = f"No significant difference between {label_a} and {label_b}"

    return {
        'z_statistic': z_stat,
        'p_value': p_value,
        'significant': significant,
        'comparison': comparison,
    }


def create_fitted_curve(x_data, y_data, num_points=100, kind='cubic'):
    """
    Create smooth interpolated curve using splines.

    Parameters:
    -----------
    x_data : array-like
        Independent variable values
    y_data : array-like
        Dependent variable values
    num_points : int, optional
        Number of points in the interpolated curve
        Default is 100
    kind : str, optional
        Type of interpolation: 'linear', 'quadratic', 'cubic'
        Default is 'cubic'

    Returns:
    --------
    tuple
        (x_smooth, y_smooth) where:
        - x_smooth: Interpolated x values
        - y_smooth: Interpolated y values

    Examples:
    ---------
    >>> x = np.array([0, 1, 2, 3, 4])
    >>> y = np.array([0, 1, 4, 9, 16])
    >>> x_smooth, y_smooth = create_fitted_curve(x, y, num_points=50)
    >>> len(x_smooth)
    50

    Notes:
    ------
    - Removes NaN and infinite values
    - Requires at least 4 points for cubic interpolation
    - Returns None, None if interpolation fails
    """
    # Remove NaN and infinite values
    mask = np.isfinite(x_data) & np.isfinite(y_data)
    x_clean = np.array(x_data)[mask]
    y_clean = np.array(y_data)[mask]

    # Sort by x values
    sort_idx = np.argsort(x_clean)
    x_clean = x_clean[sort_idx]
    y_clean = y_clean[sort_idx]

    min_points = {'linear': 2, 'quadratic': 3, 'cubic': 4}
    if len(x_clean) < min_points.get(kind, 4):
        print(f"Not enough points for {kind} interpolation")
        return None, None

    try:
        # Create interpolation function
        f = interpolate.interp1d(x_clean, y_clean, kind=kind)

        # Generate smooth curve
        x_smooth = np.linspace(x_clean.min(), x_clean.max(), num_points)
        y_smooth = f(x_smooth)

        return x_smooth, y_smooth

    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"Interpolation failed: {str(e)}")
        return None, None
