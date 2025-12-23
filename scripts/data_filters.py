"""
Data Filtering and Transformation Utilities for NIST WUI MH IAQ Analysis

This module provides common data filtering and transformation functions used
across multiple analysis scripts in the NIST WUI MH IAQ repository.

Functions:
    - filter_by_burn_dates: Filter data to specific burn dates
    - calculate_rolling_average_burn3: Apply 5-minute rolling average for burn3
    - split_data_by_nan: Split data into segments at NaN gaps
    - filter_by_status_columns: Filter data based on instrument status
    - g_mean: Calculate geometric mean

Author: Nathan Lima
Date: 2024-2025
"""

import pandas as pd
import numpy as np


def g_mean(x):
    """
    Calculate geometric mean of an array.

    The geometric mean is useful for data that are log-normally distributed,
    such as particle concentrations.

    Parameters:
    -----------
    x : array-like
        Input values (must be positive)

    Returns:
    --------
    float
        Geometric mean of the input values

    Examples:
    ---------
    >>> g_mean([1, 10, 100])
    10.0

    >>> g_mean([2, 8])
    4.0

    Notes:
    ------
    - Input values must be positive (geometric mean undefined for negative values)
    - Returns NaN if input contains NaN values
    """
    a = np.log(x)
    return np.exp(a.mean())


def filter_by_burn_dates(data, burn_range, datetime_column, burn_log):
    """
    Filter DataFrame to include only data from specified burn dates.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame to filter
    burn_range : list of str
        List of burn IDs to include (e.g., ['burn1', 'burn2', 'burn3'])
    datetime_column : str
        Name of the datetime column to use for date comparison
    burn_log : pd.DataFrame
        Burn log DataFrame with 'Burn ID' and 'Date' columns

    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame containing only rows from specified burns

    Examples:
    ---------
    >>> burn_log = pd.DataFrame({
    ...     'Burn ID': ['burn1', 'burn2'],
    ...     'Date': ['2024-01-01', '2024-01-02']
    ... })
    >>> data = pd.DataFrame({
    ...     'datetime': pd.date_range('2024-01-01', periods=3, freq='D'),
    ...     'value': [1, 2, 3]
    ... })
    >>> filtered = filter_by_burn_dates(data, ['burn1'], 'datetime', burn_log)
    >>> len(filtered)
    1
    """
    # Get dates for specified burns
    burn_dates = burn_log[burn_log['Burn ID'].isin(burn_range)]['Date'].tolist()
    burn_dates = pd.to_datetime(burn_dates).date

    # Ensure datetime column is datetime type
    data = data.copy()
    data['_temp_date'] = pd.to_datetime(data[datetime_column]).dt.date

    # Filter to only include specified burn dates
    filtered_data = data[data['_temp_date'].isin(burn_dates)].copy()

    # Remove temporary column
    filtered_data.drop(columns=['_temp_date'], inplace=True)

    return filtered_data


def calculate_rolling_average_burn3(data, datetime_column):
    """
    Calculate 5-minute rolling average for burn3 data.

    Burn3 data often has higher noise levels that benefit from smoothing.
    This function applies a 5-minute rolling average to numeric columns
    while preserving status columns.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing burn3 data
    datetime_column : str
        Name of the datetime column to use as index for rolling average

    Returns:
    --------
    pd.DataFrame
        DataFrame with rolling averages applied to numeric columns

    Examples:
    ---------
    >>> df = pd.DataFrame({
    ...     'datetime': pd.date_range('2024-01-01', periods=10, freq='1min'),
    ...     'concentration': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ...     'Flow Status': ['OK'] * 10
    ... })
    >>> smoothed = calculate_rolling_average_burn3(df, 'datetime')
    >>> # Numeric columns smoothed, status columns preserved

    Notes:
    ------
    - Only numeric columns are smoothed
    - Status columns ('Flow Status', 'Instrument Status', 'Laser Status')
      retain their first value
    - Returns original data if DataFrame is empty
    - Datetime column must be properly formatted
    """
    if data.empty:
        print("No data available for burn 3.")
        return data

    # Set datetime column as the index for rolling average calculation
    data_indexed = data.set_index(datetime_column)

    # Initialize a dictionary to hold the results
    rolling_avg_data = {}

    # Columns to calculate rolling averages (numeric only)
    numeric_columns = data_indexed.select_dtypes(include=[np.number]).columns

    # Calculate rolling averages for numeric columns
    for col in numeric_columns:
        rolling_avg_data[col] = (
            data_indexed[col]
            .rolling(pd.Timedelta(minutes=5))
            .mean()
            .astype(data_indexed[col].dtype)
        )

    # For status columns, keep the first value
    rolling_status_columns = ["Flow Status", "Instrument Status", "Laser Status"]
    for col in rolling_status_columns:
        if col in data_indexed.columns:
            rolling_avg_data[col] = data_indexed[col].iloc[0]

    # Create a new DataFrame with rolling averages and status values
    rolling_avg_df = pd.DataFrame(rolling_avg_data, index=data_indexed.index)

    # Reset index to bring datetime column back as a column
    rolling_avg_df.reset_index(inplace=True)

    return rolling_avg_df


def split_data_by_nan(df, x_col, y_col, gap_threshold_hours=0.1):
    """
    Split data into segments where NaN values or large gaps occur.

    This prevents plotting lines that connect across data gaps, which can
    be misleading in time series visualizations.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    x_col : str
        Column name for x values (typically time)
    y_col : str
        Column name for y values (typically concentration)
    gap_threshold_hours : float, optional
        Maximum gap in x values (hours) before splitting into segments
        Default is 0.1 hours (6 minutes)

    Returns:
    --------
    list of tuples
        List of (x_segment, y_segment) tuples, where each tuple contains
        numpy arrays of x and y values for a continuous segment

    Examples:
    ---------
    >>> df = pd.DataFrame({
    ...     'time': [0, 0.05, 0.1, 0.5, 0.55],
    ...     'value': [1, 2, 3, 4, 5]
    ... })
    >>> segments = split_data_by_nan(df, 'time', 'value', gap_threshold_hours=0.2)
    >>> len(segments)
    2
    >>> # First segment: [0, 0.05, 0.1], second segment: [0.5, 0.55]

    Notes:
    ------
    - Rows with NaN in either column are automatically removed
    - Data is sorted by x values before splitting
    - Segments with only one point are excluded
    - Gap threshold is in hours by default (adjust for different x units)
    """
    # Drop rows with NaN in either column
    valid_data = df.dropna(subset=[x_col, y_col])

    if valid_data.empty:
        return []

    # Sort by x values
    valid_data = valid_data.sort_values(by=x_col)

    # Get values as numpy arrays
    x = valid_data[x_col].values
    y = valid_data[y_col].values

    # Check for large gaps in x values
    dx = np.diff(x)
    gap_indices = np.where(dx > gap_threshold_hours)[0]

    # Split at gap indices
    segments = []
    start_idx = 0

    for gap_idx in gap_indices:
        segments.append((x[start_idx : gap_idx + 1], y[start_idx : gap_idx + 1]))
        start_idx = gap_idx + 1

    # Add the final segment
    if start_idx < len(x):
        segments.append((x[start_idx:], y[start_idx:]))

    return segments


def filter_by_status_columns(data, status_columns=None, valid_status="OK"):
    """
    Filter data based on instrument status columns.

    Sets numeric values to NaN for rows where any status column is not valid.
    This is commonly used for AeroTrak data which has Flow Status and Laser Status.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing data with status columns
    status_columns : list of str, optional
        List of status column names to check
        Default is ["Flow Status", "Laser Status"]
    valid_status : str, optional
        String indicating valid status
        Default is "OK"

    Returns:
    --------
    pd.DataFrame
        DataFrame with numeric values set to NaN where status is invalid

    Examples:
    ---------
    >>> df = pd.DataFrame({
    ...     'Flow Status': ['OK', 'Error', 'OK'],
    ...     'Laser Status': ['OK', 'OK', 'Error'],
    ...     'concentration': [10, 20, 30]
    ... })
    >>> filtered = filter_by_status_columns(df)
    >>> filtered['concentration'].tolist()
    [10.0, nan, nan]

    Notes:
    ------
    - Only numeric columns are modified (status columns remain unchanged)
    - Returns a copy of the input DataFrame
    - All status columns must be valid for a row to be kept
    """
    if status_columns is None:
        status_columns = ["Flow Status", "Laser Status"]

    # Create a copy to avoid modifying the original
    data = data.copy()

    # Check which status columns exist in the data
    existing_status_cols = [col for col in status_columns if col in data.columns]

    if not existing_status_cols:
        # No status columns found, return original data
        return data

    # Check if all existing status columns have valid status
    valid_status_mask = (data[existing_status_cols] == valid_status).all(axis=1)

    # Get numeric columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns

    # Set numeric values to NaN where status is invalid
    data.loc[~valid_status_mask, numeric_columns] = np.nan

    return data


def remove_outliers(data, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the data
    column : str
        Column name to check for outliers
    method : str, optional
        Method for outlier detection: 'iqr' (interquartile range) or 'std' (standard deviation)
        Default is 'iqr'
    threshold : float, optional
        Threshold multiplier for outlier detection
        For 'iqr': number of IQRs beyond Q1/Q3 (default 1.5)
        For 'std': number of standard deviations from mean (default 1.5, but typically 3)

    Returns:
    --------
    pd.DataFrame
        DataFrame with outliers in the specified column set to NaN

    Examples:
    ---------
    >>> df = pd.DataFrame({'value': [1, 2, 3, 4, 5, 100]})
    >>> cleaned = remove_outliers(df, 'value', method='iqr')
    >>> # The value 100 will be set to NaN

    Notes:
    ------
    - IQR method: values < Q1 - threshold*IQR or > Q3 + threshold*IQR
    - STD method: values > mean + threshold*std or < mean - threshold*std
    - Returns a copy of the input DataFrame
    """
    data = data.copy()

    if method == 'iqr':
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outlier_mask = (data[column] < lower_bound) | (data[column] > upper_bound)

    elif method == 'std':
        mean = data[column].mean()
        std = data[column].std()
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        outlier_mask = (data[column] < lower_bound) | (data[column] > upper_bound)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'std'")

    # Set outliers to NaN
    data.loc[outlier_mask, column] = np.nan

    return data


def resample_to_common_timebase(data, datetime_column, freq='1min', aggregation='mean'):
    """
    Resample data to a common time base.

    Useful for aligning data from multiple instruments with different sampling rates.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the data
    datetime_column : str
        Name of the datetime column
    freq : str, optional
        Resampling frequency (e.g., '1min', '30s', '5min')
        Default is '1min'
    aggregation : str or dict, optional
        Aggregation method: 'mean', 'median', 'sum', 'first', 'last', or
        dictionary mapping column names to aggregation methods
        Default is 'mean'

    Returns:
    --------
    pd.DataFrame
        Resampled DataFrame

    Examples:
    ---------
    >>> df = pd.DataFrame({
    ...     'datetime': pd.date_range('2024-01-01', periods=10, freq='30s'),
    ...     'value': range(10)
    ... })
    >>> resampled = resample_to_common_timebase(df, 'datetime', freq='1min')
    >>> # Data resampled to 1-minute intervals

    Notes:
    ------
    - Non-numeric columns are preserved using 'first' aggregation
    - Returns a DataFrame with datetime as the index
    """
    data = data.copy()

    # Set datetime as index
    data = data.set_index(datetime_column)

    # Resample
    if isinstance(aggregation, str):
        resampled = data.resample(freq).agg(aggregation)
    else:
        resampled = data.resample(freq).agg(aggregation)

    # Reset index
    resampled.reset_index(inplace=True)

    return resampled
