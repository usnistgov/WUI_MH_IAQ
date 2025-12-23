"""
Instrument Configuration and Constants for NIST WUI MH IAQ Analysis

This module provides instrument-specific configurations, bin definitions,
and constants used across multiple analysis scripts.

Constants:
    - QUANTAQ_BINS: Bin definitions for QuantAQ instruments
    - SMPS_BIN_RANGES: Size ranges for SMPS data binning
    - AEROTRAK_CHANNELS: Channel definitions for AeroTrak instruments
    - UNIT_CONVERSIONS: Common unit conversion factors

Author: Nathan Lima
Date: 2024-2025
"""

import numpy as np

# ============================================================================
# QuantAQ Bin Definitions
# ============================================================================

# QuantAQ particle count bins - maps size ranges to bin column names
QUANTAQ_BINS = {
    "Ʃ0.35-0.66µm (#/cm³)": ["bin0", "bin1"],
    "Ʃ0.66-1.0µm (#/cm³)": ["bin2"],
    "Ʃ1.0-3.0µm (#/cm³)": ["bin3", "bin4", "bin5", "bin6"],
    "Ʃ3.0-5.2µm (#/cm³)": ["bin7", "bin8"],
    "Ʃ5.2-10µm (#/cm³)": ["bin9", "bin10", "bin11"],
    "Ʃ10-20µm (#/cm³)": ["bin12", "bin13", "bin14", "bin15", "bin16"],
    "Ʃ20-40µm (#/cm³)": ["bin17", "bin18", "bin19", "bin20", "bin21", "bin22", "bin23"],
}

# QuantAQ PM mass concentration column mappings
QUANTAQ_PM_COLUMNS = {
    'pm1': 'PM1 (µg/m³)',
    'pm25': 'PM2.5 (µg/m³)',
    'pm10': 'PM10 (µg/m³)',
}

# ============================================================================
# SMPS Bin Definitions
# ============================================================================

# SMPS particle size ranges for binning (in nanometers)
# Format: (start_nm, end_nm)
SMPS_BIN_RANGES = [
    (9, 100),
    (100, 200),
    (200, 300),
    (300, 437),
]

# SMPS output column names for binned data
SMPS_BIN_COLUMNS = [f"Ʃ{start}-{end}nm (#/cm³)" for start, end in SMPS_BIN_RANGES]

# ============================================================================
# AeroTrak Channel Definitions
# ============================================================================

# AeroTrak standard size channels
AEROTRAK_CHANNELS = ["Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6"]

# AeroTrak column mappings
AEROTRAK_COLUMN_MAPPINGS = {
    'datetime': 'Date and Time',
    'volume': 'Volume (L)',
    'flow_status': 'Flow Status',
    'laser_status': 'Laser Status',
}

# AeroTrak status columns to check for data quality
AEROTRAK_STATUS_COLUMNS = ["Flow Status", "Laser Status"]

# ============================================================================
# DustTrak Configuration
# ============================================================================

DUSTTRAK_COLUMN_MAPPINGS = {
    'TOTAL [mg/m3]': 'PM15 (µg/m³)',  # Convert mg/m³ to µg/m³
}

# ============================================================================
# MiniAMS Configuration
# ============================================================================

# MiniAMS only operates for burns 1-3
MINIAMS_BURN_RANGE = ['burn1', 'burn2', 'burn3']

# MiniAMS column mappings
MINIAMS_COLUMN_MAPPINGS = {
    'Org': 'Organic (µg/m³)',
    'NO3': 'Nitrate (µg/m³)',
    'SO4': 'Sulfate (µg/m³)',
    'NH4': 'Ammonium (µg/m³)',
    'Chl': 'Chloride (µg/m³)',
}

# ============================================================================
# PurpleAir Configuration
# ============================================================================

# PurpleAir operates for burns 6-10
PURPLEAIR_BURN_RANGE = ['burn6', 'burn7', 'burn8', 'burn9', 'burn10']

# PurpleAir column mappings
PURPLEAIR_COLUMN_MAPPINGS = {
    'Average': 'PM2.5 (µg/m³)',
}

# ============================================================================
# Unit Conversions
# ============================================================================

UNIT_CONVERSIONS = {
    # Volume conversions
    'liter_to_cm3': 1000,
    'cm3_to_liter': 0.001,

    # Mass concentration conversions
    'mg_m3_to_ug_m3': 1000,
    'ug_m3_to_mg_m3': 0.001,

    # Time conversions
    'seconds_to_hours': 1 / 3600,
    'minutes_to_hours': 1 / 60,
    'hours_to_minutes': 60,
    'hours_to_seconds': 3600,
}

# ============================================================================
# Burn-Specific Special Cases
# ============================================================================

# Burn 3 requires rolling average to reduce noise
BURN3_ROLLING_WINDOW_MINUTES = 5

# Burn 6 custom decay end time offset (hours)
BURN6_DECAY_END_OFFSET = 0.25

# ============================================================================
# Experiment Event Configuration
# ============================================================================

# Common event types in experiments
EVENT_TYPES = {
    'garage_closed': 'garage closed',
    'cr_box_on': 'CR Box on',
    'burn_start': 'burn start',
    'burn_end': 'burn end',
}

# ============================================================================
# Standard Burn Configuration
# ============================================================================

# All burn IDs in the experiment
ALL_BURNS = [f"burn{i}" for i in range(1, 11)]

# Burn ranges for specific instruments
BURN_RANGES = {
    'AeroTrakB': ALL_BURNS,
    'AeroTrakK': ALL_BURNS,
    'QuantAQB': [f"burn{i}" for i in range(4, 11)],
    'QuantAQK': [f"burn{i}" for i in range(4, 11)],
    'SMPS': ALL_BURNS,
    'DustTrak': ALL_BURNS,
    'MiniAMS': ['burn1', 'burn2', 'burn3'],
    'PurpleAir': [f"burn{i}" for i in range(6, 11)],
    'PurpleAirK': [f"burn{i}" for i in range(6, 11)],
}

# ============================================================================
# Helper Functions
# ============================================================================


def get_quantaq_bins():
    """
    Get QuantAQ bin definitions.

    Returns:
    --------
    dict
        Dictionary mapping bin names to lists of bin columns

    Examples:
    ---------
    >>> bins = get_quantaq_bins()
    >>> bins['Ʃ1.0-3.0µm (#/cm³)']
    ['bin3', 'bin4', 'bin5', 'bin6']
    """
    return QUANTAQ_BINS.copy()


def get_smps_bin_ranges():
    """
    Get SMPS bin ranges.

    Returns:
    --------
    list of tuples
        List of (start_nm, end_nm) tuples

    Examples:
    ---------
    >>> ranges = get_smps_bin_ranges()
    >>> ranges[0]
    (9, 100)
    """
    return SMPS_BIN_RANGES.copy()


def get_aerotrak_channels():
    """
    Get AeroTrak channel names.

    Returns:
    --------
    list of str
        List of channel names

    Examples:
    ---------
    >>> channels = get_aerotrak_channels()
    >>> 'Ch1' in channels
    True
    """
    return AEROTRAK_CHANNELS.copy()


def get_burn_range_for_instrument(instrument):
    """
    Get valid burn range for a specific instrument.

    Parameters:
    -----------
    instrument : str
        Instrument name (e.g., 'AeroTrakB', 'QuantAQB', 'MiniAMS')

    Returns:
    --------
    list of str
        List of burn IDs valid for this instrument

    Examples:
    ---------
    >>> burns = get_burn_range_for_instrument('QuantAQB')
    >>> 'burn4' in burns
    True
    >>> 'burn1' in burns
    False

    >>> burns = get_burn_range_for_instrument('AeroTrakB')
    >>> len(burns)
    10
    """
    return BURN_RANGES.get(instrument, ALL_BURNS).copy()


def convert_unit(value, conversion_type):
    """
    Convert value using standard conversion factor.

    Parameters:
    -----------
    value : float or array-like
        Value(s) to convert
    conversion_type : str
        Type of conversion (key from UNIT_CONVERSIONS)

    Returns:
    --------
    float or array-like
        Converted value(s)

    Examples:
    ---------
    >>> convert_unit(1.5, 'liter_to_cm3')
    1500.0

    >>> convert_unit(1000, 'mg_m3_to_ug_m3')
    1000000.0

    >>> convert_unit(120, 'minutes_to_hours')
    2.0
    """
    if conversion_type not in UNIT_CONVERSIONS:
        raise ValueError(f"Unknown conversion type: {conversion_type}")

    return value * UNIT_CONVERSIONS[conversion_type]


def sum_quantaq_bins(data, bins=None):
    """
    Sum QuantAQ bins to create size range columns.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing QuantAQ bin columns (bin0, bin1, ...)
    bins : dict, optional
        Custom bin definitions (default: QUANTAQ_BINS)

    Returns:
    --------
    pd.DataFrame
        DataFrame with added summed bin columns

    Examples:
    ---------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'bin0': [1, 2, 3],
    ...     'bin1': [1, 2, 3],
    ...     'bin2': [2, 3, 4]
    ... })
    >>> result = sum_quantaq_bins(df)
    >>> 'Ʃ0.35-0.66µm (#/cm³)' in result.columns
    True
    """
    import pandas as pd

    if bins is None:
        bins = QUANTAQ_BINS

    data = data.copy()

    for new_col, bin_list in bins.items():
        # Check if all required bins exist
        if all(bin_col in data.columns for bin_col in bin_list):
            data[new_col] = data[bin_list].sum(axis=1)

    return data


def sum_smps_ranges(data, ranges=None):
    """
    Sum SMPS columns by size ranges.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing SMPS size columns (column names are sizes in nm)
    ranges : list of tuples, optional
        Custom size ranges (default: SMPS_BIN_RANGES)

    Returns:
    --------
    pd.DataFrame
        DataFrame with added summed range columns

    Examples:
    ---------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     10.0: [1, 2],
    ...     50.0: [2, 3],
    ...     150.0: [3, 4]
    ... })
    >>> result = sum_smps_ranges(df)
    >>> 'Ʃ9-100nm (#/cm³)' in result.columns
    True
    """
    import pandas as pd

    if ranges is None:
        ranges = SMPS_BIN_RANGES

    data = data.copy()

    for start, end in ranges:
        # Find columns within this range
        numeric_cols = []
        for col in data.columns:
            try:
                val = float(col)
                if start <= val <= end:
                    numeric_cols.append(col)
            except (ValueError, TypeError):
                continue

        if numeric_cols:
            bin_column_name = f"Ʃ{start}-{end}nm (#/cm³)"
            data[bin_column_name] = data[numeric_cols].sum(axis=1)

    return data


def get_instrument_datetime_column(instrument):
    """
    Get the datetime column name for a specific instrument.

    Parameters:
    -----------
    instrument : str
        Instrument name

    Returns:
    --------
    str
        Datetime column name

    Examples:
    ---------
    >>> get_instrument_datetime_column('AeroTrakB')
    'Date and Time'

    >>> get_instrument_datetime_column('QuantAQB')
    'timestamp_local'
    """
    datetime_columns = {
        'AeroTrakB': 'Date and Time',
        'AeroTrakK': 'Date and Time',
        'QuantAQB': 'timestamp_local',
        'QuantAQK': 'timestamp_local',
        'SMPS': 'datetime',
        'DustTrak': 'Date and Time',
        'MiniAMS': 'datetime',
        'PurpleAir': 'datetime',
        'PurpleAirK': 'datetime',
    }
    return datetime_columns.get(instrument, 'datetime')
