#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WUI Spatial Variation Analysis Script
=====================================

This script quantifies spatial variability of particulate matter (PM) concentrations
between two locations within a manufactured home test structure during wildfire smoke
infiltration experiments. The analysis compares bedroom2 (reference location) versus
the morning room (kitchen area) to characterize the uniformity of smoke distribution
and mitigation effectiveness across the indoor environment.

Key Metrics Calculated:
    - Peak Ratio Index (PRI): Ratio of peak PM concentrations between locations
    - CR Box Activation Ratio: Ratio of PM concentrations at CR Box activation time
    - Average Ratio: Time-averaged concentration ratio during decay period
    - Relative Standard Deviation (RSD): Coefficient of variation between locations

Analysis Features:
    - Multi-instrument processing (AeroTrak optical particle counters, QuantAQ sensors)
    - Size-resolved PM analysis (PM0.5, PM1, PM2.5, PM3, PM5, PM10, PM25)
    - Time-synchronized data alignment with burn experiment timeline
    - CR Box (portable air cleaner) activation-based analysis windows
    - Baseline correction and data quality filtering
    - Statistical outlier removal (ratios outside 0.1-10 range)

Methodology:
    1. Load peak concentration data from pre-processed Excel files
    2. Process time-series data from AeroTrak and QuantAQ instruments
    3. Apply instrument-specific time shifts and baseline corrections
    4. Calculate peak ratios from maximum concentrations during each burn
    5. Calculate CR Box activation ratios at the moment of air cleaner turn-on
    6. Calculate average ratios over 2-hour decay windows post-CR Box activation
    7. Compute RSD to quantify temporal variability in spatial ratios
    8. Export results to Excel with separate sheets for each instrument type

Output Files:
    - spatial_variation_analysis.xlsx: Complete results with AeroTrak and QuantAQ sheets

Applications:
    - Assess smoke mixing and stratification within the test house
    - Evaluate spatial uniformity assumptions for CADR calculations
    - Identify room-to-room transport characteristics
    - Validate single-point measurement representativeness

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2025
"""

import os
import warnings
import traceback
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Set the absolute path for the dataset
ABSOLUTE_PATH = "C:/Users/nml/OneDrive - NIST/Documents/NIST/WUI_smoke/"
os.chdir(ABSOLUTE_PATH)

# Output directory for results
OUTPUT_PATH = "C:/Users/nml/OneDrive - NIST/Documents/NIST/WUI_smoke/burn_data"

# Load burn log
BURN_LOG_PATH = "./burn_log.xlsx"
burn_log = pd.read_excel(BURN_LOG_PATH, sheet_name="Sheet2")

# Define instrument configurations (from wui_clean_air_delivery_rates_pmsizes_v5.py)
INSTRUMENT_CONFIG = {
    "AeroTrakB": {
        "file_path": "./burn_data/aerotraks/bedroom2/all_data.xlsx",
        "time_shift": 2.16,
        "process_pollutants": [
            "PM0.5 (µg/m³)",
            "PM1 (µg/m³)",
            "PM3 (µg/m³)",
            "PM5 (µg/m³)",
            "PM10 (µg/m³)",
            "PM25 (µg/m³)",
        ],
        "datetime_column": "Date and Time",
        "location": "bedroom2",
        "special_cases": {
            "burn3": {"apply_rolling_average": True},
            "burn6": {"custom_decay_time": True, "decay_end_offset": 0.25},
        },
        "baseline_values": {
            "PM0.5 (µg/m³)": (0.5121, 0.0079),
            "PM1 (µg/m³)": (0.5492, 0.0116),
            "PM3 (µg/m³)": (1.0855, 0.0511),
            "PM5 (µg/m³)": (2.0051, 0.0831),
            "PM10 (µg/m³)": (2.7994, 0.1160),
            "PM25 (µg/m³)": (3.3799, 0.1397),
        },
    },
    "AeroTrakK": {
        "file_path": "./burn_data/aerotraks/kitchen/all_data.xlsx",
        "time_shift": 5,
        "process_pollutants": [
            "PM0.5 (µg/m³)",
            "PM1 (µg/m³)",
            "PM3 (µg/m³)",
            "PM5 (µg/m³)",
            "PM10 (µg/m³)",
            "PM25 (µg/m³)",
        ],
        "datetime_column": "Date and Time",
        "location": "morning_room",
        "special_cases": {},
        "baseline_values": None,  # Will be calculated during processing
        "baseline_method": "weighted_average",
        "baseline_burns": ["burn5", "burn6"],
    },
    "QuantAQB": {
        "file_path": "./burn_data/quantaq/MOD-PM-00194-b0fc215029fa4852b926bc50b28fda5a.csv",
        "time_shift": -2.97,
        "process_pollutants": ["PM1 (µg/m³)", "PM2.5 (µg/m³)", "PM10 (µg/m³)"],
        "datetime_column": "timestamp_local",
        "burn_range": range(4, 11),
        "location": "bedroom2",
        "special_cases": {
            "burn6": {"custom_decay_time": True, "decay_end_offset": 0.25}
        },
    },
    "QuantAQK": {
        "file_path": "./burn_data/quantaq/MOD-PM-00197-a6dd467a147a4d95a7b98a8a10ab4ea3.csv",
        "time_shift": 0,
        "process_pollutants": ["PM1 (µg/m³)", "PM2.5 (µg/m³)", "PM10 (µg/m³)"],
        "datetime_column": "timestamp_local",
        "burn_range": range(4, 11),
        "location": "morning_room",
        "special_cases": {},
    },
}


# ============================================================================
# UTILITY FUNCTIONS (from wui_clean_air_delivery_rates_pmsizes_v5.py)
# ============================================================================
def apply_time_shift(df, instrument, burn_date):
    """Apply time shift based on instrument configuration

    Note: burn_id parameter removed as it was not used in the function.
    """
    time_shift = INSTRUMENT_CONFIG[instrument].get("time_shift", 0)
    datetime_column = INSTRUMENT_CONFIG[instrument].get(
        "datetime_column", "Date and Time"
    )

    # Ensure datetime column is in datetime format
    df.loc[:, datetime_column] = pd.to_datetime(df[datetime_column])

    # Convert burn_date to datetime
    burn_date = pd.to_datetime(burn_date).date()

    # Apply the shift only if shift value is non-zero
    if time_shift != 0:
        mask = df[datetime_column].dt.date == burn_date
        if mask.any():
            df.loc[mask, datetime_column] += pd.Timedelta(minutes=time_shift)

    return df


def create_naive_datetime(date_str, time_str):
    """Create a timezone-naive datetime object from date and time strings"""
    dt = pd.to_datetime(f"{date_str} {time_str}", errors="coerce")
    if hasattr(dt, "tz") and dt.tz is not None:
        dt = dt.tz_localize(None)
    return dt


def filter_by_burn_dates(data, burn_range, datetime_column):
    """Helper function to filter data by burn dates"""
    burn_ids = [f"burn{i}" for i in burn_range]
    burn_dates = burn_log[burn_log["Burn ID"].isin(burn_ids)]["Date"]
    burn_dates = pd.to_datetime(burn_dates)
    if datetime_column in data.columns:
        data["Date"] = pd.to_datetime(data[datetime_column]).dt.date
        return data[data["Date"].isin(burn_dates.dt.date)]
    else:
        raise KeyError(f"Column '{datetime_column}' not found in the dataset.")


def calculate_rolling_average_burn3(data):
    """Calculate 5-minute rolling average for burn3"""
    burn3_date = burn_log[burn_log["Burn ID"] == "burn3"]["Date"].values[0]
    burn3_date = pd.to_datetime(burn3_date).date()

    burn3_data = data[data["Date"] == burn3_date].copy()
    pollutant_columns = [col for col in burn3_data.columns if "(µg/m³)" in col]
    for col in pollutant_columns:
        # Ensure column is numeric before applying rolling average
        burn3_data.loc[:, col] = pd.to_numeric(burn3_data[col], errors="coerce")
        burn3_data.loc[:, col] = burn3_data[col].rolling(window=5, center=True).mean()
    # Replace original data with rolling average data for burn3
    data.loc[data["Date"] == burn3_date, pollutant_columns] = burn3_data[
        pollutant_columns
    ]
    return data


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================
def process_aerotrak_data(file_path, instrument="AeroTrakB"):
    """Process AeroTrak data with complete conversion from particle counts to mass concentration"""
    # Load the AeroTrak data from the Excel file
    aerotrak_data = pd.read_excel(file_path)
    # Strip whitespace from column names to avoid issues
    aerotrak_data.columns = aerotrak_data.columns.str.strip()
    # Define size channels and initialize a dictionary for size values
    size_channels = ["Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6"]
    size_values = {}
    # Extract size values for each channel
    for channel in size_channels:
        size_col = f"{channel} Size (µm)"
        if size_col in aerotrak_data.columns:
            size_value = aerotrak_data[size_col].iloc[0]
            if pd.notna(size_value):
                size_values[channel] = size_value
    # Check for the volume column and convert it to cm³
    volume_column = "Volume (L)"
    if volume_column in aerotrak_data.columns:
        aerotrak_data["Volume (cm³)"] = (
            aerotrak_data[volume_column] * 1000
        )  # Convert to cm³
        volume_cm = aerotrak_data["Volume (cm³)"]

    def g_mean(x):
        """Calculate geometric mean"""
        a = np.log(x)
        return np.exp(a.mean())

    # Initialize new columns for mass concentration and calculate values
    pm_columns = []  # List to store the names of new PM concentration columns
    for i, channel in enumerate(size_channels):
        if channel in size_values:
            next_channel = size_channels[i + 1] if i < len(size_channels) - 1 else None
            next_size_value = size_values[next_channel] if next_channel else 25
            # Calculate the geometric mean of the size range
            particle_size = g_mean([size_values[channel], next_size_value])
            particle_size_m = particle_size * 1e-6  # Convert size from µm to m
            # Initialize column name variable
            new_diff_col_µg_m3 = (
                f"PM{size_values[channel]}-{next_size_value} Diff (µg/m³)"
            )
            diff_col = f"{channel} Diff (#)"
            if diff_col in aerotrak_data.columns:
                particle_counts = aerotrak_data[diff_col]
                # Calculate the volume of a single particle
                radius_m = particle_size_m / 2
                volume_per_particle = (4 / 3) * np.pi * (radius_m**3)  # Volume in m³
                # Calculate particle mass density (1 g/cm³)
                particle_mass = volume_per_particle * 1e6 * 1e6  # Convert to µg
                # Ensure particle counts are numeric
                particle_counts = pd.to_numeric(particle_counts, errors="coerce")
                volume_cm = pd.to_numeric(volume_cm, errors="coerce")
                aerotrak_data[new_diff_col_µg_m3] = (
                    particle_counts / (volume_cm * 1e-6)
                ) * (particle_mass)
                # Add the new PM column name to the list
                pm_columns.append(new_diff_col_µg_m3)
                # Create new column for Diff (#/cm³)
                new_diff_col_cm3 = (
                    f"PM{size_values[channel]}-{next_size_value} Diff (#/cm³)"
                )
                aerotrak_data[new_diff_col_cm3] = (
                    particle_counts / volume_cm
                )  # Convert to #/cm³
            # Handle cumulative counts for PM concentrations
            cumul_col = f"{channel} Cumul (#)"
            if cumul_col in aerotrak_data.columns:
                new_cumul_col = f"{channel} Cumul (#/cm³)"
                # Ensure numeric conversion
                cumul_data = pd.to_numeric(aerotrak_data[cumul_col], errors="coerce")
                aerotrak_data[new_cumul_col] = (
                    cumul_data / volume_cm
                )  # Convert to #/cm³
                # Create new PM concentration column from the Diff column if it was created
                pm_column_name = f"PM{next_size_value} (µg/m³)"
                if (
                    diff_col in aerotrak_data.columns
                    and new_diff_col_µg_m3 in aerotrak_data.columns
                ):
                    aerotrak_data[pm_column_name] = aerotrak_data[
                        new_diff_col_µg_m3
                    ]  # Copy the concentration values
    # Define cumulative PM concentration columns
    cumulative_columns = [
        "PM0.5 (µg/m³)",
        "PM1 (µg/m³)",
        "PM3 (µg/m³)",
        "PM5 (µg/m³)",
        "PM10 (µg/m³)",
        "PM25 (µg/m³)",
    ]
    # Calculate cumulative PM concentrations
    for i, cumul_col in enumerate(cumulative_columns):
        if i == 0 and len(pm_columns) > 0:
            aerotrak_data[cumul_col] = pd.to_numeric(
                aerotrak_data[pm_columns[i]], errors="coerce"
            )
        elif i < len(pm_columns):
            # Sum current PM column with the previous cumulative column
            prev_cumul_col = cumulative_columns[i - 1]
            aerotrak_data[cumul_col] = pd.to_numeric(
                aerotrak_data[pm_columns[i]], errors="coerce"
            ).add(
                pd.to_numeric(aerotrak_data[prev_cumul_col], errors="coerce"),
                fill_value=0,
            )
    # Replace invalid entries with NaN for numeric columns only
    status_columns = ["Flow Status", "Laser Status"]
    if all(col in aerotrak_data.columns for col in status_columns):
        valid_status = (aerotrak_data[status_columns] == "OK").all(axis=1)
        for col in aerotrak_data.columns:
            if pd.api.types.is_numeric_dtype(aerotrak_data[col]) and col not in [
                "Date and Time",
                "Sample Time",
                "Volume (L)",
            ]:
                aerotrak_data.loc[~valid_status, col] = pd.NA
    # Filter based on burn dates
    burn_ids = [f"burn{i}" for i in range(1, 11)]
    burn_dates = burn_log[burn_log["Burn ID"].isin(burn_ids)]["Date"]
    burn_dates = pd.to_datetime(burn_dates)
    # Convert 'Date and Time' to date and filter AeroTrak data for the burn dates
    aerotrak_data["Date"] = pd.to_datetime(aerotrak_data["Date and Time"]).dt.date
    filtered_aerotrak_data = aerotrak_data[
        aerotrak_data["Date"].isin(burn_dates.dt.date)
    ]
    filtered_aerotrak_data = filtered_aerotrak_data.copy()

    # Apply time shift for each burn ID in the filtered data
    for burn_id in burn_ids:
        if burn_id in burn_log["Burn ID"].values:
            burn_date = burn_log[burn_log["Burn ID"] == burn_id]["Date"].values[0]
            filtered_aerotrak_data = apply_time_shift(
                filtered_aerotrak_data, instrument, burn_date
            )

    # Check if there's a special case for burn3 rolling average
    special_cases = INSTRUMENT_CONFIG[instrument].get("special_cases", {})
    if "burn3" in special_cases and special_cases["burn3"].get(
        "apply_rolling_average", False
    ):
        filtered_aerotrak_data = calculate_rolling_average_burn3(filtered_aerotrak_data)

    # Ensure all PM columns are numeric
    for col in cumulative_columns:
        if col in filtered_aerotrak_data.columns:
            filtered_aerotrak_data[col] = pd.to_numeric(
                filtered_aerotrak_data[col], errors="coerce"
            )

    # Apply baseline correction
    config = INSTRUMENT_CONFIG[instrument]
    baseline_values = config.get("baseline_values")

    if baseline_values:
        # Apply baseline correction to each pollutant
        for pollutant, (baseline_val, _) in baseline_values.items():
            # Note: baseline_uncertainty (_) is stored but not currently used in calculations
            if pollutant in filtered_aerotrak_data.columns:
                # Ensure column is numeric first
                filtered_aerotrak_data[pollutant] = pd.to_numeric(
                    filtered_aerotrak_data[pollutant], errors="coerce"
                )
                # Subtract baseline from measurements
                filtered_aerotrak_data[pollutant] = (
                    filtered_aerotrak_data[pollutant] - baseline_val
                )
                # Set negative values to 0
                filtered_aerotrak_data[pollutant] = filtered_aerotrak_data[
                    pollutant
                ].clip(lower=0)

    # Calculate Time Since Garage Closed for each burn
    for burn_id in burn_ids:
        burn_mask = (
            filtered_aerotrak_data["Date"]
            == pd.to_datetime(
                burn_log[burn_log["Burn ID"] == burn_id]["Date"].values[0]
                if burn_id in burn_log["Burn ID"].values
                else pd.NaT
            ).date()
        )

        if burn_mask.any() and burn_id in burn_log["Burn ID"].values:
            burn_info = burn_log[burn_log["Burn ID"] == burn_id]
            garage_closed_time_str = burn_info["garage closed"].iloc[0]
            burn_date = burn_info["Date"].iloc[0]

            if pd.notna(garage_closed_time_str) and garage_closed_time_str != "n/a":
                garage_closed_time = create_naive_datetime(
                    burn_date, garage_closed_time_str
                )

                # Calculate time since garage closed in hours
                filtered_aerotrak_data.loc[
                    burn_mask, "Time Since Garage Closed (hours)"
                ] = (
                    filtered_aerotrak_data.loc[burn_mask, "Date and Time"]
                    - garage_closed_time
                ).dt.total_seconds() / 3600

    return filtered_aerotrak_data


def process_quantaq_data(file_path, instrument="QuantAQB"):
    """Process QuantAQ data with standardized output format"""
    quantaq_data = pd.read_csv(file_path)
    quantaq_data = quantaq_data.iloc[::-1].reset_index(drop=True)
    quantaq_data.columns = quantaq_data.columns.str.strip()

    # Rename columns and ensure they're numeric
    rename_dict = {
        "pm1": "PM1 (µg/m³)",
        "pm25": "PM2.5 (µg/m³)",
        "pm10": "PM10 (µg/m³)",
    }
    quantaq_data.rename(columns=rename_dict, inplace=True)

    # Ensure PM columns are numeric
    for col in ["PM1 (µg/m³)", "PM2.5 (µg/m³)", "PM10 (µg/m³)"]:
        if col in quantaq_data.columns:
            quantaq_data[col] = pd.to_numeric(quantaq_data[col], errors="coerce")

    # Convert timestamp to datetime without timezone information
    quantaq_data["timestamp_local"] = pd.to_datetime(
        quantaq_data["timestamp_local"].str.replace("T", " ").str.replace("Z", ""),
        errors="coerce",
    ).dt.tz_localize(None)

    # Get burn range from instrument configuration
    burn_range = INSTRUMENT_CONFIG[instrument].get("burn_range", range(4, 11))

    # Filter by burn dates
    filtered_data = filter_by_burn_dates(quantaq_data, burn_range, "timestamp_local")

    # Apply time shift for each burn
    burn_ids = [f"burn{i}" for i in burn_range]
    for burn_id in burn_ids:
        if burn_id in burn_log["Burn ID"].values:
            burn_date = burn_log[burn_log["Burn ID"] == burn_id]["Date"].values[0]
            filtered_data = apply_time_shift(filtered_data, instrument, burn_date)

    # Calculate Time Since Garage Closed for each burn
    for burn_id in burn_ids:
        if burn_id in burn_log["Burn ID"].values:
            burn_info = burn_log[burn_log["Burn ID"] == burn_id]
            burn_date = pd.to_datetime(burn_info["Date"].iloc[0])

            burn_mask = filtered_data["Date"] == burn_date.date()

            if burn_mask.any():
                garage_closed_time_str = burn_info["garage closed"].iloc[0]

                if pd.notna(garage_closed_time_str) and garage_closed_time_str != "n/a":
                    garage_closed_time = create_naive_datetime(
                        burn_date, garage_closed_time_str
                    )

                    # Calculate time since garage closed in hours
                    filtered_data.loc[burn_mask, "Time Since Garage Closed (hours)"] = (
                        filtered_data.loc[burn_mask, "timestamp_local"]
                        - garage_closed_time
                    ).dt.total_seconds() / 3600

    return filtered_data


def load_peak_concentrations():
    """Load peak concentration data from Excel file"""
    peak_file = "./burn_data/peak_concentrations_all_instruments_edited.xlsx"

    try:
        # Read the data sheet
        peak_data = pd.read_excel(peak_file, sheet_name="data")

        print(f"Peak concentration data loaded: {peak_data.shape}")

        # Show actual AeroTrak and QuantAQ columns
        aerotrak_cols = [col for col in peak_data.columns if "AeroTrak" in col]
        quantaq_cols = [col for col in peak_data.columns if "QuantAQ" in col]

        print("\nActual AeroTrak columns in peak data:")
        for col in aerotrak_cols[:6]:  # Show first 6
            print(f"  '{col}'")

        print("\nActual QuantAQ columns in peak data:")
        for col in quantaq_cols[:6]:  # Show first 6
            print(f"  '{col}'")

        return peak_data
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error loading peak concentrations: {e}")
        return None


# ============================================================================
# RATIO CALCULATION FUNCTIONS
# ============================================================================
def calculate_peak_ratio(peak_data, burn_id, instrument_pair, pm_size):
    """Calculate peak concentration ratio between bedroom2 and morning room"""

    # The columns in peak data appear to be formatted as 'InstrumentName_PMx (µg/m³)'
    # with a space before the parenthesis

    if instrument_pair == "AeroTrak":
        bedroom_col = f"AeroTrakB_{pm_size}"
        morning_col = f"AeroTrakK_{pm_size}"
    else:  # QuantAQ
        bedroom_col = f"QuantAQB_{pm_size}"
        morning_col = f"QuantAQK_{pm_size}"

    # Check if columns exist
    available_cols = peak_data.columns.tolist()

    if bedroom_col not in available_cols or morning_col not in available_cols:
        return None

    # Get peak values for the specific burn
    burn_data = peak_data[peak_data["Burn_ID"] == burn_id]

    if burn_data.empty:
        return None

    bedroom_peak = burn_data[bedroom_col].values[0]
    morning_peak = burn_data[morning_col].values[0]

    # Check for valid data
    if pd.isna(bedroom_peak) or pd.isna(morning_peak) or morning_peak == 0:
        return None

    # Calculate ratio (bedroom2/morning room)
    ratio = bedroom_peak / morning_peak

    return ratio


def calculate_crbox_activation_ratio(
    bedroom_data,
    morning_data,
    burn_id,
    pm_size,
    datetime_col_b,
    datetime_col_m,
):
    """
    Calculate concentration ratio at CR Box activation time between bedroom2 and morning room

    This ratio represents the spatial variation at the moment the portable air cleaner
    was turned on, providing insight into smoke distribution at the start of decay.

    Parameters
    ----------
    bedroom_data : pd.DataFrame
        Bedroom2 concentration data
    morning_data : pd.DataFrame
        Morning room concentration data
    burn_id : str
        Burn identifier (e.g., 'burn4')
    pm_size : str
        PM size column name (e.g., 'PM2.5 (µg/m³)')
    datetime_col_b : str
        Datetime column name for bedroom data
    datetime_col_m : str
        Datetime column name for morning room data

    Returns
    -------
    float or None
        Ratio of bedroom2/morning room concentration at CR Box activation time,
        or None if data is unavailable
    """
    # Get burn information
    burn_info = burn_log[burn_log["Burn ID"] == burn_id]

    if burn_info.empty:
        return None

    burn_date = burn_info["Date"].iloc[0]
    cr_box_time_str = burn_info["CR Box on"].iloc[0]

    # Check if CR Box was used
    if pd.isna(cr_box_time_str) or cr_box_time_str == "n/a":
        return None

    # Create datetime for CR Box activation
    cr_box_time = create_naive_datetime(burn_date, cr_box_time_str)

    if pd.isna(cr_box_time):
        return None

    # Filter data for the current burn date
    burn_date_only = pd.to_datetime(burn_date).date()

    bedroom_burn_data = bedroom_data[bedroom_data["Date"] == burn_date_only].copy()
    morning_burn_data = morning_data[morning_data["Date"] == burn_date_only].copy()

    if bedroom_burn_data.empty or morning_burn_data.empty:
        return None

    # Find the closest measurement to CR Box activation time (within ±5 minutes)
    time_window = pd.Timedelta(minutes=5)

    bedroom_window = bedroom_burn_data[
        (bedroom_burn_data[datetime_col_b] >= cr_box_time - time_window)
        & (bedroom_burn_data[datetime_col_b] <= cr_box_time + time_window)
    ].copy()

    morning_window = morning_burn_data[
        (morning_burn_data[datetime_col_m] >= cr_box_time - time_window)
        & (morning_burn_data[datetime_col_m] <= cr_box_time + time_window)
    ].copy()

    if bedroom_window.empty or morning_window.empty:
        return None

    # Check if PM size column exists
    if pm_size not in bedroom_window.columns or pm_size not in morning_window.columns:
        return None

    # Ensure PM columns are numeric
    bedroom_window[pm_size] = pd.to_numeric(bedroom_window[pm_size], errors="coerce")
    morning_window[pm_size] = pd.to_numeric(morning_window[pm_size], errors="coerce")

    # Get the measurement closest to CR Box activation time
    bedroom_window["time_diff"] = abs(
        (bedroom_window[datetime_col_b] - cr_box_time).dt.total_seconds()
    )
    morning_window["time_diff"] = abs(
        (morning_window[datetime_col_m] - cr_box_time).dt.total_seconds()
    )

    bedroom_closest = bedroom_window.loc[bedroom_window["time_diff"].idxmin()]
    morning_closest = morning_window.loc[morning_window["time_diff"].idxmin()]

    bedroom_conc = bedroom_closest[pm_size]
    morning_conc = morning_closest[pm_size]

    # Check for valid data
    if pd.isna(bedroom_conc) or pd.isna(morning_conc) or morning_conc <= 0:
        return None

    # Calculate ratio (bedroom2/morning room)
    ratio = bedroom_conc / morning_conc

    return ratio


def calculate_average_ratio_and_rsd(
    bedroom_data,
    morning_data,
    burn_id,
    pm_size,
    datetime_col_b,
    datetime_col_m,
    analysis_duration_hours=2,
):
    """
    Calculate average concentration ratio and RSD for the analysis window
    Starting from CR Box on time for specified duration
    """

    # Get burn information
    burn_info = burn_log[burn_log["Burn ID"] == burn_id]

    if burn_info.empty:
        return None, None

    burn_date = burn_info["Date"].iloc[0]
    cr_box_time_str = burn_info["CR Box on"].iloc[0]

    # Check if CR Box was used
    if pd.isna(cr_box_time_str) or cr_box_time_str == "n/a":
        return None, None

    # Create datetime for CR Box activation
    cr_box_time = create_naive_datetime(burn_date, cr_box_time_str)

    if pd.isna(cr_box_time):
        return None, None

    # Define analysis window
    start_time = cr_box_time
    end_time = cr_box_time + pd.Timedelta(hours=analysis_duration_hours)

    # Filter data for the current burn date first
    burn_date_only = pd.to_datetime(burn_date).date()

    # Filter bedroom data
    bedroom_burn_data = bedroom_data[bedroom_data["Date"] == burn_date_only].copy()
    morning_burn_data = morning_data[morning_data["Date"] == burn_date_only].copy()

    if bedroom_burn_data.empty or morning_burn_data.empty:
        return None, None

    # Now filter for the analysis window
    bedroom_window = bedroom_burn_data[
        (bedroom_burn_data[datetime_col_b] >= start_time)
        & (bedroom_burn_data[datetime_col_b] <= end_time)
    ].copy()

    morning_window = morning_burn_data[
        (morning_burn_data[datetime_col_m] >= start_time)
        & (morning_burn_data[datetime_col_m] <= end_time)
    ].copy()

    if bedroom_window.empty or morning_window.empty:
        return None, None

    # Check if PM size column exists and is numeric
    if pm_size not in bedroom_window.columns or pm_size not in morning_window.columns:
        return None, None

    # Ensure PM columns are numeric
    bedroom_window[pm_size] = pd.to_numeric(bedroom_window[pm_size], errors="coerce")
    morning_window[pm_size] = pd.to_numeric(morning_window[pm_size], errors="coerce")

    # Remove NaN values
    bedroom_window = bedroom_window.dropna(subset=[pm_size])
    morning_window = morning_window.dropna(subset=[pm_size])

    if bedroom_window.empty or morning_window.empty:
        return None, None

    # Align data by time (resample to 1-minute intervals)
    # Select only the datetime column and PM size column for resampling
    bedroom_resample = bedroom_window[[datetime_col_b, pm_size]].copy()
    morning_resample = morning_window[[datetime_col_m, pm_size]].copy()

    # Set index and resample
    bedroom_resample = (
        bedroom_resample.set_index(datetime_col_b)[pm_size].resample("1T").mean()
    )
    morning_resample = (
        morning_resample.set_index(datetime_col_m)[pm_size].resample("1T").mean()
    )

    # Convert back to DataFrame for merging
    bedroom_resample = pd.DataFrame({f"{pm_size}_bedroom": bedroom_resample})
    morning_resample = pd.DataFrame({f"{pm_size}_morning": morning_resample})

    # Merge on time index
    merged = pd.merge(
        bedroom_resample,
        morning_resample,
        left_index=True,
        right_index=True,
        how="inner",
    )

    if merged.empty:
        return None, None

    # Remove rows with NaN or zero/negative values
    merged = merged.dropna()
    merged = merged[
        (merged[f"{pm_size}_morning"] > 0) & (merged[f"{pm_size}_bedroom"] > 0)
    ]

    if merged.empty or len(merged) < 5:  # Need at least 5 data points
        return None, None

    # Calculate ratios for each time point
    ratios = merged[f"{pm_size}_bedroom"] / merged[f"{pm_size}_morning"]

    # Remove outliers (ratios > 10 or < 0.1)
    ratios = ratios[(ratios > 0.1) & (ratios < 10)]

    if ratios.empty:
        return None, None

    # Calculate average ratio and RSD
    avg_ratio = ratios.mean()
    std_ratio = ratios.std()
    rsd = (std_ratio / avg_ratio * 100) if avg_ratio > 0 else None

    return avg_ratio, rsd


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================
def analyze_spatial_variation():
    """Main function to analyze spatial variation for all burns and instruments"""

    print("=" * 80)
    print("WUI SPATIAL VARIATION ANALYSIS")
    print("=" * 80)

    # Load peak concentration data
    peak_data = load_peak_concentrations()
    if peak_data is None:
        print("ERROR: Could not load peak concentration data")
        return None, None

    # Initialize results storage
    results = {"AeroTrak": [], "QuantAQ": []}

    # Process AeroTrak instruments
    print("\n" + "=" * 40)
    print("PROCESSING AEROTRAK DATA")
    print("=" * 40)

    try:
        aerotrakb_data = process_aerotrak_data(
            INSTRUMENT_CONFIG["AeroTrakB"]["file_path"], "AeroTrakB"
        )
        aerotrakk_data = process_aerotrak_data(
            INSTRUMENT_CONFIG["AeroTrakK"]["file_path"], "AeroTrakK"
        )
        print(f"AeroTrakB data loaded: {aerotrakb_data.shape}")
        print(f"AeroTrakK data loaded: {aerotrakk_data.shape}")

        # Check for required columns
        aerotrakb_pm_cols = [
            col for col in aerotrakb_data.columns if "PM" in col and "(µg/m³)" in col
        ]
        aerotrakk_pm_cols = [
            col for col in aerotrakk_data.columns if "PM" in col and "(µg/m³)" in col
        ]
        print(f"AeroTrakB PM columns: {aerotrakb_pm_cols}")
        print(f"AeroTrakK PM columns: {aerotrakk_pm_cols}")

    except (FileNotFoundError, KeyError, ValueError, pd.errors.ParserError) as e:
        print(f"Error loading AeroTrak data: {e}")
        traceback.print_exc()
        aerotrakb_data = None
        aerotrakk_data = None

    # Process QuantAQ instruments
    print("\n" + "=" * 40)
    print("PROCESSING QUANTAQ DATA")
    print("=" * 40)

    try:
        quantaqb_data = process_quantaq_data(
            INSTRUMENT_CONFIG["QuantAQB"]["file_path"], "QuantAQB"
        )
        quantaqk_data = process_quantaq_data(
            INSTRUMENT_CONFIG["QuantAQK"]["file_path"], "QuantAQK"
        )
        print(f"QuantAQB data loaded: {quantaqb_data.shape}")
        print(f"QuantAQK data loaded: {quantaqk_data.shape}")
    except (FileNotFoundError, KeyError, ValueError, pd.errors.ParserError) as e:
        print(f"Error loading QuantAQ data: {e}")
        traceback.print_exc()
        quantaqb_data = None
        quantaqk_data = None

    # Define PM sizes for each instrument type
    aerotrak_pm_sizes = [
        "PM0.5 (µg/m³)",
        "PM1 (µg/m³)",
        "PM3 (µg/m³)",
        "PM5 (µg/m³)",
        "PM10 (µg/m³)",
        "PM25 (µg/m³)",
    ]
    quantaq_pm_sizes = ["PM1 (µg/m³)", "PM2.5 (µg/m³)", "PM10 (µg/m³)"]

    # Analyze each burn
    for burn_id in burn_log["Burn ID"]:

        print(f"\n{'='*60}")
        print(f"Analyzing {burn_id}")
        print(f"{'='*60}")

        # Check if CR Box was used
        burn_info = burn_log[burn_log["Burn ID"] == burn_id]
        cr_box_time = burn_info["CR Box on"].iloc[0]

        if pd.isna(cr_box_time) or cr_box_time == "n/a":
            print(f"  CR Box not used for {burn_id}, skipping")
            continue

        # Process AeroTrak data for this burn
        if aerotrakb_data is not None and aerotrakk_data is not None:
            print("\n  Processing AeroTrak...")

            for pm_size in aerotrak_pm_sizes:
                try:
                    # Calculate peak ratio
                    peak_ratio = calculate_peak_ratio(
                        peak_data, burn_id, "AeroTrak", pm_size
                    )

                    # Calculate CR Box activation ratio
                    crbox_ratio = calculate_crbox_activation_ratio(
                        aerotrakb_data,
                        aerotrakk_data,
                        burn_id,
                        pm_size,
                        "Date and Time",
                        "Date and Time",
                    )

                    # Calculate average ratio and RSD
                    avg_ratio, rsd = calculate_average_ratio_and_rsd(
                        aerotrakb_data,
                        aerotrakk_data,
                        burn_id,
                        pm_size,
                        "Date and Time",
                        "Date and Time",
                    )

                    # Store results if we have at least one metric
                    if (
                        peak_ratio is not None
                        or crbox_ratio is not None
                        or avg_ratio is not None
                    ):
                        results["AeroTrak"].append(
                            {
                                "Burn_ID": burn_id,
                                "PM_Size": pm_size,
                                "Peak_Ratio_Index": peak_ratio,
                                "CRBox_Activation_Ratio": crbox_ratio,
                                "Average_Ratio": avg_ratio,
                                "RSD_%": rsd,
                            }
                        )

                        # Fixed f-string formatting
                        peak_str = (
                            f"{peak_ratio:.3f}" if peak_ratio is not None else "N/A"
                        )
                        crbox_str = (
                            f"{crbox_ratio:.3f}" if crbox_ratio is not None else "N/A"
                        )
                        avg_str = f"{avg_ratio:.3f}" if avg_ratio is not None else "N/A"
                        rsd_str = f"{rsd:.1f}" if rsd is not None else "N/A"

                        print(
                            f"    {pm_size}: R_I={peak_str}, R_CR={crbox_str}, "
                            f"R_ave={avg_str}, RSD={rsd_str}%"
                        )
                    else:
                        print(f"    {pm_size}: No valid data for ratios")

                except (KeyError, ValueError, TypeError, IndexError) as e:
                    print(f"    Error processing {pm_size}: {str(e)[:100]}")

        # Process QuantAQ data for this burn (only for burns 4-10)
        burn_num = int(burn_id.replace("burn", ""))
        if burn_num >= 4 and quantaqb_data is not None and quantaqk_data is not None:
            print("\n  Processing QuantAQ...")

            for pm_size in quantaq_pm_sizes:
                try:
                    # Calculate peak ratio
                    peak_ratio = calculate_peak_ratio(
                        peak_data, burn_id, "QuantAQ", pm_size
                    )

                    # Calculate CR Box activation ratio
                    crbox_ratio = calculate_crbox_activation_ratio(
                        quantaqb_data,
                        quantaqk_data,
                        burn_id,
                        pm_size,
                        "timestamp_local",
                        "timestamp_local",
                    )

                    # Calculate average ratio and RSD
                    avg_ratio, rsd = calculate_average_ratio_and_rsd(
                        quantaqb_data,
                        quantaqk_data,
                        burn_id,
                        pm_size,
                        "timestamp_local",
                        "timestamp_local",
                    )

                    # Store results if we have at least one metric
                    if (
                        peak_ratio is not None
                        or crbox_ratio is not None
                        or avg_ratio is not None
                    ):
                        results["QuantAQ"].append(
                            {
                                "Burn_ID": burn_id,
                                "PM_Size": pm_size,
                                "Peak_Ratio_Index": peak_ratio,
                                "CRBox_Activation_Ratio": crbox_ratio,
                                "Average_Ratio": avg_ratio,
                                "RSD_%": rsd,
                            }
                        )

                        # Fixed f-string formatting
                        peak_str = (
                            f"{peak_ratio:.3f}" if peak_ratio is not None else "N/A"
                        )
                        crbox_str = (
                            f"{crbox_ratio:.3f}" if crbox_ratio is not None else "N/A"
                        )
                        avg_str = f"{avg_ratio:.3f}" if avg_ratio is not None else "N/A"
                        rsd_str = f"{rsd:.1f}" if rsd is not None else "N/A"

                        print(
                            f"    {pm_size}: R_I={peak_str}, R_CR={crbox_str}, "
                            f"R_ave={avg_str}, RSD={rsd_str}%"
                        )
                    else:
                        print(f"    {pm_size}: No valid data for ratios")

                except (KeyError, ValueError, TypeError, IndexError) as e:
                    print(f"    Error processing {pm_size}: {str(e)[:100]}")

    # Create DataFrames from results
    aerotrak_df = pd.DataFrame(results["AeroTrak"])
    quantaq_df = pd.DataFrame(results["QuantAQ"])

    # Save to Excel file
    output_file = os.path.join(OUTPUT_PATH, "spatial_variation_analysis.xlsx")

    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")

    # Check if there's any data to save
    if aerotrak_df.empty and quantaq_df.empty:
        print(
            "WARNING: No results to save - both AeroTrak and QuantAQ DataFrames are empty"
        )
        print("Check the error messages above for issues with data processing")
    else:
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            if not aerotrak_df.empty:
                aerotrak_df.to_excel(writer, sheet_name="AeroTrak", index=False)
                print(f"AeroTrak results: {len(aerotrak_df)} rows")

            if not quantaq_df.empty:
                quantaq_df.to_excel(writer, sheet_name="QuantAQ", index=False)
                print(f"QuantAQ results: {len(quantaq_df)} rows")

        print(f"\nResults saved to: {output_file}")

    # Print summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")

    if not aerotrak_df.empty:
        print("\nAeroTrak Summary:")
        print(f"  Burns analyzed: {aerotrak_df['Burn_ID'].nunique()}")
        print(f"  Total measurements: {len(aerotrak_df)}")

        # Calculate means only for non-null values
        peak_mean = aerotrak_df["Peak_Ratio_Index"].dropna().mean()
        crbox_mean = aerotrak_df["CRBox_Activation_Ratio"].dropna().mean()
        avg_mean = aerotrak_df["Average_Ratio"].dropna().mean()
        rsd_mean = aerotrak_df["RSD_%"].dropna().mean()

        # Fixed f-string formatting
        print(
            f"  Mean Peak Ratio: {peak_mean:.3f}"
            if not pd.isna(peak_mean)
            else "  Mean Peak Ratio: N/A"
        )
        print(
            f"  Mean CR Box Activation Ratio: {crbox_mean:.3f}"
            if not pd.isna(crbox_mean)
            else "  Mean CR Box Activation Ratio: N/A"
        )
        print(
            f"  Mean Average Ratio: {avg_mean:.3f}"
            if not pd.isna(avg_mean)
            else "  Mean Average Ratio: N/A"
        )
        print(
            f"  Mean RSD: {rsd_mean:.1f}%"
            if not pd.isna(rsd_mean)
            else "  Mean RSD: N/A"
        )

    if not quantaq_df.empty:
        print("\nQuantAQ Summary:")
        print(f"  Burns analyzed: {quantaq_df['Burn_ID'].nunique()}")
        print(f"  Total measurements: {len(quantaq_df)}")

        # Calculate means only for non-null values
        peak_mean = quantaq_df["Peak_Ratio_Index"].dropna().mean()
        crbox_mean = quantaq_df["CRBox_Activation_Ratio"].dropna().mean()
        avg_mean = quantaq_df["Average_Ratio"].dropna().mean()
        rsd_mean = quantaq_df["RSD_%"].dropna().mean()

        # Fixed f-string formatting
        print(
            f"  Mean Peak Ratio: {peak_mean:.3f}"
            if not pd.isna(peak_mean)
            else "  Mean Peak Ratio: N/A"
        )
        print(
            f"  Mean CR Box Activation Ratio: {crbox_mean:.3f}"
            if not pd.isna(crbox_mean)
            else "  Mean CR Box Activation Ratio: N/A"
        )
        print(
            f"  Mean Average Ratio: {avg_mean:.3f}"
            if not pd.isna(avg_mean)
            else "  Mean Average Ratio: N/A"
        )
        print(
            f"  Mean RSD: {rsd_mean:.1f}%"
            if not pd.isna(rsd_mean)
            else "  Mean RSD: N/A"
        )

    return aerotrak_df, quantaq_df


# ============================================================================
# SCRIPT EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Run the analysis
    aerotrak_results, quantaq_results = analyze_spatial_variation()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
