#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WUI Spatial Variation Analysis Time-Series Plotting Script
============================================================

This script generates interactive Bokeh visualizations of spatial variation analysis
results over time from wildfire smoke infiltration experiments. It creates plots showing
how concentration ratios between two locations (bedroom2 vs morning room) evolve over time.

Key Features:
    - X-axis: Hours since garage door closed (-1 to 5 hours)
    - 30-minute average ratios with uncertainty (standard error) indicators
    - Compares data from two instrument types: OPC (AeroTrak) and Nef+OPC (QuantAQ)
    - Analyzes all burns with data from both instrument pairs
    - 30-minute average ratios calculated for fixed 30-minute bins
    - Error bars show standard error of the mean for each bin
    - Small x-axis jitter added to prevent overlapping points
    - Solid markers for Nef+OPC, hollow markers for OPC
    - Color-coded by burn (ordered by number of CR boxes)
    - Shaded region indicates smoke injection period (-30 min to 0 hour)
    - No curve fitting applied

Input:
    - Time-series data from AeroTrak and QuantAQ instruments
    - Burn log with garage door closed times

Output:
    - Individual HTML files for each PM size with interactive Bokeh plots

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2026-01-12
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.layouts import column
from bokeh.models import BoxAnnotation, Div, Label, Span
from bokeh.plotting import figure, output_file, save

# ============================================================================
# SYSTEM DETECTION AND PATH SETUP
# ============================================================================

# Use portable data paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from scripts import get_script_metadata  # noqa: E402
from scripts.data_loaders import load_burn_log  # noqa: E402
from scripts.datetime_utils import TIME_SHIFTS, create_naive_datetime  # noqa: E402
from scripts.instrument_config import (  # noqa: E402
    get_baseline_values,
    get_instrument_datetime_column,
    get_instrument_location,
)
from scripts.plotting_utils import apply_text_formatting  # noqa: E402
from src.data_paths import get_common_file, get_data_root  # noqa: E402

# Get portable paths
BASE_DIR = str(get_data_root())
print(f"[OK] Using data directory: {BASE_DIR}")

# Load burn log
burn_log_path = get_common_file("burn_log")
burn_log = load_burn_log(burn_log_path)
print(f"[OK] Burn log loaded from: {burn_log_path}")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Output directory for figures
OUTPUT_DIR = str(get_common_file("output_figures"))

# Burns to exclude (user can modify this list)
EXCLUDED_BURNS = [
    "burn2",
    "burn3",
    "burn4",
    "burn5",
    "burn6",
]  # Example: ["burn1", "burn5", "burn6"]

# PM sizes to create plots for
PM_SIZES = ["PM1 (µg/m³)", "PM2.5 (µg/m³)", "PM10 (µg/m³)"]

# AeroTrak PM3 is used as a proxy for PM2.5 comparison
AEROTRAK_PM_SIZE_FOR_PM25 = "PM3 (µg/m³)"

# Ratio metrics
RATIO_TYPES = ["Peak", "CR_Box", "Hourly_Avg"]

# Marker shapes for ratio types
RATIO_SHAPES = {
    "Peak": "triangle",
    "CR_Box": "square",
    "Hourly_Avg": "circle",
}

# PAC (Portable Air Cleaner) labels for each burn
PAC_LABELS = {
    "burn1": "0",
    "burn2": "4N",
    "burn3": "1U",
    "burn4": "1N",
    "burn5": "0",
    "burn6": "1N",
    "burn7": "2AN",
    "burn8": "2AU",
    "burn9": "2N",
    "burn10": "2U",
}

# Time range for x-axis
TIME_RANGE = (-1, 6)  # Hours since garage door closed

# 30-minute bins for average ratios (fixed 30-minute windows)
HOURLY_BINS = [
    (-1, -0.5),
    (-0.5, 0),
    (0, 0.5),
    (0.5, 1),
    (1, 1.5),
    (1.5, 2),
    (2, 2.5),
    (2.5, 3),
    (3, 3.5),
    (3.5, 4),
    (4, 4.5),
    (4.5, 5),
]

# Plot configuration
PLOT_Y_RANGE = (0, 2.0)  # Y-axis range for ratio plots

# ============================================================================
# INSTRUMENT DATA LOADING
# ============================================================================


def load_instrument_data():
    """
    Load time-series data from all four instruments

    Returns
    -------
    dict
        Dictionary with instrument names as keys and DataFrames as values
    """
    from src.spatial_variation_analysis import (
        INSTRUMENT_CONFIG,
        process_aerotrak_data,
        process_quantaq_data,
    )

    print("\n" + "=" * 60)
    print("LOADING INSTRUMENT DATA")
    print("=" * 60)

    instruments = {}

    # Load AeroTrak bedroom2 (OPC)
    try:
        print("Loading AeroTrakB (bedroom2)...")
        instruments["AeroTrakB"] = process_aerotrak_data(
            INSTRUMENT_CONFIG["AeroTrakB"]["file_path"], "AeroTrakB"
        )
        print(f"  Loaded {len(instruments['AeroTrakB'])} rows")
    except Exception as e:
        print(f"  ERROR: {e}")
        instruments["AeroTrakB"] = None

    # Load AeroTrak kitchen/morning room (OPC)
    try:
        print("Loading AeroTrakK (morning room)...")
        instruments["AeroTrakK"] = process_aerotrak_data(
            INSTRUMENT_CONFIG["AeroTrakK"]["file_path"], "AeroTrakK"
        )
        print(f"  Loaded {len(instruments['AeroTrakK'])} rows")
    except Exception as e:
        print(f"  ERROR: {e}")
        instruments["AeroTrakK"] = None

    # Load QuantAQ bedroom2 (Nef+OPC)
    try:
        print("Loading QuantAQB (bedroom2)...")
        instruments["QuantAQB"] = process_quantaq_data(
            INSTRUMENT_CONFIG["QuantAQB"]["file_path"], "QuantAQB"
        )
        print(f"  Loaded {len(instruments['QuantAQB'])} rows")
    except Exception as e:
        print(f"  ERROR: {e}")
        instruments["QuantAQB"] = None

    # Load QuantAQ kitchen/morning room (Nef+OPC)
    try:
        print("Loading QuantAQK (morning room)...")
        instruments["QuantAQK"] = process_quantaq_data(
            INSTRUMENT_CONFIG["QuantAQK"]["file_path"], "QuantAQK"
        )
        print(f"  Loaded {len(instruments['QuantAQK'])} rows")
    except Exception as e:
        print(f"  ERROR: {e}")
        instruments["QuantAQK"] = None

    return instruments


def load_peak_times():
    """
    Load peak times from peak concentration file

    Returns
    -------
    pd.DataFrame or None
        DataFrame with peak times for all instruments and burns
    """
    peak_file = os.path.join(
        BASE_DIR, "burn_data", "peak_concentrations_all_instruments_edited.xlsx"
    )

    try:
        peak_data = pd.read_excel(peak_file, sheet_name="data")
        print(f"\n[OK] Peak data loaded: {peak_data.shape}")
        return peak_data
    except Exception as e:
        print(f"\n[ERROR] Could not load peak data: {e}")
        return None


# ============================================================================
# RATIO CALCULATION FUNCTIONS
# ============================================================================


def get_burns_with_complete_data(instruments, burn_log):
    """
    Identify burns that have data from at least one instrument pair (OPC or Nef+OPC)

    Parameters
    ----------
    instruments : dict
        Dictionary of instrument DataFrames
    burn_log : pd.DataFrame
        Burn log with burn information

    Returns
    -------
    list
        List of burn IDs with at least one instrument pair, ordered by PAC configuration
    """
    valid_burns = []

    for _, row in burn_log.iterrows():
        burn_id = row["Burn ID"]

        # Skip excluded burns
        if burn_id in EXCLUDED_BURNS:
            continue

        # Check if CR Box was used (skip burn5 and burn1 which have 0 PACs)
        if pd.isna(row["CR Box on"]) or row["CR Box on"] == "n/a":
            continue

        burn_date = pd.to_datetime(row["Date"]).date()

        # Check if at least one instrument pair has data
        has_opc_pair = False
        has_nef_pair = False

        # Check OPC pair (AeroTrak)
        if (
            instruments["AeroTrakB"] is not None
            and instruments["AeroTrakK"] is not None
        ):
            aerotrak_b_data = instruments["AeroTrakB"][
                instruments["AeroTrakB"]["Date"] == burn_date
            ]
            aerotrak_k_data = instruments["AeroTrakK"][
                instruments["AeroTrakK"]["Date"] == burn_date
            ]
            if not aerotrak_b_data.empty and not aerotrak_k_data.empty:
                has_opc_pair = True

        # Check Nef+OPC pair (QuantAQ)
        if instruments["QuantAQB"] is not None and instruments["QuantAQK"] is not None:
            quantaq_b_data = instruments["QuantAQB"][
                instruments["QuantAQB"]["Date"] == burn_date
            ]
            quantaq_k_data = instruments["QuantAQK"][
                instruments["QuantAQK"]["Date"] == burn_date
            ]
            if not quantaq_b_data.empty and not quantaq_k_data.empty:
                has_nef_pair = True

        # Include burn if at least one pair has data
        if has_opc_pair or has_nef_pair:
            # Get PAC label for ordering
            pac_label = PAC_LABELS.get(burn_id, burn_id)
            valid_burns.append((burn_id, pac_label, has_opc_pair, has_nef_pair))

    # Sort by PAC label to maintain consistent ordering
    valid_burns.sort(key=lambda x: x[1])

    return [(burn_id, has_opc, has_nef) for burn_id, _, has_opc, has_nef in valid_burns]


def calculate_peak_ratio_with_time(
    bedroom_data,
    morning_data,
    burn_id,
    pm_size,
    datetime_col_b,
    datetime_col_m,
    instrument_type,
):
    """
    Calculate peak ratio and mean peak time for a burn

    Parameters
    ----------
    bedroom_data : pd.DataFrame
        Bedroom concentration data
    morning_data : pd.DataFrame
        Morning room concentration data
    burn_id : str
        Burn identifier
    pm_size : str
        PM size column name
    datetime_col_b : str
        Datetime column name for bedroom data
    datetime_col_m : str
        Datetime column name for morning room data
    instrument_type : str
        Either "OPC" or "Nef+OPC"

    Returns
    -------
    tuple
        (ratio, time_hours) or (None, None) if calculation fails
    """
    # Get burn date
    burn_info = burn_log[burn_log["Burn ID"] == burn_id]
    if burn_info.empty:
        return None, None

    burn_date = pd.to_datetime(burn_info["Date"].iloc[0]).date()
    garage_closed_str = burn_info["garage closed"].iloc[0]

    if pd.isna(garage_closed_str) or garage_closed_str == "n/a":
        return None, None

    garage_closed_time = create_naive_datetime(
        burn_info["Date"].iloc[0], garage_closed_str
    )

    # Filter for burn date
    bedroom_burn = bedroom_data[bedroom_data["Date"] == burn_date].copy()
    morning_burn = morning_data[morning_data["Date"] == burn_date].copy()

    if bedroom_burn.empty or morning_burn.empty:
        return None, None

    # Check if PM size exists
    if pm_size not in bedroom_burn.columns or pm_size not in morning_burn.columns:
        return None, None

    # Ensure numeric
    bedroom_burn[pm_size] = pd.to_numeric(bedroom_burn[pm_size], errors="coerce")
    morning_burn[pm_size] = pd.to_numeric(morning_burn[pm_size], errors="coerce")

    # Find peak concentrations
    bedroom_peak_idx = bedroom_burn[pm_size].idxmax()
    morning_peak_idx = morning_burn[pm_size].idxmax()

    if pd.isna(bedroom_peak_idx) or pd.isna(morning_peak_idx):
        return None, None

    bedroom_peak_conc = bedroom_burn.loc[bedroom_peak_idx, pm_size]
    morning_peak_conc = morning_burn.loc[morning_peak_idx, pm_size]

    if (
        pd.isna(bedroom_peak_conc)
        or pd.isna(morning_peak_conc)
        or morning_peak_conc <= 0
    ):
        return None, None

    # Calculate ratio
    ratio = bedroom_peak_conc / morning_peak_conc

    # Calculate mean peak time
    bedroom_peak_time = bedroom_burn.loc[bedroom_peak_idx, datetime_col_b]
    morning_peak_time = morning_burn.loc[morning_peak_idx, datetime_col_m]

    if pd.isna(bedroom_peak_time) or pd.isna(morning_peak_time):
        return None, None

    # Mean time in hours since garage closed
    bedroom_time_diff = (
        pd.Timestamp(bedroom_peak_time) - pd.Timestamp(garage_closed_time)
    ).total_seconds() / 3600
    morning_time_diff = (
        pd.Timestamp(morning_peak_time) - pd.Timestamp(garage_closed_time)
    ).total_seconds() / 3600
    mean_time_hours = (bedroom_time_diff + morning_time_diff) / 2

    return ratio, mean_time_hours


def calculate_crbox_ratio_with_time(
    bedroom_data, morning_data, burn_id, pm_size, datetime_col_b, datetime_col_m
):
    """
    Calculate CR Box activation ratio and time

    Returns
    -------
    tuple
        (ratio, time_hours) or (None, None)
    """
    burn_info = burn_log[burn_log["Burn ID"] == burn_id]
    if burn_info.empty:
        return None, None

    burn_date = burn_info["Date"].iloc[0]
    cr_box_time_str = burn_info["CR Box on"].iloc[0]
    garage_closed_str = burn_info["garage closed"].iloc[0]

    if pd.isna(cr_box_time_str) or cr_box_time_str == "n/a":
        return None, None
    if pd.isna(garage_closed_str) or garage_closed_str == "n/a":
        return None, None

    cr_box_time = create_naive_datetime(burn_date, cr_box_time_str)
    garage_closed_time = create_naive_datetime(burn_date, garage_closed_str)

    if pd.isna(cr_box_time) or pd.isna(garage_closed_time):
        return None, None

    # Calculate time in hours since garage closed
    time_hours = (
        pd.Timestamp(cr_box_time) - pd.Timestamp(garage_closed_time)
    ).total_seconds() / 3600

    # Filter data
    burn_date_only = pd.to_datetime(burn_date).date()
    bedroom_burn = bedroom_data[bedroom_data["Date"] == burn_date_only].copy()
    morning_burn = morning_data[morning_data["Date"] == burn_date_only].copy()

    if bedroom_burn.empty or morning_burn.empty:
        return None, None

    # Find closest measurement to CR Box time (within ±5 minutes)
    time_window = pd.Timedelta(minutes=5)

    bedroom_window = bedroom_burn[
        (bedroom_burn[datetime_col_b] >= cr_box_time - time_window)
        & (bedroom_burn[datetime_col_b] <= cr_box_time + time_window)
    ].copy()

    morning_window = morning_burn[
        (morning_burn[datetime_col_m] >= cr_box_time - time_window)
        & (morning_burn[datetime_col_m] <= cr_box_time + time_window)
    ].copy()

    if bedroom_window.empty or morning_window.empty:
        return None, None

    if pm_size not in bedroom_window.columns or pm_size not in morning_window.columns:
        return None, None

    # Ensure numeric
    bedroom_window[pm_size] = pd.to_numeric(bedroom_window[pm_size], errors="coerce")
    morning_window[pm_size] = pd.to_numeric(morning_window[pm_size], errors="coerce")

    # Get closest measurement
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

    if pd.isna(bedroom_conc) or pd.isna(morning_conc) or morning_conc <= 0:
        return None, None

    ratio = bedroom_conc / morning_conc

    return ratio, time_hours


def calculate_hourly_average_ratios(
    bedroom_data, morning_data, burn_id, pm_size, datetime_col_b, datetime_col_m
):
    """
    Calculate average ratios for fixed 30-minute bins with uncertainty

    Returns
    -------
    list of tuples
        List of (ratio, time_hours, std_error) for each 30-minute bin
    """
    burn_info = burn_log[burn_log["Burn ID"] == burn_id]
    if burn_info.empty:
        return []

    burn_date = burn_info["Date"].iloc[0]
    garage_closed_str = burn_info["garage closed"].iloc[0]

    if pd.isna(garage_closed_str) or garage_closed_str == "n/a":
        return []

    garage_closed_time = create_naive_datetime(burn_date, garage_closed_str)
    if pd.isna(garage_closed_time):
        return []

    # Filter for burn date
    burn_date_only = pd.to_datetime(burn_date).date()
    bedroom_burn = bedroom_data[bedroom_data["Date"] == burn_date_only].copy()
    morning_burn = morning_data[morning_data["Date"] == burn_date_only].copy()

    if bedroom_burn.empty or morning_burn.empty:
        return []

    if pm_size not in bedroom_burn.columns or pm_size not in morning_burn.columns:
        return []

    # Ensure numeric
    bedroom_burn[pm_size] = pd.to_numeric(bedroom_burn[pm_size], errors="coerce")
    morning_burn[pm_size] = pd.to_numeric(morning_burn[pm_size], errors="coerce")

    results = []

    for start_hour, end_hour in HOURLY_BINS:
        # Define time window
        start_time = garage_closed_time + pd.Timedelta(hours=start_hour)
        end_time = garage_closed_time + pd.Timedelta(hours=end_hour)

        # Filter data for this window
        bedroom_window = bedroom_burn[
            (bedroom_burn[datetime_col_b] >= start_time)
            & (bedroom_burn[datetime_col_b] < end_time)
        ].copy()

        morning_window = morning_burn[
            (morning_burn[datetime_col_m] >= start_time)
            & (morning_burn[datetime_col_m] < end_time)
        ].copy()

        if bedroom_window.empty or morning_window.empty:
            continue

        # Remove NaN values
        bedroom_window = bedroom_window.dropna(subset=[pm_size])
        morning_window = morning_window.dropna(subset=[pm_size])

        if bedroom_window.empty or morning_window.empty:
            continue

        # Resample to 1-minute intervals and align
        bedroom_resample = bedroom_window[[datetime_col_b, pm_size]].copy()
        morning_resample = morning_window[[datetime_col_m, pm_size]].copy()

        bedroom_resample = (
            bedroom_resample.set_index(datetime_col_b)[pm_size].resample("1T").mean()
        )
        morning_resample = (
            morning_resample.set_index(datetime_col_m)[pm_size].resample("1T").mean()
        )

        # Convert to DataFrames for merging
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
            continue

        # Remove NaN or zero/negative values
        merged = merged.dropna()
        merged = merged[
            (merged[f"{pm_size}_morning"] > 0) & (merged[f"{pm_size}_bedroom"] > 0)
        ]

        if merged.empty or len(merged) < 3:  # Need at least 3 data points
            continue

        # Calculate ratios for each time point
        ratios = merged[f"{pm_size}_bedroom"] / merged[f"{pm_size}_morning"]

        # Remove outliers (ratios > 10 or < 0.1)
        ratios = ratios[(ratios > 0.1) & (ratios < 10)]

        if ratios.empty:
            continue

        # Calculate average ratio and standard error
        avg_ratio = ratios.mean()
        std_error = ratios.std() / np.sqrt(len(ratios))  # Standard error of the mean

        # Use midpoint of 30-minute bin for plotting
        time_hours = (start_hour + end_hour) / 2

        results.append((avg_ratio, time_hours, std_error))

    return results


# ============================================================================
# PLOTTING FUNCTION
# ============================================================================


def create_timeseries_plot(instruments, valid_burns, pm_size):
    """
    Create time-series plot for spatial variation ratios

    Parameters
    ----------
    instruments : dict
        Dictionary of instrument DataFrames
    valid_burns : list
        List of tuples (burn_id, has_opc, has_nef) with available data
    pm_size : str
        PM size label (e.g., 'PM2.5 (µg/m³)')

    Returns
    -------
    bokeh.plotting.figure.Figure
        Bokeh figure
    """
    # Create figure
    p = figure(
        x_axis_label="Time Since Garage Door Closed (hours)",
        y_axis_label="Concentration Ratio (Bedroom2 / Morning Room)",
        x_range=(TIME_RANGE[0], TIME_RANGE[1]),
        y_range=(PLOT_Y_RANGE[0], PLOT_Y_RANGE[1]),
        width=900,
        height=600,
    )

    # Apply standard text formatting
    apply_text_formatting(p)

    # Add shaded region for smoke injection period (-30 min to 0 hour)
    smoke_injection_box = BoxAnnotation(
        left=-0.5,  # -30 minutes = -0.5 hours
        right=0,
        fill_alpha=0.15,
        fill_color="orange",
        line_width=0,
    )
    p.add_layout(smoke_injection_box)

    # Add label for smoke injection period
    smoke_label = Label(
        x=-0.25,  # Center of the shaded region
        y=PLOT_Y_RANGE[1] * 0.95,  # Near top of plot
        text="Smoke Injection",
        text_align="center",
        text_baseline="top",
        text_font_size="9pt",
        text_color="darkorange",
        text_font_style="italic",
    )
    p.add_layout(smoke_label)

    # Use Plasma color palette
    from bokeh.palettes import Plasma

    # Generate color list for burns using Plasma palette
    n_burns = len(valid_burns)

    # Plasma palettes are available in sizes 3-256
    if n_burns <= 3:
        colors = Plasma[3][:n_burns]
    elif n_burns <= 256:
        colors = Plasma[n_burns]
    else:
        # If more than 256 burns, repeat the 256-color palette
        colors = (Plasma[256] * ((n_burns // 256) + 1))[:n_burns]

    # Create color mapping using burn_id only
    burn_ids = [burn_id for burn_id, _, _ in valid_burns]
    burn_colors = dict(zip(burn_ids, colors))

    # Get appropriate PM size for AeroTrak
    aerotrak_pm_size = (
        AEROTRAK_PM_SIZE_FOR_PM25 if pm_size == "PM2.5 (µg/m³)" else pm_size
    )

    # Track which PAC labels we've added to legend (only show circle markers)
    legend_added = set()

    # Process each burn
    for burn_id, has_opc_pair, has_nef_pair in valid_burns:
        burn_color = burn_colors[burn_id]

        # Get PAC label for legend
        pac_label = PAC_LABELS.get(burn_id, "?")

        # Only process OPC data if this burn has the OPC pair
        if has_opc_pair:
            # Calculate 30-minute average ratios for OPC (AeroTrak)
            hourly_ratios_opc = calculate_hourly_average_ratios(
                instruments["AeroTrakB"],
                instruments["AeroTrakK"],
                burn_id,
                aerotrak_pm_size,
                "Date and Time",
                "Date and Time",
            )

            # Plot OPC markers (hollow) with error bars
            # 30-minute average (circle) - ADD TO LEGEND only if not already added for this PAC
            if hourly_ratios_opc:
                # Extract data with proper unpacking (ratio, time, std_error)
                ratios_opc = [r for r, _, _ in hourly_ratios_opc]
                times_opc = [t for _, t, _ in hourly_ratios_opc]
                errors_opc = [e for _, _, e in hourly_ratios_opc]

                # Add small random jitter to x-axis to avoid overlapping points
                np.random.seed(hash(burn_id) % 2**32)  # Reproducible jitter per burn
                jitter = np.random.uniform(-0.05, 0.05, len(times_opc))
                times_jittered_opc = [t + j for t, j in zip(times_opc, jitter)]

                # Create legend key for OPC
                legend_key_opc = f"{pac_label} PAC - OPC"

                if legend_key_opc not in legend_added:
                    # Plot with error bars
                    p.scatter(
                        times_jittered_opc,
                        ratios_opc,
                        marker=RATIO_SHAPES["Hourly_Avg"],
                        size=10,
                        color=burn_color,
                        fill_alpha=0,  # Hollow
                        line_width=2,
                        legend_label=legend_key_opc,
                    )
                    # Add error bars (whiskers)
                    for x, y, err in zip(times_jittered_opc, ratios_opc, errors_opc):
                        p.line(
                            [x, x],
                            [y - err, y + err],
                            color=burn_color,
                            line_width=1.5,
                            alpha=0.6,
                        )
                    legend_added.add(legend_key_opc)
                else:
                    # Plot with error bars
                    p.scatter(
                        times_jittered_opc,
                        ratios_opc,
                        marker=RATIO_SHAPES["Hourly_Avg"],
                        size=10,
                        color=burn_color,
                        fill_alpha=0,  # Hollow
                        line_width=2,
                    )
                    # Add error bars (whiskers)
                    for x, y, err in zip(times_jittered_opc, ratios_opc, errors_opc):
                        p.line(
                            [x, x],
                            [y - err, y + err],
                            color=burn_color,
                            line_width=1.5,
                            alpha=0.6,
                        )

        # Only process Nef+OPC data if this burn has the Nef+OPC pair
        if has_nef_pair:
            # Calculate 30-minute average ratios for Nef+OPC (QuantAQ)
            hourly_ratios_nef = calculate_hourly_average_ratios(
                instruments["QuantAQB"],
                instruments["QuantAQK"],
                burn_id,
                pm_size,
                "timestamp_local",
                "timestamp_local",
            )

            # Plot Nef+OPC markers (solid) with error bars
            # 30-minute average (circle) - ADD TO LEGEND only if not already added for this PAC
            if hourly_ratios_nef:
                # Extract data with proper unpacking (ratio, time, std_error)
                ratios_nef = [r for r, _, _ in hourly_ratios_nef]
                times_nef = [t for _, t, _ in hourly_ratios_nef]
                errors_nef = [e for _, _, e in hourly_ratios_nef]

                # Add small random jitter to x-axis to avoid overlapping points
                # Use different seed offset for Nef+OPC to ensure different jitter than OPC
                np.random.seed(
                    (hash(burn_id) + 1) % 2**32
                )  # Reproducible jitter per burn
                jitter = np.random.uniform(-0.05, 0.05, len(times_nef))
                times_jittered_nef = [t + j for t, j in zip(times_nef, jitter)]

                # Create legend key for Nef+OPC
                legend_key_nef = f"{pac_label} PAC - Nef+OPC"

                if legend_key_nef not in legend_added:
                    # Plot with error bars
                    p.scatter(
                        times_jittered_nef,
                        ratios_nef,
                        marker=RATIO_SHAPES["Hourly_Avg"],
                        size=10,
                        color=burn_color,
                        fill_alpha=1.0,  # Solid
                        line_width=2,
                        legend_label=legend_key_nef,
                    )
                    # Add error bars (whiskers)
                    for x, y, err in zip(times_jittered_nef, ratios_nef, errors_nef):
                        p.line(
                            [x, x],
                            [y - err, y + err],
                            color=burn_color,
                            line_width=1.5,
                            alpha=0.6,
                        )
                    legend_added.add(legend_key_nef)
                else:
                    # Plot with error bars
                    p.scatter(
                        times_jittered_nef,
                        ratios_nef,
                        marker=RATIO_SHAPES["Hourly_Avg"],
                        size=10,
                        color=burn_color,
                        fill_alpha=1.0,  # Solid
                        line_width=2,
                    )
                    # Add error bars (whiskers)
                    for x, y, err in zip(times_jittered_nef, ratios_nef, errors_nef):
                        p.line(
                            [x, x],
                            [y - err, y + err],
                            color=burn_color,
                            line_width=1.5,
                            alpha=0.6,
                        )

    # Add horizontal line at y=1.0 (perfect spatial uniformity)
    uniform_line = Span(
        location=1.0,
        dimension="width",
        line_color="black",
        line_dash="dotted",
        line_width=1,
        line_alpha=0.7,
    )
    p.add_layout(uniform_line)

    # Add vertical line at x=0 (garage door closed time)
    zero_line = Span(
        location=0,
        dimension="height",
        line_color="black",
        line_width=1,
        line_alpha=0.7,
    )
    p.add_layout(zero_line)

    # Customize legend
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    p.legend.label_text_font_size = "8pt"

    return p


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """
    Main function to generate spatial variation time-series plots
    """
    print("=" * 80)
    print("WUI SPATIAL VARIATION ANALYSIS - TIME-SERIES PLOTTING")
    print("=" * 80)
    print(f"Base directory: {BASE_DIR}")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Get script metadata
    try:
        metadata = get_script_metadata()
        print("[OK] Metadata loaded successfully")
    except Exception as e:
        metadata = "Metadata unavailable"
        print(f"[WARNING] Metadata not available: {e}")

    # Load instrument data
    instruments = load_instrument_data()

    # Check if at least one instrument pair loaded successfully
    has_opc = (
        instruments["AeroTrakB"] is not None and instruments["AeroTrakK"] is not None
    )
    has_nef = (
        instruments["QuantAQB"] is not None and instruments["QuantAQK"] is not None
    )

    if not has_opc and not has_nef:
        print("\n[ERROR] No instrument pairs loaded successfully. Cannot proceed.")
        return

    # Get burns with available data
    print("\n" + "=" * 60)
    print("IDENTIFYING BURNS WITH AVAILABLE DATA")
    print("=" * 60)
    valid_burns = get_burns_with_complete_data(instruments, burn_log)
    print(
        f"Found {len(valid_burns)} burns with data from at least one instrument pair:"
    )
    for burn_id, has_opc_pair, has_nef_pair in valid_burns:
        pac_label = PAC_LABELS.get(burn_id, "?")
        instruments_str = []
        if has_opc_pair:
            instruments_str.append("OPC")
        if has_nef_pair:
            instruments_str.append("Nef+OPC")
        print(f"  {burn_id} ({pac_label} PAC): {', '.join(instruments_str)}")

    if not valid_burns:
        print("\n[ERROR] No burns with complete data found. Cannot proceed.")
        return

    # Generate plots for each PM size
    print(f"\n{'=' * 60}")
    print(f"GENERATING TIME-SERIES PLOTS")
    print(f"{'=' * 60}")

    for pm_size in PM_SIZES:
        print(f"\nProcessing {pm_size}...")

        # Create plot
        p = create_timeseries_plot(instruments, valid_burns, pm_size)

        # Create metadata div
        burn_ids_str = ", ".join([burn_id for burn_id, _, _ in valid_burns])
        div_text = (
            f'<div style="font-size: 10pt; font-weight: normal;">'
            f"<strong>Spatial Variation Analysis - Time Series</strong><br><br>"
            f"PM Size: {pm_size}<br>"
            f"Burns analyzed: {burn_ids_str}<br>"
            f"Excluded burns: {', '.join(EXCLUDED_BURNS) if EXCLUDED_BURNS else 'None'}<br><br>"
            f"<strong>Plot Features:</strong><br>"
            f"&nbsp;&nbsp;• 30-minute average concentration ratios (circle markers)<br>"
            f"&nbsp;&nbsp;• Error bars show standard error of the mean<br>"
            f"&nbsp;&nbsp;• Hollow markers: OPC (AeroTrak)<br>"
            f"&nbsp;&nbsp;• Solid markers: Nef+OPC (QuantAQ)<br>"
            f"&nbsp;&nbsp;• Orange shaded region: Smoke injection period (-30 min to 0 hour)<br>"
            f"&nbsp;&nbsp;• Small random x-axis jitter applied to reduce overlap<br><br>"
            f"<strong>Note:</strong> Click legend entries to hide/show data series.<br><br>"
            f"<hr><br>{metadata}</div>"
        )
        metadata_div = Div(text=div_text, width=900)

        # Combine plot and metadata
        layout = column(p, metadata_div)

        # Save figure
        output_filename = f"spatial_variation_timeseries_{pm_size.replace(' ', '_').replace('/', '_')}.html"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        output_file(output_path)
        save(layout)
        print(f"  Saved plot to: {output_path}")

    print("\n" + "=" * 80)
    print("PLOTTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
