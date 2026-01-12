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
    - Three ratio metrics: Peak Ratio, CR Box Activation Ratio, Hourly Average Ratios
    - Compares data from two instrument types: OPC (AeroTrak) and Nef+OPC (QuantAQ)
    - Analyzes all burns with data from both instrument pairs
    - Peak ratios plotted at mean peak time for each instrument type
    - Hourly average ratios calculated for fixed 1-hour bins
    - Solid markers for Nef+OPC, hollow markers for OPC
    - Different marker shapes for different ratio types
    - Color-coded by burn (ordered by number of CR boxes)
    - No curve fitting applied

Input:
    - Time-series data from AeroTrak and QuantAQ instruments
    - Peak concentration data from peak_concentrations_all_instruments_edited.xlsx
    - Burn log with CR Box activation times

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
from bokeh.models import ColumnDataSource, Div, Span
from bokeh.plotting import figure, output_file, save

# ============================================================================
# SYSTEM DETECTION AND PATH SETUP
# ============================================================================

# Use portable data paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from scripts import get_script_metadata  # noqa: E402
from scripts.plotting_utils import apply_text_formatting  # noqa: E402
from src.data_paths import get_common_file, get_data_root  # noqa: E402
from scripts.datetime_utils import create_naive_datetime, TIME_SHIFTS  # noqa: E402
from scripts.data_loaders import load_burn_log  # noqa: E402
from scripts.instrument_config import (  # noqa: E402
    get_instrument_datetime_column,
    get_baseline_values,
    get_instrument_location,
)

# Get portable paths
BASE_DIR = str(get_data_root())
print(f"[OK] Using data directory: {BASE_DIR}")

# Load burn log
burn_log_path = get_common_file('burn_log')
burn_log = load_burn_log(burn_log_path)
print(f"[OK] Burn log loaded from: {burn_log_path}")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Output directory for figures
OUTPUT_DIR = str(get_common_file("output_figures"))

# Burns to exclude (user can modify this list)
EXCLUDED_BURNS = []  # Example: ["burn1", "burn5", "burn6"]

# PM sizes to create plots for
PM_SIZES = ["PM1 (µg/m³)", "PM2.5 (µg/m³)", "PM10 (µg/m³)"]

# AeroTrak PM3 is used as a proxy for PM2.5 comparison
AEROTRAK_PM_SIZE_FOR_PM25 = "PM3 (µg/m³)"

# Ratio metrics
RATIO_TYPES = ["Peak", "CR_Box", "Hourly_Avg"]

# Marker shapes for ratio types
RATIO_SHAPES = {
    "Peak": "circle",
    "CR_Box": "square",
    "Hourly_Avg": "triangle",
}

# Time range for x-axis
TIME_RANGE = (-1, 5)  # Hours since garage door closed

# Hourly bins for average ratios (fixed 1-hour windows)
HOURLY_BINS = [
    (-1, 0),
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
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
        process_aerotrak_data,
        process_quantaq_data,
        INSTRUMENT_CONFIG,
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
    peak_file = os.path.join(BASE_DIR, "burn_data", "peak_concentrations_all_instruments_edited.xlsx")

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
    Identify burns that have data from both OPC and Nef+OPC pairs

    Parameters
    ----------
    instruments : dict
        Dictionary of instrument DataFrames
    burn_log : pd.DataFrame
        Burn log with burn information

    Returns
    -------
    list
        List of burn IDs with complete data, ordered by number of CR boxes
    """
    valid_burns = []

    for _, row in burn_log.iterrows():
        burn_id = row["Burn ID"]

        # Skip excluded burns
        if burn_id in EXCLUDED_BURNS:
            continue

        # Check if CR Box was used
        if pd.isna(row["CR Box on"]) or row["CR Box on"] == "n/a":
            continue

        # Check if all four instruments have data for this burn
        burn_date = pd.to_datetime(row["Date"]).date()
        has_all_data = True

        for inst_name in ["AeroTrakB", "AeroTrakK", "QuantAQB", "QuantAQK"]:
            if instruments[inst_name] is None:
                has_all_data = False
                break

            inst_data = instruments[inst_name]
            if "Date" in inst_data.columns:
                burn_data = inst_data[inst_data["Date"] == burn_date]
                if burn_data.empty:
                    has_all_data = False
                    break

        if has_all_data:
            # Get number of CR boxes for ordering
            n_crboxes = row.get("# CR Boxes", 0)
            valid_burns.append((burn_id, n_crboxes))

    # Sort by number of CR boxes
    valid_burns.sort(key=lambda x: x[1])

    return [burn_id for burn_id, _ in valid_burns]


def calculate_peak_ratio_with_time(
    bedroom_data, morning_data, burn_id, pm_size,
    datetime_col_b, datetime_col_m, instrument_type
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

    garage_closed_time = create_naive_datetime(burn_info["Date"].iloc[0], garage_closed_str)

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

    if pd.isna(bedroom_peak_conc) or pd.isna(morning_peak_conc) or morning_peak_conc <= 0:
        return None, None

    # Calculate ratio
    ratio = bedroom_peak_conc / morning_peak_conc

    # Calculate mean peak time
    bedroom_peak_time = bedroom_burn.loc[bedroom_peak_idx, datetime_col_b]
    morning_peak_time = morning_burn.loc[morning_peak_idx, datetime_col_m]

    if pd.isna(bedroom_peak_time) or pd.isna(morning_peak_time):
        return None, None

    # Mean time in hours since garage closed
    bedroom_time_diff = (pd.Timestamp(bedroom_peak_time) - pd.Timestamp(garage_closed_time)).total_seconds() / 3600
    morning_time_diff = (pd.Timestamp(morning_peak_time) - pd.Timestamp(garage_closed_time)).total_seconds() / 3600
    mean_time_hours = (bedroom_time_diff + morning_time_diff) / 2

    return ratio, mean_time_hours


def calculate_crbox_ratio_with_time(
    bedroom_data, morning_data, burn_id, pm_size,
    datetime_col_b, datetime_col_m
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
    time_hours = (pd.Timestamp(cr_box_time) - pd.Timestamp(garage_closed_time)).total_seconds() / 3600

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
    bedroom_data, morning_data, burn_id, pm_size,
    datetime_col_b, datetime_col_m
):
    """
    Calculate average ratios for fixed hourly bins

    Returns
    -------
    list of tuples
        List of (ratio, time_hours) for each hourly bin
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

        # Calculate average ratio
        avg_ratio = ratios.mean()

        # Use midpoint of hourly bin for plotting
        time_hours = (start_hour + end_hour) / 2

        results.append((avg_ratio, time_hours))

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
        List of burn IDs with complete data (ordered by # CR boxes)
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

    # Generate color palette for burns
    from bokeh.palettes import Category20
    n_burns = len(valid_burns)
    if n_burns <= 20:
        colors = Category20[max(3, n_burns)]
    else:
        # Repeat colors if more than 20 burns
        colors = (Category20[20] * ((n_burns // 20) + 1))[:n_burns]

    burn_colors = dict(zip(valid_burns, colors))

    # Get appropriate PM size for AeroTrak
    aerotrak_pm_size = AEROTRAK_PM_SIZE_FOR_PM25 if pm_size == "PM2.5 (µg/m³)" else pm_size

    # Process each burn
    for burn_id in valid_burns:
        burn_color = burn_colors[burn_id]

        # Get burn info for CR box count (for legend)
        burn_info = burn_log[burn_log["Burn ID"] == burn_id]
        n_crboxes = burn_info["# CR Boxes"].iloc[0] if "# CR Boxes" in burn_info.columns else "?"

        # Calculate ratios for OPC (AeroTrak)
        # Peak ratio
        peak_ratio_opc, peak_time_opc = calculate_peak_ratio_with_time(
            instruments["AeroTrakB"], instruments["AeroTrakK"],
            burn_id, aerotrak_pm_size,
            "Date and Time", "Date and Time", "OPC"
        )

        # CR Box activation ratio
        crbox_ratio_opc, crbox_time_opc = calculate_crbox_ratio_with_time(
            instruments["AeroTrakB"], instruments["AeroTrakK"],
            burn_id, aerotrak_pm_size,
            "Date and Time", "Date and Time"
        )

        # Hourly average ratios
        hourly_ratios_opc = calculate_hourly_average_ratios(
            instruments["AeroTrakB"], instruments["AeroTrakK"],
            burn_id, aerotrak_pm_size,
            "Date and Time", "Date and Time"
        )

        # Calculate ratios for Nef+OPC (QuantAQ)
        # Peak ratio
        peak_ratio_nef, peak_time_nef = calculate_peak_ratio_with_time(
            instruments["QuantAQB"], instruments["QuantAQK"],
            burn_id, pm_size,
            "timestamp_local", "timestamp_local", "Nef+OPC"
        )

        # CR Box activation ratio
        crbox_ratio_nef, crbox_time_nef = calculate_crbox_ratio_with_time(
            instruments["QuantAQB"], instruments["QuantAQK"],
            burn_id, pm_size,
            "timestamp_local", "timestamp_local"
        )

        # Hourly average ratios
        hourly_ratios_nef = calculate_hourly_average_ratios(
            instruments["QuantAQB"], instruments["QuantAQK"],
            burn_id, pm_size,
            "timestamp_local", "timestamp_local"
        )

        # Plot OPC markers (hollow)
        if peak_ratio_opc is not None and peak_time_opc is not None:
            p.scatter(
                [peak_time_opc], [peak_ratio_opc],
                marker=RATIO_SHAPES["Peak"],
                size=10,
                color=burn_color,
                fill_alpha=0,  # Hollow
                line_width=2,
                legend_label=f"{burn_id} ({n_crboxes} CR Box{'es' if n_crboxes != 1 else ''}) - OPC",
            )

        if crbox_ratio_opc is not None and crbox_time_opc is not None:
            p.scatter(
                [crbox_time_opc], [crbox_ratio_opc],
                marker=RATIO_SHAPES["CR_Box"],
                size=10,
                color=burn_color,
                fill_alpha=0,  # Hollow
                line_width=2,
                legend_label=f"{burn_id} ({n_crboxes} CR Box{'es' if n_crboxes != 1 else ''}) - OPC",
            )

        if hourly_ratios_opc:
            times_opc = [t for _, t in hourly_ratios_opc]
            ratios_opc = [r for r, _ in hourly_ratios_opc]
            p.scatter(
                times_opc, ratios_opc,
                marker=RATIO_SHAPES["Hourly_Avg"],
                size=10,
                color=burn_color,
                fill_alpha=0,  # Hollow
                line_width=2,
                legend_label=f"{burn_id} ({n_crboxes} CR Box{'es' if n_crboxes != 1 else ''}) - OPC",
            )

        # Plot Nef+OPC markers (solid)
        if peak_ratio_nef is not None and peak_time_nef is not None:
            p.scatter(
                [peak_time_nef], [peak_ratio_nef],
                marker=RATIO_SHAPES["Peak"],
                size=10,
                color=burn_color,
                fill_alpha=1.0,  # Solid
                line_width=2,
                legend_label=f"{burn_id} ({n_crboxes} CR Box{'es' if n_crboxes != 1 else ''}) - Nef+OPC",
            )

        if crbox_ratio_nef is not None and crbox_time_nef is not None:
            p.scatter(
                [crbox_time_nef], [crbox_ratio_nef],
                marker=RATIO_SHAPES["CR_Box"],
                size=10,
                color=burn_color,
                fill_alpha=1.0,  # Solid
                line_width=2,
                legend_label=f"{burn_id} ({n_crboxes} CR Box{'es' if n_crboxes != 1 else ''}) - Nef+OPC",
            )

        if hourly_ratios_nef:
            times_nef = [t for _, t in hourly_ratios_nef]
            ratios_nef = [r for r, _ in hourly_ratios_nef]
            p.scatter(
                times_nef, ratios_nef,
                marker=RATIO_SHAPES["Hourly_Avg"],
                size=10,
                color=burn_color,
                fill_alpha=1.0,  # Solid
                line_width=2,
                legend_label=f"{burn_id} ({n_crboxes} CR Box{'es' if n_crboxes != 1 else ''}) - Nef+OPC",
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

    # Check if all instruments loaded successfully
    if any(instruments[inst] is None for inst in ["AeroTrakB", "AeroTrakK", "QuantAQB", "QuantAQK"]):
        print("\n[ERROR] Not all instruments loaded successfully. Cannot proceed.")
        return

    # Get burns with complete data
    print("\n" + "=" * 60)
    print("IDENTIFYING BURNS WITH COMPLETE DATA")
    print("=" * 60)
    valid_burns = get_burns_with_complete_data(instruments, burn_log)
    print(f"Found {len(valid_burns)} burns with data from both OPC and Nef+OPC pairs:")
    for burn_id in valid_burns:
        burn_info = burn_log[burn_log["Burn ID"] == burn_id]
        n_crboxes = burn_info["# CR Boxes"].iloc[0] if "# CR Boxes" in burn_info.columns else "?"
        print(f"  {burn_id}: {n_crboxes} CR Box(es)")

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
        div_text = (
            f'<div style="font-size: 10pt; font-weight: normal;">'
            f'<strong>Spatial Variation Analysis - Time Series</strong><br><br>'
            f'PM Size: {pm_size}<br>'
            f'Burns analyzed: {", ".join(valid_burns)}<br>'
            f'Excluded burns: {", ".join(EXCLUDED_BURNS) if EXCLUDED_BURNS else "None"}<br><br>'
            f'<strong>Marker Legend:</strong><br>'
            f'&nbsp;&nbsp;• Circle: Peak Ratio<br>'
            f'&nbsp;&nbsp;• Square: CR Box Activation Ratio<br>'
            f'&nbsp;&nbsp;• Triangle: Hourly Average Ratio<br>'
            f'&nbsp;&nbsp;• Hollow: OPC (AeroTrak)<br>'
            f'&nbsp;&nbsp;• Solid: Nef+OPC (QuantAQ)<br><br>'
            f'<hr><br>{metadata}</div>'
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
