#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WUI Spatial Variation Analysis Plotting Script
===============================================

This script generates interactive Bokeh visualizations of spatial variation analysis
results from wildfire smoke infiltration experiments. It creates plots comparing
particulate matter (PM) concentration ratios between two locations (bedroom2 vs
morning room) under different CR Box (portable air cleaner) operating conditions.

Key Features:
    - Visualizes three ratio metrics: Peak Ratio Index, CR Box Activation Ratio,
      and Average Ratio
    - Compares data from two instrument types: AeroTrak and QuantAQ
    - Analyzes three burns with different CR Box configurations (1, 2, and 4 units)
    - Generates size-resolved plots for PM1, PM2.5, and PM10
    - Fits smooth curves to data for trend analysis
    - Embeds script metadata into HTML output for reproducibility
    - Automatically detects desktop vs laptop system for file paths

Input:
    - spatial_variation_analysis.xlsx: Excel file with AeroTrak and QuantAQ sheets
      containing calculated ratios for each burn and PM size

Output:
    - Individual HTML files for each PM size with interactive Bokeh plots
    - Plots saved to Paper_figures directory

Methodology:
    - X-axis represents number of CR Boxes operating (1, 2, or 4 units)
    - Y-axis shows concentration ratios (bedroom2/morning room)
    - Scatter points show measured ratios for each burn
    - Colors distinguish different burns (burn2, burn4, burn9)
    - Marker shapes distinguish instruments (AeroTrak vs QuantAQ)
    - Fitted curves use spline interpolation (or quadratic fallback)
    - Ratio = 1.0 indicates perfect spatial uniformity

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2025
"""

import os
import sys
import pandas as pd
from bokeh.plotting import figure, output_file, save
from bokeh.models import Span
import numpy as np
from scipy.interpolate import interp1d, make_interp_spline


# ============================================================================
# SYSTEM DETECTION AND PATH SETUP
# ============================================================================


def detect_system():
    """
    Detect which system the script is running on

    Returns
    -------
    str
        'desktop' if running on desktop computer with OneDrive
        'laptop' if running on laptop computer
    """
    desktop_onedrive_path = r"C:\Users\nml\OneDrive - NIST"
    if os.path.exists(desktop_onedrive_path):
        return "desktop"
    return "laptop"


# Detect system and set paths accordingly
SYSTEM = detect_system()

if SYSTEM == "desktop":
    BASE_DIR = r"C:\Users\nml\OneDrive - NIST\Documents\NIST\WUI_smoke"
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(SCRIPT_DIR)
    GRANDPARENT_DIR = os.path.dirname(PARENT_DIR)
    UTILS_PATH = os.path.join(GRANDPARENT_DIR, "general_utils", "scripts")
else:  # laptop
    BASE_DIR = r"C:\Users\Nathan\Documents\NIST\WUI_smoke"
    UTILS_PATH = r"C:\Users\Nathan\Documents\GitHub\python_coding\general_utils\scripts"

# Add utils to path
sys.path.append(UTILS_PATH)

# Import metadata utilities
# pylint: disable=import-error, wrong-import-position
try:
    from metadata_utils import get_script_metadata

    METADATA_AVAILABLE = True
except ImportError:
    METADATA_AVAILABLE = False
    print("Warning: metadata_utils not available, metadata will not be added to plots")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Input file path - Excel file with spatial variation analysis results
EXCEL_FILE_PATH = os.path.join(BASE_DIR, "burn_data", "spatial_variation_analysis.xlsx")

# Output directory for figures
OUTPUT_DIR = os.path.join(BASE_DIR, "Paper_figures")

# Burns to analyze and their corresponding CR Box counts
# burn2: 4 CR Boxes, burn4: 1 CR Box, burn9: 2 CR Boxes
BURN_IDS = ["burn2", "burn4", "burn9"]
BURN_TO_CRBOX_COUNT = {"burn2": 4, "burn4": 1, "burn9": 2}

# Color mapping for burns (each burn gets a unique color)
BURN_COLORS = {
    "burn2": "#1f77b4",  # Blue
    "burn4": "#ff7f0e",  # Orange
    "burn9": "#2ca02c",  # Green
}

# Marker mapping for instruments (each instrument gets a unique marker)
INSTRUMENT_MARKERS = {
    "AeroTrak": "circle",
    "QuantAQ": "square",
}

# PM sizes to create plots for (QuantAQ sizes)
PM_SIZES = ["PM1 (µg/m³)", "PM2.5 (µg/m³)", "PM10 (µg/m³)"]

# AeroTrak PM3 is used as a proxy for PM2.5 comparison
# Note: AeroTrak measures PM3 which is the closest size bin to PM2.5
AEROTRAK_PM_SIZE_FOR_PM25 = "PM3 (µg/m³)"

# Ratio metrics to plot
RATIO_METRICS = ["Peak_Ratio_Index", "CRBox_Activation_Ratio", "Average_Ratio"]

# Plot configuration
PLOT_Y_RANGE = (0, 1.5)  # Y-axis range for ratio plots
X_AXIS_TICKS = [1, 2, 3, 4]  # X-axis tick positions (CR Box counts)
X_AXIS_LABELS = {1: "1", 2: "2", 3: "", 4: "4"}  # Custom labels (3 is hidden)


# ============================================================================
# DATA LOADING
# ============================================================================
def load_spatial_variation_data(file_path):
    """
    Load spatial variation analysis results from Excel file

    Parameters
    ----------
    file_path : str
        Path to the Excel file containing spatial variation analysis results

    Returns
    -------
    tuple of pd.DataFrame
        (aerotrak_df, quantaq_df) - DataFrames for AeroTrak and QuantAQ data

    Raises
    ------
    FileNotFoundError
        If the Excel file is not found
    ValueError
        If required sheets are missing from the Excel file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found: {file_path}")

    try:
        aerotrak_df = pd.read_excel(file_path, sheet_name="AeroTrak")
        quantaq_df = pd.read_excel(file_path, sheet_name="QuantAQ")
    except ValueError as e:
        raise ValueError(
            f"Error reading Excel file. Ensure it has 'AeroTrak' and 'QuantAQ' sheets: {e}"
        ) from e

    return aerotrak_df, quantaq_df


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================
def create_fitted_curve(x_data, y_data, num_points=100):
    """
    Create a smooth fitted curve through data points

    Uses cubic spline interpolation if possible, falls back to quadratic
    interpolation if the spline fails (e.g., with insufficient data points).

    Parameters
    ----------
    x_data : array-like
        X coordinates of data points
    y_data : array-like
        Y coordinates of data points
    num_points : int, optional
        Number of points in the fitted curve (default: 100)

    Returns
    -------
    tuple of np.ndarray
        (x_fit, y_fit) - X and Y coordinates of fitted curve

    Notes
    -----
    - Cubic spline (k=3) requires at least 4 data points
    - Quadratic interpolation requires at least 3 data points
    - Returns None, None if fitting fails
    """
    try:
        # Need at least 3 unique x values for interpolation
        if len(x_data) < 3:
            return None, None

        x_fit = np.linspace(x_data.min(), x_data.max(), num_points)

        try:
            # Try cubic spline for smooth curve
            f = make_interp_spline(x_data, y_data, k=3)
            y_fit = f(x_fit)
        except (ValueError, TypeError):
            # Fallback to quadratic fit if spline fails
            try:
                f = interp1d(x_data, y_data, kind="quadratic")
                y_fit = f(x_fit)
            except (ValueError, TypeError):
                # If quadratic also fails, try linear
                try:
                    f = interp1d(x_data, y_data, kind="linear")
                    y_fit = f(x_fit)
                except (ValueError, TypeError):
                    return None, None

        return x_fit, y_fit

    except (ValueError, TypeError) as e:
        print(f"  Warning: Could not create fitted curve: {e}")
        return None, None


def create_spatial_variation_plot(
    aerotrak_data, quantaq_data, pm_size, burn_to_crbox_map
):
    """
    Create Bokeh plot for spatial variation ratios

    Parameters
    ----------
    aerotrak_data : pd.DataFrame
        Filtered AeroTrak data for the specific PM size
    quantaq_data : pd.DataFrame
        Filtered QuantAQ data for the specific PM size
    pm_size : str
        PM size label (e.g., 'PM2.5 (µg/m³)')
    burn_to_crbox_map : dict
        Mapping from burn ID to CR Box count

    Returns
    -------
    bokeh.plotting.figure.Figure
        Bokeh figure object with plotted data and fitted curves
    """
    # Create figure
    p = figure(
        title=f"Spatial Variation Analysis - {pm_size}",
        x_axis_label="Number of CR Boxes",
        y_axis_label="Concentration Ratio (Bedroom2 / Morning Room)",
        y_range=PLOT_Y_RANGE,
        width=800,
        height=600,
    )

    # Plot data for each ratio type
    for ratio in RATIO_METRICS:
        # Process each instrument separately
        for device_data, device_name in [
            (aerotrak_data, "AeroTrak"),
            (quantaq_data, "QuantAQ"),
        ]:
            if device_data.empty:
                continue

            # Skip if ratio column doesn't exist or has no valid data
            if ratio not in device_data.columns:
                print(f"  Warning: {ratio} not found in {device_name} data")
                continue

            # Get marker shape for this instrument
            marker = INSTRUMENT_MARKERS.get(device_name, "circle")

            # Collect all data points for this instrument and ratio for fitting
            all_x = []
            all_y = []

            # Plot individual data points, colored by burn
            for burn_id in BURN_IDS:
                burn_data = device_data[device_data["Burn_ID"] == burn_id]

                if burn_data.empty:
                    continue

                # Map burn ID to CR Box count for x-axis
                x_val = burn_to_crbox_map.get(burn_id)
                y_val = burn_data[ratio].values

                if x_val is None or len(y_val) == 0:
                    continue

                y_val = y_val[0]  # Get first value

                # Skip NaN values
                if pd.isna(x_val) or pd.isna(y_val):
                    continue

                # Get color for this burn
                color = BURN_COLORS.get(burn_id, "black")

                # Create legend label (only for first ratio to avoid duplicates)
                if ratio == RATIO_METRICS[0]:
                    legend_label = f"{device_name} - {burn_id}"
                else:
                    legend_label = None

                # Plot scatter point
                p.scatter(
                    [x_val],
                    [y_val],
                    legend_label=legend_label,
                    color=color,
                    marker=marker,
                    size=10,
                    alpha=0.7,
                )

                # Collect for fitting
                all_x.append(x_val)
                all_y.append(y_val)

            # Add fitted curve for this instrument and ratio (if we have enough points)
            if len(all_x) >= 2:  # Need at least 2 points
                # Convert to numpy arrays and sort by x
                x_array = np.array(all_x)
                y_array = np.array(all_y)
                sort_idx = np.argsort(x_array)
                x_sorted = x_array[sort_idx]
                y_sorted = y_array[sort_idx]

                x_fit, y_fit = create_fitted_curve(x_sorted, y_sorted)
                if x_fit is not None and y_fit is not None:
                    # Use same color as the dominant burn or a neutral color
                    # For simplicity, use a muted version of the instrument color
                    fit_color = "gray" if device_name == "AeroTrak" else "darkgray"

                    # Plot fit line WITHOUT adding to legend
                    p.line(
                        x_fit,
                        y_fit,
                        color=fit_color,
                        line_width=2,
                        alpha=0.3,
                        line_dash="dashed",
                    )

    # Customize x-axis ticks and labels
    p.xaxis.ticker = X_AXIS_TICKS
    p.xaxis.major_label_overrides = X_AXIS_LABELS

    # Customize legend
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"  # Allow clicking legend to hide/show lines

    # Add horizontal line at y=1.0 (perfect spatial uniformity)
    uniform_line = Span(
        location=1.0,
        dimension="width",
        line_color="black",
        line_dash="dotted",
        line_width=1,
        line_alpha=0.5,
    )
    p.add_layout(uniform_line)

    return p


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """
    Main function to generate spatial variation plots

    Loads data, filters by selected burns, creates plots for each PM size,
    and saves HTML files with embedded metadata.
    """
    print("=" * 80)
    print("WUI SPATIAL VARIATION ANALYSIS - PLOTTING")
    print("=" * 80)
    print(f"Running on {SYSTEM} system")
    print(f"Base directory: {BASE_DIR}")
    print(f"Utils path: {UTILS_PATH}")

    # Check if base directory exists
    if not os.path.exists(BASE_DIR):
        print(f"\nWARNING: Base directory not found: {BASE_DIR}")
        print("Please check the path configuration.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Load data
    print(f"\nLoading data from: {EXCEL_FILE_PATH}")
    try:
        aerotrak_df, quantaq_df = load_spatial_variation_data(EXCEL_FILE_PATH)
        print(f"  AeroTrak data: {aerotrak_df.shape}")
        print(f"  QuantAQ data: {quantaq_df.shape}")
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        return

    # Filter data for selected burn IDs
    print(f"\nFiltering data for burns: {BURN_IDS}")
    aerotrak_df = aerotrak_df[aerotrak_df["Burn_ID"].isin(BURN_IDS)]
    quantaq_df = quantaq_df[quantaq_df["Burn_ID"].isin(BURN_IDS)]
    print(f"  Filtered AeroTrak data: {aerotrak_df.shape}")
    print(f"  Filtered QuantAQ data: {quantaq_df.shape}")

    # Generate plots for each PM size
    print(f"\nGenerating plots for PM sizes: {PM_SIZES}")
    for pm_size in PM_SIZES:
        print(f"\n{'='*60}")
        print(f"Processing {pm_size}")
        print(f"{'='*60}")

        # Filter data for current PM size
        # Special case: Use PM3 from AeroTrak as proxy for PM2.5
        if pm_size == "PM2.5 (µg/m³)":
            aerotrak_data = aerotrak_df[
                aerotrak_df["PM_Size"] == AEROTRAK_PM_SIZE_FOR_PM25
            ]
            print(f"  Using {AEROTRAK_PM_SIZE_FOR_PM25} from AeroTrak as PM2.5 proxy")
        else:
            aerotrak_data = aerotrak_df[aerotrak_df["PM_Size"] == pm_size]

        quantaq_data = quantaq_df[quantaq_df["PM_Size"] == pm_size]

        print(f"  AeroTrak data points: {len(aerotrak_data)}")
        print(f"  QuantAQ data points: {len(quantaq_data)}")

        # Create plot
        p = create_spatial_variation_plot(
            aerotrak_data, quantaq_data, pm_size, BURN_TO_CRBOX_COUNT
        )

        # Save figure
        output_filename = f"{pm_size.replace(' ', '_').replace('/', '_')}.html"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        output_file(output_path)
        save(p)
        print(f"  Saved plot to: {output_path}")

        # Add metadata to HTML file for reproducibility
        if METADATA_AVAILABLE:
            try:
                metadata = get_script_metadata(__file__)

                # Read the HTML file
                with open(output_path, "r", encoding="utf-8") as f:
                    html_content = f.read()

                # Add metadata as HTML comment before closing </body> tag
                metadata_comment = f"\n<!-- Script Metadata:\n{metadata}\n-->\n"

                # Insert before </body> or at the end if no </body>
                if "</body>" in html_content:
                    html_content = html_content.replace(
                        "</body>", f"{metadata_comment}</body>"
                    )
                else:
                    html_content += metadata_comment

                # Write back
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html_content)

                print("  Added metadata to HTML file")
            except (OSError, RuntimeError, ValueError, TypeError) as e:
                print(f"  Warning: Could not add metadata to HTML file: {e}")
        else:
            print("  Metadata not available (metadata_utils not imported)")

    print("\n" + "=" * 80)
    print("PLOTTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
