#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WUI AeroTrak vs SMPS Comparison Analysis

This script compares particle size distributions and mass concentrations measured
by AeroTrak optical particle counters, SMPS (Scanning Mobility Particle Sizer),
and QuantAQ monitors during wildland-urban interface smoke experiments. It evaluates
agreement between optical and mobility-based sizing methods across different particle
size ranges.

Comparison Features:
    - Multi-instrument particle count comparison (AeroTrak, SMPS, QuantAQ)
    - Temporal trends during decay period after peak concentration
    - Size-binned number concentrations across multiple size ranges
    - CR Box (air purifier) activation impact analysis

Instrument Characteristics:
    AeroTrak (Optical):
        - Size range: 0.3-10 µm (optical diameter)
        - 6 size channels (0.3, 0.5, 1.0, 3.0, 5.0, 10.0 µm)
        - 1-minute time resolution
        - Light scattering detection

    SMPS (Electrical Mobility):
        - Size range: 9-500+ nm (mobility diameter)
        - High-resolution size bins
        - Variable scan time
        - Electrical mobility classification

    QuantAQ (Optical):
        - Size range: 0.35-40 µm
        - 24 size bins aggregated into ranges
        - 1-minute time resolution

Analysis Methods:
    1. Data loading and temporal alignment
    2. Unit conversion and normalization
    3. Time-series overlay visualization
    4. Concentration comparison across overlapping size ranges

Script Outputs:
    - Interactive Bokeh HTML plot showing:
        * AeroTrak particle counts (6 channels)
        * SMPS size-binned concentrations (4 ranges)
        * QuantAQ size-binned concentrations (7 ranges)
        * CR Box activation timing marker
    - Terminal output: Data loading confirmation and file paths
    - Figure displayed in Jupyter notebook (if running in notebook)

Dependencies:
    - pandas: Data manipulation and time series handling
    - bokeh: Interactive visualization
    - pathlib: Path operations

Configuration Requirements:
    - data_config.json must be configured with:
        * 'aerotrak_bedroom' instrument path (Excel file)
        * 'smps' instrument path (Excel file)
        * 'quantaq_bedroom' instrument path (CSV file)

Author: Nathan Lima
Date: 2024-2025
Institution: National Institute of Standards and Technology (NIST)
"""
# Import needed modules
import sys
from pathlib import Path
import pandas as pd
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import Range1d, Label

# Add repository root to path for portable data access
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

from src.data_paths import get_instrument_path, get_common_file, get_instrument_files


# User inputs for burn and burn_date
BURN = input("Enter the burn number (e.g., 10 for burn10): ")
BURN_DATE = "2024-05-31"

# Constants
AEROTRAK_UNIT_CONVERSION = 3000  # Convert from L to cm³
TIME_WINDOW_START = -120  # minutes from peak
CR_BOX_ACTIVATION_TIME = -8  # minutes from peak

# %% AeroTrak Data Processing
# Define columns to drop from the AeroTrak dataset
AEROTRAK_COLUMNS_TO_DROP = [
    "Date and Time",
    "Ch1 Size (µm)",
    "Ch1 0.3µm (#)",
    "Ch2 Size (µm)",
    "Ch2 0.5µm (#)",
    "Ch3 Size (µm)",
    "Ch3 1.0µm (#)",
    "Ch4 Size (µm)",
    "Ch4 3.0µm (#)",
    "Ch5 Size (µm)",
    "Ch5 5.0µm (#)",
    "Ch6 Size (µm)",
    "Ch6 10.0µm (#)",
    "Unnamed: 21",
    "Unnamed: 22",
    "Unnamed: 23",
    "total_count (#)",
]


def rename_aerotrak_columns(burn_sheet_name):
    """
    Rename AeroTrak columns to include sheet name and proper units.

    Parameters
    ----------
    burn_sheet_name : str
        Name of the Excel sheet (e.g., 'burn10')

    Returns
    -------
    dict
        Mapping of old column names to new formatted names
    """
    return {
        "Ch1 Diff (#)": f"{burn_sheet_name} Ch1 0.3µm (#/cm³)",
        "Ch2 Diff (#)": f"{burn_sheet_name} Ch2 0.5µm (#/cm³)",
        "Ch3 Diff (#)": f"{burn_sheet_name} Ch3 1.0µm (#/cm³)",
        "Ch4 Diff (#)": f"{burn_sheet_name} Ch4 3.0µm (#/cm³)",
        "Ch5 Diff (#)": f"{burn_sheet_name} Ch5 5.0µm (#/cm³)",
        "Ch6 Diff (#)": f"{burn_sheet_name} Ch6 10.0µm (#/cm³)",
    }


# Load data only for the specified burn
burn_sheet_name = f"burn{BURN}"
# AeroTrak burn dates file is in common_folders, not instrument path
aerotrak_file = get_common_file('burn_dates_decay_aerotrak_bedroom')
aerotrak_df = pd.read_excel(aerotrak_file, sheet_name=burn_sheet_name)
aerotrak_df = aerotrak_df.drop(columns=AEROTRAK_COLUMNS_TO_DROP)

# Unit conversion from L to cubic centimeters
diff_columns = [col for col in aerotrak_df.columns if "Diff" in col]
aerotrak_df[diff_columns] = aerotrak_df[diff_columns] / AEROTRAK_UNIT_CONVERSION

# Rename columns
aerotrak_df.rename(columns=rename_aerotrak_columns(burn_sheet_name), inplace=True)

# Set the index and interpolate missing values
aerotrak_df = aerotrak_df.set_index("min_since_peak")
aerotrak_df = aerotrak_df.interpolate("index")
aerotrak_df["min_from_peak"] = aerotrak_df.index
aerotrak_df = aerotrak_df[aerotrak_df["min_from_peak"] >= TIME_WINDOW_START]

# %% RUN Load and Process SMPS Data
# Find SMPS NumConc file for the specified date
smps_path = get_instrument_path('smps')
# Convert BURN_DATE from YYYY-MM-DD to MMDDYYYY format for filename
date_parts = BURN_DATE.split("-")  # ['2024', '05', '31']
smps_date_str = f"{date_parts[1]}{date_parts[2]}{date_parts[0]}"  # 05312024
smps_file = smps_path / f'MH_apollo_bed_{smps_date_str}_NumConc.xlsx'
smps_df = pd.read_excel(smps_file, sheet_name="all_data")

# Combine 'Date' and 'Start Time' into a single datetime column
smps_df["Datetime"] = pd.to_datetime(
    smps_df["Date"].astype(str) + " " + smps_df["Start Time"].astype(str)
)

# Filter data for the specific date
date_filter = pd.Timestamp(BURN_DATE)
smps_df = smps_df[smps_df["Datetime"].dt.date == date_filter.date()]


def sum_columns_by_range(data_df, size_ranges):
    """
    Sum SMPS columns based on specified particle size ranges.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame with numeric column names representing particle sizes
    size_ranges : list of tuple
        List of (start, end) size ranges in nanometers

    Returns
    -------
    dict
        Dictionary mapping range labels to summed Series
    """
    summed_data = {}
    for start, end in size_ranges:
        col_range = [
            col
            for col in data_df.columns
            if str(col).replace(".", "", 1).isdigit() and start <= float(col) <= end
        ]
        summed_data[f"Ʃ{start}-{end}nm (#/cm³)"] = data_df[col_range].sum(axis=1)
    return summed_data


# Define the ranges for summing
SMPS_RANGES = [(9, 100), (101, 200), (201, 300), (300, float("inf"))]
smps_summed_data = sum_columns_by_range(smps_df, SMPS_RANGES)

# Calculate min_from_peak for SMPS data
start_time = pd.Timestamp(f"{BURN_DATE} 09:03:00")
smps_df["min_from_peak"] = (smps_df["Datetime"] - start_time).dt.total_seconds() / 60

# %% RUN Load and Process QuantAQ Data
# Find QuantAQ file using the pattern from config
quantaq_files = get_instrument_files('quantaq_bedroom')
if not quantaq_files:
    raise FileNotFoundError("No QuantAQ bedroom files found. Check data_config.json path.")
# Use the first matching file (or could filter by date if multiple exist)
quantaq_file = quantaq_files[0]
quantaq_df = pd.read_csv(quantaq_file)

# Reverse the order of the data and parse datetime
quantaq_df = quantaq_df[::-1].reset_index(drop=True)
quantaq_df["timestamp_local"] = pd.to_datetime(
    quantaq_df["timestamp_local"].str.replace("T", " ").str.replace("Z", "")
)

# Calculate min_from_peak for QuantAQ data
quantaq_df["min_from_peak"] = (
    quantaq_df["timestamp_local"] - start_time - pd.Timedelta(minutes=5.5)
).dt.total_seconds() / 60

# Define the ranges for summing in QuantAQ data
QUANTAQ_BINS = {
    "Ʃ0.35-0.66 µm (#/cm³)": ["bin0", "bin1"],
    "0.66-1.0 µm (#/cm³)": ["bin2"],
    "Ʃ1.0-3.0 µm (#/cm³)": ["bin3", "bin4", "bin5", "bin6"],
    "Ʃ3.0-5.2 µm (#/cm³)": ["bin7", "bin8"],
    "Ʃ5.2-10 µm (#/cm³)": ["bin9", "bin10", "bin11"],
    "Ʃ10-20 µm (#/cm³)": ["bin12", "bin13", "bin14", "bin15", "bin16"],
    "Ʃ20-40 µm (#/cm³)": [
        "bin17",
        "bin18",
        "bin19",
        "bin20",
        "bin21",
        "bin22",
        "bin23",
    ],
}

# Calculate summed data for QuantAQ
quantaq_summed_data = {}
for label, cols in QUANTAQ_BINS.items():
    quantaq_summed_data[label] = quantaq_df[cols].sum(axis=1)

# %% RUN Plotting AeroTrak, SMPS, and QuantAQ Data
# Define color scales for different data types
SMPS_COLORS = ["#ff0000", "#ff3333", "#ff6666", "#ff9999"]
QUANTAQ_COLORS = [
    "#080be5",
    "#0068ff",
    "#0093ff",
    "#00b4ff",
    "#00d1d2",
    "#00eb7f",
    "#29ff06",
]
AEROTRAK_COLORS = ["#f8aa39", "#f1ae58", "#e8b274", "#ddb78d", "#d0bba7", "#c0c0c0"]

# Plot configuration
X_RANGE_MIN = -60
X_RANGE_MAX = 120
Y_RANGE_MIN = 1e-3
Y_RANGE_MAX = 1e5

# For visualization in Jupyter Notebook
output_notebook()

# Create a Bokeh figure with logarithmic y-axis and minutes on x-axis
p = figure(
    x_axis_label="Time since peak conc. after particle growth (minutes)",
    y_axis_label="Particulate Matter (#/cm³)",
    y_axis_type="log",
    x_axis_type="linear",
    width=900,
    height=700,
    title=f"Burn {BURN} Bedroom Two Particle Count Comparison",
    x_range=Range1d(start=X_RANGE_MIN, end=X_RANGE_MAX),
    y_range=Range1d(start=Y_RANGE_MIN, end=Y_RANGE_MAX),
)

# After plotting AeroTrak data
p.line(
    [], [], legend_label="AeroTrak", color=None, line_width=0
)  # Add a dummy line for AeroTrak

for label, color in zip(aerotrak_df.keys(), AEROTRAK_COLORS):
    data = aerotrak_df[label]
    # Remove the units from the label
    clean_label = label.replace(" (#/cm³)", "").replace(f"{burn_sheet_name} ", "")
    p.line(
        aerotrak_df["min_from_peak"].tolist(),
        data.tolist(),
        legend_label=clean_label,
        color=color,
        line_dash="solid",
        line_width=1.5,
    )

# After plotting SMPS data
p.line(
    [], [], legend_label="SMPS", color=None, line_width=0
)  # Add a dummy line for SMPS

for idx, (label, series) in enumerate(smps_summed_data.items()):
    # Remove the units from the label
    clean_label = label.replace(" (#/cm³)", "")
    p.line(
        smps_df["min_from_peak"].tolist(),
        series.tolist(),
        legend_label=clean_label,
        color=SMPS_COLORS[idx % len(SMPS_COLORS)],
        line_dash="dotted",
        line_width=1.5,
    )

# After plotting QuantAQ data
p.line(
    [], [], legend_label="QuantAQ", color=None, line_width=0
)  # Add a dummy line for QuantAQ

for idx, (label, series) in enumerate(quantaq_summed_data.items()):
    # Remove the units from the label
    clean_label = label.replace(" (#/cm³)", "")
    p.line(
        quantaq_df["min_from_peak"].tolist(),
        series.tolist(),
        legend_label=clean_label,
        color=QUANTAQ_COLORS[idx % len(QUANTAQ_COLORS)],
        line_dash="dashed",
        line_width=1.5,
    )

# Add a vertical line at CR Box activation time
p.line(
    [CR_BOX_ACTIVATION_TIME, CR_BOX_ACTIVATION_TIME],
    [Y_RANGE_MIN, Y_RANGE_MAX],
    line_color="black",
    line_width=2,
    legend_label="CR Boxes Activation",
)

# Add a label for the vertical line
label = Label(
    x=CR_BOX_ACTIVATION_TIME - 2,
    y=1e4,
    text="CR Boxes Activation",
    text_font_size="10pt",
    text_color="black",
    text_align="right",
    text_baseline="middle",
)
p.add_layout(label)

# Customize and show the plot
p.legend.location = "top_right"
p.legend.orientation = "vertical"

# Customize font properties
p.axis.axis_label_text_font = "Calibri"
p.axis.axis_label_text_font_size = "12pt"
p.axis.axis_label_text_font_style = "normal"

# Show the plot
# output_file('./Paper_figures/AeroTrak_SMPS_QuantAQ_comparison.html')
show(p)
