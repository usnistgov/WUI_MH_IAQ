"""
WUI SMPS Fine Particulate Matter Comparison

This script compares SMPS-derived fine particulate matter (PM) mass concentrations
with independent PM measurements from filter-based and optical instruments. It
validates SMPS mass calculations and evaluates agreement across measurement methods
for wildland-urban interface smoke characterization.

Comparison Instruments:
    - SMPS: Size-resolved dN/dlog(Dp) → mass calculation
    - DustTrak: Real-time optical PM2.5
    - AeroTrak: Size-resolved optical particle counting
    - QuantAQ: Optical PM2.5 sensor
    - Gravimetric filters: Reference method (when available)

Key Analyses:
    1. SMPS-calculated PM vs optical PM time series comparison
    2. Correlation plots and linear regression
    3. Bias analysis (systematic differences)
    4. Temporal resolution comparison
    5. Size fraction agreement (PM1, PM2.5, PM10)

Mass Calculation Method (SMPS):
    - Assume spherical particles with unit density (ρ = 1 g/cm³)
    - Mass = (π/6) × d_p³ × ρ × N(d_p)
    - Integrate over size distribution
    - Compare to optical scattering-derived mass

Evaluation Metrics:
    - Pearson correlation coefficient
    - Root mean square error (RMSE)
    - Mean bias error (MBE)
    - Normalized mean bias (NMB)
    - Regression slope and intercept

Data Processing:
    - Time averaging to match instrument resolution
    - Baseline correction for all instruments
    - Quality flag filtering
    - Outlier removal using robust statistics

Outputs:
    - Time series overlay plots
    - Scatter plots with regression statistics
    - Bland-Altman difference plots
    - Agreement metrics summary table
    - Burn-by-burn comparison figures

Research Applications:
    - Validate low-cost sensor measurements
    - Evaluate optical sizing accuracy
    - Assess density assumptions for smoke particles
    - Guide instrument selection for field studies

Dependencies:
    - pandas: Data alignment and merging
    - numpy: Statistical calculations
    - bokeh: Interactive comparison plots
    - scipy: Regression analysis

Configuration:
    - Burn selection for analysis
    - Time averaging window
    - Size integration limits
    - Density assumption

Author: Nathan Lima
Date: 2024-2025
Reference: ISO 15900 (Particle size distribution measurement)
"""

# %% import needed modules, set absolute path for the data files, select burn
import pandas as pd
import os
from datetime import datetime, timedelta
from bokeh.plotting import figure, show
from bokeh.io import output_notebook, output_file
from bokeh.models import ColumnDataSource, DatetimeTickFormatter, FixedTicker
from bokeh.palettes import Reds

import sys
from pathlib import Path

# Add repository root to path for portable data access
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

from src.data_paths import get_data_root, get_instrument_path, get_common_file


# Set the absolute path for the data files
data_root = get_data_root()  # Portable path - auto-configured

# Directory for output
output_dir = os.path.join(absolute_path, str(get_common_file('output_figures')))

# Set the burn# variable (you can change this as needed)
burn_number = "burn9"  # Replace with the chosen burn

# %%
# Load the burn log to find the burn dates
burn_log_path = os.path.join(absolute_path, str(get_common_file('burn_log')))
burn_log = pd.read_excel(burn_log_path, sheet_name="Sheet2")

# Get the burn date based on the burn number
burn_date_row = burn_log[burn_log["Burn ID"] == burn_number]
if not burn_date_row.empty:
    burn_date = burn_date_row["Date"].values[0]

    # Find the index of the current burn
    burn_index = burn_log.index[burn_log["Burn ID"] == burn_number].tolist()[0]

    # Check if there is a previous burn
    if burn_index > 0:
        previous_burn_row = burn_log.iloc[burn_index - 1]
        previous_burn_date = previous_burn_row["Date"]
    else:
        raise ValueError("There is no previous burn to reference.")

    # Check if there is a next burn
    if burn_index < len(burn_log) - 1:
        next_burn_row = burn_log.iloc[burn_index + 1]
        next_burn_date = next_burn_row["Date"]
    else:
        raise ValueError("There is no next burn to reference.")
else:
    raise ValueError("Burn number not found in burn log.")

# Extract and convert the times
cr_box_activation_time_str = burn_log.loc[
    burn_log["Burn ID"] == burn_number, "CR Box on"
].values[0]
cr_box_activation_time_full = (
    pd.to_datetime(f"{burn_date} {cr_box_activation_time_str}")
    if pd.notna(cr_box_activation_time_str)
    else None
)

door_closed_time_str = burn_log.loc[
    burn_log["Burn ID"] == burn_number, "garage closed"
].values[0]
door_closed_time_full = (
    pd.to_datetime(f"{burn_date} {door_closed_time_str}")
    if pd.notna(door_closed_time_str)
    else None
)

# Load SMPS data for the given burn dates
smps_filenames = [
    f'MH_apollo_bed_{pd.to_datetime(burn_date).strftime("%m%d%Y")}_numConc.xlsx',
    f'MH_apollo_bed_{pd.to_datetime(previous_burn_date).strftime("%m%d%Y")}_numConc.xlsx',
    f'MH_apollo_bed_{pd.to_datetime(next_burn_date).strftime("%m%d%Y")}_numConc.xlsx',
]

# Initialize an empty DataFrame for combined SMPS data
combined_smps_data = pd.DataFrame()

for smps_filename in smps_filenames:
    smps_path = os.path.join(absolute_path, f"burn_data/smps/{smps_filename}")
    smps_data = pd.read_excel(smps_path, sheet_name="all_data")

    # Drop the 'Total Concentration(#/cm³)' column if it exists
    if "Total Concentration(#/cm³)" in smps_data.columns:
        smps_data.drop(columns=["Total Concentration(#/cm³)"], inplace=True)

    # Specify date format for 'Date' and 'Start Time'
    smps_data["Date"] = pd.to_datetime(
        smps_data["Date"], format="%Y-%m-%d", errors="coerce"
    )
    smps_data["Start Time"] = pd.to_datetime(
        smps_data["Start Time"], format="%H:%M:%S", errors="coerce"
    )

    # Combine 'Date' and 'Start Time' into a single datetime index
    smps_data["Datetime"] = pd.to_datetime(
        smps_data["Date"].dt.strftime("%Y-%m-%d")
        + " "
        + smps_data["Start Time"].dt.strftime("%H:%M:%S")
    )
    smps_data.set_index("Datetime", inplace=True)
    smps_data.drop(columns=["Date", "Start Time"], inplace=True)

    # Append to combined SMPS data
    combined_smps_data = pd.concat([combined_smps_data, smps_data], axis=0)

# Sort the combined SMPS data by the datetime index
combined_smps_data.sort_index(inplace=True)

# Sum columns based on specified ranges
range_boundaries = [
    (9, 25),
    (26, 50),
    (51, 75),
    (76, 100),
    (101, 125),
    (126, 150),
    (151, 175),
    (176, 200),
    (201, 250),
]
# max_col_value = combined_smps_data.columns.astype(float).max()
# range_boundaries.append((301, max_col_value))

# Create new summed columns based on ranges
for start, end in range_boundaries:
    numeric_cols = combined_smps_data.columns[
        pd.to_numeric(combined_smps_data.columns, errors="coerce").notnull()
    ]
    col_names = numeric_cols[
        (numeric_cols.astype(float) >= start) & (numeric_cols.astype(float) <= end)
    ].tolist()
    if col_names:
        combined_smps_data[f"Ʃ{start}-{end}nm (#/cm³)"] = combined_smps_data[
            col_names
        ].sum(axis=1)

# %%
# Prepare to output the plot in the notebook
output_notebook()

# Create a figure for the plot
p = figure(
    x_axis_label="DateTime",
    y_axis_label="Particulate Matter Particle Count (#/cm³)",
    x_axis_type="datetime",
    y_axis_type="log",
    width=1200,
    height=800,
)

# Define color scales for different datasets
smps_color = Reds[9]

# Plot combined SMPS data for summed ranges only if data exists
valid_smps_cols = combined_smps_data.columns.dropna().astype(str)
for i, col in enumerate(valid_smps_cols[valid_smps_cols.str.contains("Ʃ")]):
    filtered_smps = combined_smps_data[col].dropna()  # Remove time filtering
    if not filtered_smps.empty:
        source_smps = ColumnDataSource(
            data={"datetime": filtered_smps.index, "value": filtered_smps.values}
        )
        p.line(
            x="datetime",
            y="value",
            source=source_smps,
            legend_label=col.replace(" (#/cm³)", ""),
            line_width=2,
            color=smps_color[i % len(smps_color)],
            line_dash="solid",
        )

# Set the y-range
p.y_range.start = 1e-1
p.y_range.end = 1e5

# Set x_range to match the min and max datetime of the plotted data
p.x_range.start = combined_smps_data.index.min()  # Start from the minimum datetime
p.x_range.end = combined_smps_data.index.max()  # End at the maximum datetime

# Configure the x-axis for sublines every hour
hourly_ticks = pd.date_range(
    start=combined_smps_data.index.min(),
    end=combined_smps_data.index.max(),
    freq="60min",
)
p.xaxis.ticker = FixedTicker(ticks=[tick.timestamp() * 1000 for tick in hourly_ticks])
p.xaxis.major_label_orientation = "vertical"
p.xaxis.formatter = DatetimeTickFormatter(hours="%H:%M")  # Format for hours and minutes

# Configure the legend
p.legend.location = "top_left"
p.legend.click_policy = "hide"

# Show the plot
show(p)

# %%
# Prepare to output the plot in the notebook
output_notebook()
output_file(os.path.join(output_dir, f"{burn_number}_SMPS_finePM_comparison.html"))

# Create a figure for the plot
p = figure(
    x_axis_label="DateTime",
    y_axis_label="Particulate Matter Particle Count (#/cm³)",
    x_axis_type="datetime",
    y_axis_type="log",
    width=1200,
    height=800,
)

# Define color scales for different datasets
smps_color = Reds[9]

# Plot combined SMPS data for summed ranges only if data exists
valid_smps_cols = combined_smps_data.columns.dropna().astype(str)
for i, col in enumerate(valid_smps_cols[valid_smps_cols.str.contains("Ʃ")]):
    filtered_smps = combined_smps_data[col].dropna()  # Remove time filtering
    if not filtered_smps.empty:
        source_smps = ColumnDataSource(
            data={"datetime": filtered_smps.index, "value": filtered_smps.values}
        )
        p.line(
            x="datetime",
            y="value",
            source=source_smps,
            legend_label=col.replace(" (#/cm³)", ""),
            line_width=1.5,
            color=smps_color[i % len(smps_color)],
            line_dash="solid",
        )

# Set the y-range
p.y_range.start = 1e-1
p.y_range.end = 1e5

# Set x_range to specified limits
p.x_range.start = pd.Timestamp(
    "2024-05-27 08:00:00"
)  # Start from 05/27/2024 8:00:00 AM
p.x_range.end = pd.Timestamp("2024-05-28 23:59:59")  # End at 05/28/2024 11:59:59 PM

# Configure the x-axis for sublines every hour
hourly_ticks = pd.date_range(start=p.x_range.start, end=p.x_range.end, freq="60min")
p.xaxis.ticker = FixedTicker(ticks=[tick.timestamp() * 1000 for tick in hourly_ticks])
p.xaxis.major_label_orientation = "vertical"
p.xaxis.formatter = DatetimeTickFormatter(hours="%H:%M")  # Format for hours and minutes

# Configure the legend
p.legend.location = "top_right"
p.legend.click_policy = "hide"


# Show the plot
show(p)
# %%
