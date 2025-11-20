"""
WUI SMPS Mass vs Number Concentration Analysis

This script analyzes the relationship between particle number concentration
and particle mass concentration from SMPS measurements during wildland-urban
interface smoke experiments. It examines how size distribution characteristics
affect the number-to-mass conversion.

Key Analyses:
    - Number concentration (#/cm³) vs mass concentration (µg/m³) correlation
    - Mode diameter evolution and its effect on mass/number ratio
    - Geometric standard deviation impacts
    - Temporal dynamics of distribution parameters

Methodology:
    - Extract number concentration: ∫dN/dlog(Dp) d(log Dp)
    - Calculate mass concentration: ∫(π/6 × d_p³ × ρ × dN/dlog(Dp)) d(log Dp)
    - Track mass/number ratio: M/N (pg per particle)
    - Identify dominant size modes

Distribution Characterization:
    - Count median diameter (CMD)
    - Mass median diameter (MMD)
    - Geometric mean diameter (GMD)
    - Geometric standard deviation (GSD)
    - Total concentration metrics

Research Questions:
    1. How does particle growth affect mass vs number trends?
    2. Does coagulation or condensation dominate during decay?
    3. What is the characteristic smoke particle size?
    4. How do different filter configurations affect size distribution?

Outputs:
    - Mass vs number scatter plots
    - Mass/number ratio time series
    - Size parameter tracking plots
    - Distribution parameter summary tables

Dependencies:
    - pandas: Data processing
    - numpy: Numerical calculations
    - bokeh: Visualization

Author: Nathan Lima
Date: 2024-2025
"""

# %% RUN
# Import needed modules
print("Importing needed modules")
import os
import pandas as pd
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import LinearAxis, LogAxis, Range1d

# RUN User defines directory path for dataset and dataset selection
# User set absolute_path
absolute_path = "C:/Users/nml/OneDrive - NIST/Documents/NIST/WUI_smoke/"  # USER ENTERED PROJECT PATH
os.chdir(absolute_path)

# %% RUN Load and process SMPS #/cm3
SMPS_path = "./burn_data/smps/MH_apollo_bed_05012024_numConc.xlsx"
SMPS_df = pd.read_excel(SMPS_path, sheet_name="all_data")

# Combine 'Date' and 'Start Time' into a single datetime column
SMPS_df["Datetime"] = pd.to_datetime(
    SMPS_df["Date"].astype(str) + " " + SMPS_df["Start Time"].astype(str)
)

# Filter data for the specific date (5/2/2024)
date_filter = pd.Timestamp("2024-05-02")
SMPS_df = SMPS_df[SMPS_df["Datetime"].dt.date == date_filter.date()]


# Define column ranges for summing
def sum_columns(df, ranges):
    summed_data = {}
    for start, end in ranges:
        col_range = [
            col
            for col in df.columns
            if str(col).replace(".", "", 1).isdigit() and start <= float(col) <= end
        ]
        summed_data[f"Ʃ{start}-{end}nm (#/cm³)"] = df[col_range].sum(axis=1)
    return summed_data


# Define the ranges
ranges = [(9, 100), (101, 200), (201, 300), (300, float("inf"))]
summed_data = sum_columns(SMPS_df, ranges)

# Calculate min_from_peak
start_time = pd.Timestamp("2024-05-02 09:36:00")
SMPS_df["min_from_peak"] = (SMPS_df["Datetime"] - start_time).dt.total_seconds() / 60

# %% RUN Load and process SMPS #/cm3 (mass concentration)
SMPS_path2 = (
    "./burn_data/smps/MH_apollo_bed_05010024_MassConc.xlsx"  # Corrected file path
)
SMPS_df2 = pd.read_excel(SMPS_path2, sheet_name="all_data")

# Combine 'Date' and 'Start Time' into a single datetime column
SMPS_df2["Datetime"] = pd.to_datetime(
    SMPS_df2["Date"].astype(str) + " " + SMPS_df2["Start Time"].astype(str)
)

# Filter data for the specific date (5/2/2024)
SMPS_df2 = SMPS_df2[SMPS_df2["Datetime"].dt.date == date_filter.date()]


# Define column ranges for summing
def sum_columns2(df, ranges):
    summed_data2 = {}
    for start, end in ranges:
        col_range2 = [
            col
            for col in df.columns
            if str(col).replace(".", "", 1).isdigit() and start <= float(col) <= end
        ]
        summed_data2[f"Ʃ{start}-{end}nm (µg/m³)"] = df[col_range2].sum(axis=1)
    return summed_data2


summed_data2 = sum_columns2(SMPS_df2, ranges)

# Calculate min_from_peak
SMPS_df2["min_from_peak"] = (SMPS_df2["Datetime"] - start_time).dt.total_seconds() / 60

# %% RUN to plot the AeroTrakK and additional data
# Define color scaling for data
color_scale_orange = [
    "#ff7f0e",
    "#fbad4c",
    "#fad289",
    "#fff3cc",
]  # Shades of orange for SMPS_df
color_scale_blue = [
    "#003e58",
    "#417a8e",
    "#7ebbc6",
    "#c3ffff",
]  # Shades of blue for SMPS_df2

# For visualization in Jupyter Notebook
output_notebook()

# Define the x-axis ranges
x_range_start = -60
x_range_end = 100
y_range_start = 0.001
y_range_end = 10**5

# Create a Bokeh figure with logarithmic y-axis and minutes on x-axis
p = figure(
    x_axis_label="Time since peak conc. after particle growth (minutes)",
    y_axis_label="Particulate Matter (number count)",
    y_axis_type="log",
    x_axis_type="linear",
    x_range=(x_range_start, x_range_end),
    y_range=(y_range_start, y_range_end),
    max_width=600,
    height=800,
)

# Plot SMPS_df data in shades of orange
for (label, data), color in zip(summed_data.items(), color_scale_orange):
    p.line(
        SMPS_df["min_from_peak"], data, legend_label=label, line_width=2, color=color
    )

# Add a secondary y-axis for mass concentration data (log scale)
p.extra_y_ranges = {
    "mass_conc": Range1d(start=0.001, end=10**5)
}  # Initial range to ensure visibility
p.add_layout(
    LogAxis(
        y_range_name="mass_conc", axis_label="Particulate Matter (mass concentration)"
    ),
    "right",
)

# Plot SMPS_df2 data on the right y-axis
for (label, data), color in zip(summed_data2.items(), color_scale_blue):
    p.line(
        SMPS_df2["min_from_peak"],
        data,
        legend_label=label,
        line_width=2,
        color=color,
        y_range_name="mass_conc",
    )

# Adjust the legend and axis labels
p.legend.location = "top_right"
p.legend.label_text_font = "Calibri"
p.legend.label_text_font_size = "12pt"
p.legend.orientation = "vertical"

p.axis.axis_label_text_font = "Calibri"
p.axis.axis_label_text_font_size = "12pt"
p.axis.axis_label_text_font_style = "normal"

p.title.align = "center"

# Show the plot
show(p)
