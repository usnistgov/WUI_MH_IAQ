"""
WUI AeroTrak vs SMPS Comparison Analysis

This script compares particle size distributions and mass concentrations measured
by AeroTrak optical particle counters and the SMPS (Scanning Mobility Particle Sizer)
during wildland-urban interface smoke experiments. It evaluates agreement between
optical and mobility-based sizing methods.

Comparison Features:
    - Size distribution overlap region (~300-500 nm)
    - Number concentration comparison in overlapping size bins
    - Mass concentration comparison for PM fractions
    - Temporal correlation analysis
    - Sizing accuracy assessment

Instrument Characteristics:
    AeroTrak (Optical):
        - Size range: 0.3-25 µm (optical diameter)
        - 6 size channels
        - 1-minute time resolution
        - Light scattering detection

    SMPS (Electrical Mobility):
        - Size range: 10-500 nm (mobility diameter)
        - ~100 size bins per scan
        - 3-5 minute scan time
        - Electrical mobility classification

Analysis Methods:
    1. Diameter conversion (optical ↔ mobility)
    2. Concentration normalization (dN/dlog Dp)
    3. Cross-correlation in overlap region
    4. Integrated mass concentration comparison

Key Metrics:
    - Agreement ratio: C_AeroTrak / C_SMPS
    - Correlation coefficient (r)
    - Systematic bias assessment
    - Size-dependent agreement analysis

Outputs:
    - Overlaid size distribution plots
    - Agreement ratio time series
    - Scatter plots with regression
    - Summary statistics tables

Dependencies:
    - pandas: Data manipulation
    - numpy: Numerical operations
    - bokeh: Visualization

Author: Nathan Lima
Date: 2024-2025
"""

# %% RUN
# Import needed modules
print("Importing needed modules")
import os
import numpy as np
import pandas as pd
from functools import reduce
from bokeh.plotting import figure, show, output_notebook, output_file
from bokeh.models import Range1d

# Set the project directory path
absolute_path = "C:/Users/nml/OneDrive - NIST/Documents/NIST/WUI_smoke/"  # USER ENTERED PROJECT PATH
os.chdir(absolute_path)

# User inputs for burn and burn_date
burn = input("Enter the burn number (e.g., 10 for burn10): ")
burn_date = "2024-05-31"

# %% AeroTrak Data Processing
# Define columns to drop from the AeroTrak dataset
columns_to_drop = [
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


# Function to rename columns based on sheet name
def rename_columns(sheet_name):
    return {
        "Ch1 Diff (#)": f"{sheet_name} Ch1 0.3µm (#/cm³)",
        "Ch2 Diff (#)": f"{sheet_name} Ch2 0.5µm (#/cm³)",
        "Ch3 Diff (#)": f"{sheet_name} Ch3 1.0µm (#/cm³)",
        "Ch4 Diff (#)": f"{sheet_name} Ch4 3.0µm (#/cm³)",
        "Ch5 Diff (#)": f"{sheet_name} Ch5 5.0µm (#/cm³)",
        "Ch6 Diff (#)": f"{sheet_name} Ch6 10.0µm (#/cm³)",
    }


# Load data only for the specified burn
sheet_name = f"burn{burn}"
df = pd.read_excel("./burn_dates_decay_aerotraks_bedroom.xlsx", sheet_name=sheet_name)
df = df.drop(columns=columns_to_drop)

# Unit conversion from L to cubic centimeters (divide by 3000)
for col in df.columns:
    if "Diff" in col:  # Only process columns with 'Diff' in their name
        df[col] /= 3000

# Rename columns
df.rename(columns=rename_columns(sheet_name), inplace=True)

# Set the index and interpolate missing values
df = df.set_index("min_since_peak")
df = df.interpolate("index")
df["min_from_peak"] = df.index
df = df[df["min_from_peak"] >= -120]

# %% RUN Load and Process SMPS Data
SMPS_path = (
    f"./burn_data/smps/MH_apollo_bed_05312024_NumConc.xlsx"  # Use user-specified date
)
SMPS_df = pd.read_excel(SMPS_path, sheet_name="all_data")

# Combine 'Date' and 'Start Time' into a single datetime column
SMPS_df["Datetime"] = pd.to_datetime(
    SMPS_df["Date"].astype(str) + " " + SMPS_df["Start Time"].astype(str)
)

# Filter data for the specific date
date_filter = pd.Timestamp(burn_date)
SMPS_df = SMPS_df[SMPS_df["Datetime"].dt.date == date_filter.date()]


# Function to sum columns based on specified ranges
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


# Define the ranges for summing
ranges = [(9, 100), (101, 200), (201, 300), (300, float("inf"))]
summed_data = sum_columns(SMPS_df, ranges)

# Calculate min_from_peak for SMPS data
start_time = pd.Timestamp(f"{burn_date} 09:03:00")
SMPS_df["min_from_peak"] = (SMPS_df["Datetime"] - start_time).dt.total_seconds() / 60

# %% RUN Load and Process QuantAQ Data
csv_path = "./burn_data/quantaq/MOD-PM-00194-b0fc215029fa4852b926bc50b28fda5a.csv"
new_df = pd.read_csv(csv_path)

# Reverse the order of the data and parse datetime
new_df = new_df[::-1].reset_index(drop=True)
new_df["timestamp_local"] = pd.to_datetime(
    new_df["timestamp_local"].str.replace("T", " ").str.replace("Z", "")
)

# Calculate min_from_peak for QuantAQ data
new_df["min_from_peak"] = (
    new_df["timestamp_local"] - start_time - pd.Timedelta(minutes=5.5)
).dt.total_seconds() / 60

# Define the ranges for summing in QuantAQ data
bins = {
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
summed_new_data = {}
for label, cols in bins.items():
    summed_new_data[label] = new_df[cols].sum(axis=1)

# %% RUN Plotting AeroTrakK, SMPS, and QuantAQ Data
# Define color scales for different data types
color_scale = ["#ff0000", "#ff3333", "#ff6666", "#ff9999"]
new_data_colors = [
    "#080be5",
    "#0068ff",
    "#0093ff",
    "#00b4ff",
    "#00d1d2",
    "#00eb7f",
    "#29ff06",
]
aerotrak_colors = ["#f8aa39", "#f1ae58", "#e8b274", "#ddb78d", "#d0bba7", "#c0c0c0"]

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
    title=f"Burn {burn} Bedroom Two Particle Count Comparison",
    x_range=(-60, 120),  # Set the x-axis range from -60 to 120
    y_range=Range1d(start=1e-3, end=1e5),  # Set y-axis range from 0.001 to 100,000
)

# After plotting AeroTrakK data
p.line(
    [], [], legend_label="AeroTrak", color=None, line_width=0
)  # Add a dummy line for AeroTrak

for i, (label, color) in enumerate(zip(df.keys(), aerotrak_colors)):
    data = df[label]
    # Remove the units from the label
    clean_label = label.replace(" (#/cm³)", "").replace(f"burn{burn} ", "")
    p.line(
        df["min_from_peak"],
        data,
        legend_label=clean_label,
        color=color,
        line_dash="solid",
        line_width=1.5,
    )

# After plotting SMPS data
p.line(
    [], [], legend_label="SMPS", color=None, line_width=0
)  # Add a dummy line for SMPS

for idx, (label, series) in enumerate(summed_data.items()):
    # Remove the units from the label
    clean_label = label.replace(" (#/cm³)", "")
    p.line(
        SMPS_df["min_from_peak"],
        series,
        legend_label=clean_label,
        color=color_scale[idx % len(color_scale)],
        line_dash="dotted",
        line_width=1.5,
    )

# After plotting QuantAQ data
p.line(
    [], [], legend_label="QuantAQ", color=None, line_width=0
)  # Add a dummy line for QuantAQ

for idx, (label, series) in enumerate(summed_new_data.items()):
    # Remove the units from the label
    clean_label = label.replace(" (#/cm³)", "")
    p.line(
        new_df["min_from_peak"],
        series,
        legend_label=clean_label,
        color=new_data_colors[idx % len(new_data_colors)],
        line_dash="dashed",
        line_width=1.5,
    )

# Add a vertical line at x = -10
p.line(
    [-8, -8],
    [1e-3, 1e5],
    line_color="black",
    line_width=2,
    legend_label="CR Boxes Activation",
)

# Add a label for the vertical line
p.text(
    x=-10,
    y=1e4,  # Position the label vertically where you want it
    text=["CR Boxes Activation"],
    text_font_size="10pt",
    text_color="black",
    text_align="right",
    text_baseline="middle",
)

# Customize and show the plot
p.legend.location = "top_right"
p.legend.orientation = "vertical"

# Customize font properties
p.axis.axis_label_text_font = "Calibri"
p.axis.axis_label_text_font_size = "12pt"
p.axis.axis_label_text_font_style = "normal"
p.title.align = "center"

# Show the plot
# output_file('./Paper_figures/AeroTrak_SMPS_QuantAQ_comparison.html')
show(p)
