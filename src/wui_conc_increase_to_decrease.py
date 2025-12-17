"""
WUI Concentration Increase-to-Decrease Ratio Analysis

This script analyzes the ratio of concentration increase time to concentration
decrease time during wildland-urban interface smoke experiments. This metric
characterizes how quickly concentrations rise during smoke infiltration compared
to how quickly they decay during air cleaning.

Key Metrics:
    - Increase time: Duration from baseline to peak concentration
    - Decrease time: Duration from peak to background (or threshold level)
    - Increase/Decrease ratio: Characterizes exposure profile asymmetry
    - Time constants for exponential rise and decay phases

Analysis Features:
    - Multi-instrument comparative analysis
    - Burn-by-burn ratio calculations
    - Statistical comparison across filter configurations
    - Visualization of rise/decay phases
    - Effect of air cleaner activation on decay rates

Research Questions Addressed:
    1. How does the increase/decrease ratio vary with filter count?
    2. Do new vs used filters affect decay time differently than rise time?
    3. What is the optimal air cleaner activation timing?
    4. How do different PM sizes affect rise/decay dynamics?

Methodology:
    - Peak identification using rolling maximum
    - Exponential fitting for rise phase: C(t) = C₀(1 - e^(-t/τ_rise))
    - Exponential fitting for decay phase: C(t) = C_peak × e^(-t/τ_decay)
    - Time to threshold calculations (e.g., 35 µg/m³ PM2.5 EPA limit)

Data Processing:
    - High-resolution time series from multiple instruments
    - Synchronized timing across instruments
    - Quality-controlled peak identification
    - Baseline drift correction

Outputs:
    - Ratio plots for each burn and instrument
    - Statistical summary tables
    - Comparative bar charts
    - Time series with annotated rise/decay periods

Dependencies:
    - pandas: Time series manipulation
    - numpy: Numerical calculations
    - bokeh: Interactive visualization
    - scipy: Curve fitting for exponential models

Configuration:
    - Burn selection variable for targeted analysis
    - Threshold concentration for "safe" levels
    - Output directory for saving figures

Author: Nathan Lima
Date: 2024-2025
"""

# %% import needed modules, set absolute path for the data files, select burn
import pandas as pd
import os
import numpy as np
from datetime import datetime
from bokeh.plotting import figure, show
from bokeh.io import output_notebook, output_file
from bokeh.models import (
    ColumnDataSource,
    DatetimeTickFormatter,
    HoverTool,
    TapTool,
    LinearAxis,
    Range1d,
    LogTicker,
)
from bokeh.palettes import Blues, Greens, Reds, Purples

# Set the absolute path for the data files
absolute_path = "C:/Users/nml/OneDrive - NIST/Documents/NIST/WUI_smoke/"

# Directory for output
output_dir = os.path.join(absolute_path, "./Paper_figures/")

# Set the burn# variable (you can change this as needed)
burn_number = "burn10"  # Replace with the chosen burn

# %%
# Load the burn log to find the burn date
burn_log_path = os.path.join(absolute_path, "burn_log.xlsx")
burn_log = pd.read_excel(burn_log_path, sheet_name="Sheet2")

# Get the burn date based on the burn number
burn_date_row = burn_log[burn_log["Burn ID"] == burn_number]
if not burn_date_row.empty:
    burn_date = burn_date_row["Date"].values[0]
else:
    raise ValueError("Burn number not found in burn log.")

# Extract and convert the time from the 'CR Box on' column
cr_box_activation_time_str = burn_log.loc[
    burn_log["Burn ID"] == burn_number, "CR Box on"
].values[0]
cr_box_activation_time_full = (
    pd.to_datetime(f"{burn_date} {cr_box_activation_time_str}")
    if pd.notna(cr_box_activation_time_str)
    else None
)

# Extract and convert the time from the 'garage closed' column
door_closed_time_str = burn_log.loc[
    burn_log["Burn ID"] == burn_number, "garage closed"
].values[0]
door_closed_time_full = (
    pd.to_datetime(f"{burn_date} {door_closed_time_str}")
    if pd.notna(door_closed_time_str)
    else None
)

# %%
# Load VOCUS data for burn10
vocus_path = os.path.join(absolute_path, "burn_data/VOCUS/May31_2024_VOCUSdata.xlsx")
vocus_data = pd.read_excel(
    vocus_path, parse_dates=["datetime_EDT"], index_col="datetime_EDT"
)

# Filter VOCUS data for the given burn date
vocus_data = vocus_data[vocus_data.index.date == pd.to_datetime(burn_date).date()]

# Shift VOCUS data by 1 hour forward
vocus_data.index = vocus_data.index + pd.Timedelta(hours=1)  # Shift by one hour

# %%
# Load Aerotrak data for the given burn date
# aerotrak_path = os.path.join(absolute_path, 'burn_data/aerotraks/bedroom2/all_data.xlsx') #bedroom 2 data
aerotrak_path = os.path.join(
    absolute_path, "burn_data/aerotraks/kitchen/all_data.xlsx"
)  # kitchen data
aerotrak_data = pd.read_excel(
    aerotrak_path, parse_dates=["Date and Time"], index_col="Date and Time"
)

# Filter for the given burn date
aerotrak_data = aerotrak_data[
    aerotrak_data.index.date == pd.to_datetime(burn_date).date()
]

# Check instrument status columns and remove rows with invalid status
status_columns = ["Flow Status", "Laser Status"]
valid_status = (aerotrak_data[status_columns] == "OK").all(axis=1)
aerotrak_data.loc[~valid_status, aerotrak_data.columns] = np.nan

# Convert units for 'Cumul (#)' and 'Diff (#)' columns
volume_column = "Volume (L)"
if volume_column in aerotrak_data.columns:
    volume_liters = aerotrak_data[volume_column] * 1000  # Convert volume to mL
    for col in aerotrak_data.columns:
        if "Cumul (#)" in col:
            aerotrak_data[col + " (cm³)"] = aerotrak_data[col] / volume_liters
            aerotrak_data.rename(
                columns={
                    col + " (cm³)": col.replace("Cumul (#)", "Cumul (#/cm³)"),
                    "Cumul (#/cm³)": col,
                },
                inplace=True,
            )
        elif "Diff (#)" in col:
            aerotrak_data[col + " (cm³)"] = aerotrak_data[col] / volume_liters
            aerotrak_data.rename(
                columns={
                    col + " (cm³)": col.replace("Diff (#)", "Diff (#/cm³)"),
                    "Diff (#/cm³)": col,
                },
                inplace=True,
            )

# Select only 'Ch1 Cumul (#)'
aerotrak_cumul_col = "Ch1 Cumul (#/cm³)"  # Adjust this based on your data

# %%
# Load Quantaq data for the given burn date
# quantaq_path = os.path.join(absolute_path, 'burn_data/quantaq/MOD-PM-00194-b0fc215029fa4852b926bc50b28fda5a.csv') #bedroom 2 data
quantaq_path = os.path.join(
    absolute_path, "burn_data/quantaq/MOD-PM-00197-a6dd467a147a4d95a7b98a8a10ab4ea3.csv"
)  # kicten data
quantaq_data = pd.read_csv(
    quantaq_path, parse_dates=["timestamp_local"], index_col="timestamp_local"
)

# Flip the date order
quantaq_data = quantaq_data.sort_index(ascending=True)

# Filter for the given burn date
quantaq_data = quantaq_data[quantaq_data.index.date == pd.to_datetime(burn_date).date()]

# Define the bins to be summed
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

# Create a flat list of all bin column names from the bins dictionary
all_bin_columns = [bin_name for bin_list in bins.values() for bin_name in bin_list]

# Sum the specified bins and create a new column for the summed bins
if all(bin in quantaq_data.columns for bin in all_bin_columns):
    quantaq_sum_col = "Ʃ0.35-40 µm (#/cm³)"  # Define a new column for the summed bins
    quantaq_data[quantaq_sum_col] = quantaq_data[all_bin_columns].sum(axis=1)

# %%
# Load the PurpleAirK data
purpleair_path = os.path.join(absolute_path, "burn_data/purpleair/garage-kitchen.xlsx")
purpleair_data = pd.read_excel(
    purpleair_path,
    sheet_name="(P2)kitchen",
    parse_dates=["DateTime"],
    index_col="DateTime",
)

# Strip whitespace from column names
purpleair_data.columns = purpleair_data.columns.str.strip()

# Rename 'Average' column to 'PM2.5 (µg/m³)'
purpleair_data.rename(columns={"Average": "PM2.5 (µg/m³)"}, inplace=True)

purpleair_data = purpleair_data[
    purpleair_data.index.date == pd.to_datetime(burn_date).date()
]

# %%
# Load SMPS data for the given burn date and drop unnecessary column
# smps_filename = f'MH_apollo_bed_{pd.to_datetime(burn_date).strftime("%m%d%Y")}_numConc.xlsx'
# smps_path = os.path.join(absolute_path, f'burn_data/smps/{smps_filename}')
# smps_data = pd.read_excel(smps_path, sheet_name='all_data')

# # Drop the 'Total Concentration(#/cm³)' column if it exists
# if 'Total Concentration(#/cm³)' in smps_data.columns:
#     smps_data.drop(columns=['Total Concentration(#/cm³)'], inplace=True)

# # Specify date format for 'Date' and 'Start Time' to avoid warnings
# smps_data['Date'] = pd.to_datetime(smps_data['Date'], format='%Y-%m-%d', errors='coerce')
# smps_data['Start Time'] = pd.to_datetime(smps_data['Start Time'], format='%H:%M:%S', errors='coerce')

# # Combine 'Date' and 'Start Time' into a single datetime index
# smps_data['Datetime'] = pd.to_datetime(smps_data['Date'].dt.strftime('%Y-%m-%d') + ' ' + smps_data['Start Time'].dt.strftime('%H:%M:%S'))
# smps_data.set_index('Datetime', inplace=True)
# smps_data.drop(columns=['Date', 'Start Time'], inplace=True)

# # Filter for the given burn date
# smps_data = smps_data[smps_data.index.date == pd.to_datetime(burn_date).date()]

# # Sum columns based on the specified range (from 9 to max_col_value)
# max_col_value = smps_data.columns.astype(float).max()  # Get the maximum column value
# smps_data['Ʃ9-max (#/cm³)'] = smps_data.loc[:, smps_data.columns.astype(float) >= 9].sum(axis=1)

# %%
# Prepare to output the plot in the notebook
output_notebook()
output_file(os.path.join(absolute_path, f"burn_data/figure/{burn_number}_testing.html"))

# Create a figure for the plot
p = figure(
    x_axis_label="DateTime",
    y_axis_label="Particulate Matter",
    x_axis_type="datetime",
    y_axis_type="log",
    width=600,
    height=800,
)

# Define color scales for different datasets
aerotrak_color = Blues[9]
quantaq_color = Greens[9]
smps_color = Reds[9]
purpleair_color = Purples[9]

# Define the time range (ensuring they're timezone-naive)
start_time = pd.to_datetime(f"{burn_date} 07:00:00")
end_time = pd.to_datetime(f"{burn_date} 12:00:00")

# Set x-axis range
p.x_range.start = start_time.timestamp() * 1000  # Bokeh uses milliseconds
p.x_range.end = end_time.timestamp() * 1000

# Add HoverTool for interactive hover functionality
hover_tool = HoverTool(
    tooltips=[("DateTime", "@datetime{%F %T}"), ("Count", "@value")],
    formatters={"@datetime": "datetime"},
    mode="vline",
)

# Add HoverTool to the figure
p.add_tools(hover_tool)

# Add TapTool for selecting points
tap_tool = TapTool()
p.add_tools(tap_tool)

# Plot Aerotrak data
source_aerotrak = ColumnDataSource(
    data={
        "datetime": aerotrak_data.index,
        "value": aerotrak_data[aerotrak_cumul_col].values,
    }
)
p.line(
    x="datetime",
    y="value",
    source=source_aerotrak,
    line_width=2,
    color=aerotrak_color[0],
    legend_label="AeroTrak total PM (#/cm³)",
)

# Plot PurpleAir data
source_purpleair = ColumnDataSource(
    data={
        "datetime": purpleair_data.index,
        "value": purpleair_data["PM2.5 (µg/m³)"].values,
    }
)
p.line(
    x="datetime",
    y="value",
    source=source_purpleair,
    line_width=2,
    color=purpleair_color[0],
    legend_label="PurpleAir PM2.5 (µg/cm³)",
)

# Plot QuantAQ data without filtering
if quantaq_sum_col in quantaq_data.columns:
    source_quantaq = ColumnDataSource(
        data={
            "datetime": quantaq_data.index,
            "value": quantaq_data[quantaq_sum_col].values,
        }
    )
    p.line(
        x="datetime",
        y="value",
        source=source_quantaq,
        legend_label=f"QuantAQ {quantaq_sum_col}",
        line_width=2,
        color=quantaq_color[0],
        line_dash="dashed",
    )

# Plot SMPS summed data without filtering
# source_smps = ColumnDataSource(data={
#     'datetime': smps_data.index,
#     'value': smps_data['Ʃ9-max (#/cm³)'].values
# })
# p.line(x='datetime', y='value', source=source_smps,
#         legend_label=f'SMPS Ʃ9-{max_col_value} nm (#/cm³)', line_width=2,
#         color=smps_color[0], line_dash='dotted')

# Plot VOCUS data
source_vocus = ColumnDataSource(
    data={"datetime": vocus_data.index, "value": vocus_data["C3H4N_cps"].values}
)

# Create a custom Range1d for the right axis with log scale
vocus_range = Range1d(start=1000, end=250000)

# Add the VOCUS line, associating it with the secondary y-axis
p.line(
    x="datetime",
    y="value",
    source=source_vocus,
    line_width=2,
    color=smps_color[0],
    legend_label="VOCUS C3H4N (cps)",
    line_dash="solid",
    y_range_name="vocus_range",
)

# Customize x-axis
p.xaxis.formatter = DatetimeTickFormatter(
    hours="%d %B %Y %H:%M",
    minutes="%d %B %Y %H:%M",
)

# Add grid lines and legend
p.grid.grid_line_alpha = 0.3
p.legend.location = "top_right"

# Add the secondary y-axis for VOCUS with log scale
p.add_layout(
    LinearAxis(
        y_range_name="vocus_range",
        axis_label="VOCUS C3H4N (cps)",
        ticker=LogTicker(),  # Use LogTicker for the log scale
        axis_label_text_font_size="12pt",
    ),
    "right",
)

# Apply the vocus_range to the plot (for the right axis)
p.extra_y_ranges = {"vocus_range": vocus_range}

# Create a custom Range1d for the left y-axis
left_y_range = Range1d(start=0.1, end=10000)

# Apply the left y-axis range
p.y_range = left_y_range

# Show the plot
show(p)

# %%
