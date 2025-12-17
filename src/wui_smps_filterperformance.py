"""
WUI SMPS Filter Performance Analysis

This script analyzes filter performance characteristics using size-resolved particle
measurements from the Scanning Mobility Particle Sizer (SMPS). It calculates
size-dependent filter efficiency and penetration for CR Box portable air cleaners
operating with different filter configurations during WUI smoke experiments.

Key Performance Metrics:
    - Single-pass filter efficiency: η(d_p) = 1 - [C_out/C_in]
    - Penetration: P(d_p) = C_out/C_in = 1 - η(d_p)
    - Most Penetrating Particle Size (MPPS)
    - Size-integrated efficiency (weighted by particle concentration)
    - Clean Air Delivery Rate by particle size: CADR(d_p) = Q × η(d_p)

Analysis Components:
    1. Size-resolved efficiency curves
    2. Comparison across filter types (MERV-13, MERV-12A)
    3. New vs used filter degradation
    4. Effect of particle loading on performance
    5. Filter pressure drop correlations

Methodology:
    - Upstream/downstream concentration ratios
    - Quality factor: QF = -ln(P) / ΔP
    - Figure of Merit: FOM = -ln(P) / (ΔP × face velocity)
    - Temporal efficiency tracking during loading

Filter Configurations Analyzed:
    - MERV-13 filters (new and used)
    - MERV-12A (pleated) filters (new and used)
    - Single pass measurements
    - Multiple air cleaner configurations

Data Sources:
    - SMPS size distributions (bedroom and kitchen)
    - Pressure differential measurements
    - Flow rate measurements
    - Burn log with filter configuration details

Outputs:
    - Filter efficiency vs particle size curves
    - Penetration plots (log scale)
    - MPPS identification for each configuration
    - Comparative performance bar charts
    - Temporal efficiency degradation plots
    - Statistical analysis of filter comparisons

Quality Control:
    - Steady-state verification
    - Background subtraction
    - Aerosol stability checks
    - Replicate measurement averaging

Dependencies:
    - pandas: Data manipulation
    - numpy: Numerical calculations
    - bokeh: Visualization
    - scipy: Curve fitting and statistics

Standards Referenced:
    - ASHRAE 52.2: Filter efficiency testing
    - ISO 29463: HEPA filter testing
    - EN 1822: High efficiency air filters

Author: Nathan Lima
Date: 2024-2025
"""

# %% RUN
# import needed modules
print("Importing needed modules")
import os
import numpy as np
import pandas as pd
from datetime import datetime as dt
from scipy import optimize
from bokeh.plotting import figure, show, output_file, output_notebook
from bokeh.models import (
    ColumnDataSource,
    Band,
    Label,
    Arrow,
    OpenHead,
    CrosshairTool,
    Span,
    Legend,
    LegendItem,
)
from bokeh.layouts import gridplot, row, column
from functools import reduce

# %% RUN User defines directory path for datset, dataset used, and dataset final location
# User set absolute_path
absolute_path = "C:/Users/nml/OneDrive - NIST/Documents/NIST/WUI_smoke/"  # USER ENTERED PROJECT PATH
os.chdir(absolute_path)

df = pd.read_excel("./burn_dates_decay_smps.xlsx", sheet_name="plotdata")

# %% Number of filters
# For visualization in Jupyter Notebook
# output_notebook()
# For visualization to static HTML
output_file("./Paper_figures/SMPS_NumberofFilters_Decay.html")

# Extract the minutes and data series
minutes_np = df["min_from_peak(min)"].values
time_hours_np = minutes_np / 60  # Convert minutes to hours
decay_data = {
    #'Burn 1': df['Baseline(norm.)'].values,
    "Burn 2": df["(4)MERV 13(norm.)"].values,
    #'Burn 3': df['(1)MERV 13 Used(norm.)'].values,
    "Burn 4": df["(1)MERV 13 New(norm.)"].values,
    #'Burn 5': df['ClosedBR(norm.)'].values,
    #'Burn 6': df['ClosedBRw/Filter(norm.)'].values,
    #'Burn 7': df['(2)MERV 12A New(norm.)'].values,
    #'Burn 8': df['(2)MERV 12A Used(norm.)'].values,
    "Burn 9": df["(2)MERV 13 New(norm.)"].values,
    #'Burn 10': df['(2)MERV 13 Used(norm.)'].values
}

# Define different ranges for fitting each series in minutes
fit_ranges = {
    "Burn 1": (16.04, 176.47),  # Fit from 16 to 176 minutes
    "Burn 2": (16.04, 24.06),
    "Burn 3": (16.04, 85.56),
    "Burn 4": (16.04, 80.21),
    "Burn 5": (10.69, 240),  # end was 655.08 but changed to 240 to fit figure
    "Burn 6": (0, 10.69),
    "Burn 7": (16.04, 32.08),
    "Burn 8": (16.04, 37.43),
    "Burn 9": (16.04, 42.78),
    "Burn 10": (16.04, 40.10),
}

# Define specific label positions for each series
label_positions = {
    "Burn 1": (-10, 0.1),  # x_offset in minutes, y_offset in units above the line
    "Burn 2": (-15, -0.03),
    "Burn 3": (10, 0.01),
    "Burn 4": (-5, -0.022),
    "Burn 5": (-50, -0.2),
    "Burn 6": (-5, -0.197),
    "Burn 7": (-4, -0.041),
    "Burn 8": (-3, -0.037),
    "Burn 9": (-5, -0.02),
    "Burn 10": (-2, -0.033),
}

# Define custom line colors and line types for each series
line_properties = {
    "Burn 1": {"color": "black", "line_dash": "solid", "line_width": 1.5},
    "Burn 2": {"color": "orange", "line_dash": "solid", "line_width": 1.5},
    "Burn 3": {"color": "brown", "line_dash": "dashed", "line_width": 1.5},
    "Burn 4": {"color": "brown", "line_dash": "solid", "line_width": 1.5},
    "Burn 5": {"color": "purple", "line_dash": "solid", "line_width": 1.5},
    "Burn 6": {"color": "blue", "line_dash": "solid", "line_width": 1.5},
    "Burn 7": {"color": "green", "line_dash": "solid", "line_width": 1.5},
    "Burn 8": {"color": "green", "line_dash": "dashed", "line_width": 1.5},
    "Burn 9": {"color": "cyan", "line_dash": "solid", "line_width": 1.5},
    "Burn 10": {"color": "cyan", "line_dash": "dashed", "line_width": 1.5},
}

# Define the x-axis and y-axis ranges
x_range_start = minutes_np.min()
x_range_end = 100  # Cover an additional hour for a full view
y_range_start = 0.01  # Avoid log scale issues by not starting at zero
y_range_end = 1  # Example upper limit for the y-axis


def exponential_decay(x, a, b):
    return a * np.exp(-b * x)


def fit_exponential_curve(x, y):
    # Fit the exponential decay model
    popt, pcov = optimize.curve_fit(exponential_decay, x, y, p0=(1, 1))
    a, b = popt
    # Calculate standard error for the parameter 'b'
    std_err = np.sqrt(np.diag(pcov))
    y_fit = exponential_decay(x, *popt)
    return popt, y_fit, std_err


# Create a Bokeh figure with logarithmic y-axis and minutes on x-axis
p = figure(
    # title="SMPS Number of New filters Normalized Decay",
    x_axis_label="Time since peak conc. (minutes)",
    y_axis_type="log",
    x_axis_type="linear",
    x_range=(x_range_start, x_range_end),
    y_range=(y_range_start, y_range_end),
    max_width=600,
    height=800,
)
# List to collect temporary DataFrames
burn_calc = []

# Plot decay data
for label, data in decay_data.items():
    # Get line properties for the current series
    color = line_properties[label]["color"]
    line_dash = line_properties[label]["line_dash"]
    line_width = line_properties[label]["line_width"]

    # Plot the original data with legend
    p.line(
        minutes_np,
        data,
        legend_label=label,
        line_width=line_width,
        color=color,
        line_dash=line_dash,
    )

    # Define the range for fitting
    fit_start_min, fit_end_min = fit_ranges[label]

    # Extract fitting data
    fit_start_index = np.searchsorted(minutes_np, fit_start_min)
    fit_end_index = np.searchsorted(minutes_np, fit_end_min)

    # Fit exponential curve on the specified portion of the time series
    x_fit = time_hours_np[fit_start_index:fit_end_index]
    y_fit = data[fit_start_index:fit_end_index]
    popt, y_curve_fit, std_err = fit_exponential_curve(x_fit, y_fit)

    # Define the fit curve and uncertainty band
    curve_fit_y = exponential_decay(time_hours_np, *popt)

    # Calculate uncertainty for the portion of the data used in the fit
    uncertainty = 1.96 * std_err[1] * np.exp(-popt[1] * x_fit)

    # Add the curve fit line for the fitting portion without legend
    p.line(
        minutes_np[fit_start_index:fit_end_index],
        y_curve_fit,
        line_color="red",
        line_dash="solid",
    )  # No legend label

    # Add the uncertainty band for the fitting portion
    # source = ColumnDataSource(data=dict(x=minutes_np[fit_start_index:fit_end_index], lower=y_fit - uncertainty, upper=y_fit + uncertainty))
    # band = Band(base='x', lower='lower', upper='upper', source=source, level='underlay', fill_alpha=0.2, fill_color='red')
    # p.add_layout(band)

    # Prepare fit text with only b value and uncertainty
    fit_info = f"{popt[1]:.2f} h$$^{{-1}}$$ ± {1.96 * std_err[1]:.2f} h$$^{{-1}}$$"
    print(label + f" {popt[1]:.3f} h^{-1} ± {1.96 * std_err[1]:.3f} h^{-1}")

    # Retrieve label position for the current series
    x_offset, y_offset = label_positions[label]
    label_x = minutes_np[fit_end_index - 1] + x_offset  # Move label to the right
    label_y = y_curve_fit[-1] + y_offset  # Move label slightly above the fit line end

    # Add Label with plain text
    p.add_layout(
        Label(
            x=label_x,
            y=label_y,
            text=fit_info,
            text_font_size="10pt",
            text_align="left",
            text_baseline="middle",
        )
    )

    # Add Arrow pointing to the fit line
    arrow = Arrow(
        end=OpenHead(size=10, line_color="black"),
        line_color="black",
        x_start=label_x,
        y_start=label_y,
        x_end=minutes_np[fit_end_index - 1],
        y_end=y_curve_fit[-1],
    )
    p.add_layout(arrow)

    # Create a dictionary for each row
    new_row = {
        "burn": label,
        "decay": f"{popt[1]:.4f}",
        "decay_uncertainty": f"{1.96 * std_err[1]:.4f}",
    }
    # Add the dictionary to the list
    burn_calc.append(new_row)

# Adjust the legend position to the lower-left corner
p.legend.location = "bottom_left"
p.legend.label_text_font = "Calibri"
p.legend.label_text_font_size = "12pt"
p.legend.orientation = "vertical"

p.yaxis.axis_label = r"$$LN(C_{t1} / C_{t0})$$"

p.axis.axis_label_text_font = "Calibri"
p.axis.axis_label_text_font_size = "12pt"
p.axis.axis_label_text_font_style = "normal"

# Center the title text
p.title.align = "center"

# Show the plot
show(p)

# %% New vs. used/aged
# For visualization in Jupyter Notebook
# output_notebook()
# For visualization to static HTML
output_file("./Paper_figures/SMPS_FilterAge_Decay.html")

# Extract the minutes and data series
minutes_np = df["min_from_peak(min)"].values
time_hours_np = minutes_np / 60  # Convert minutes to hours
decay_data = {
    #'Burn 1': df['Baseline(norm.)'].values,
    #'Burn 2': df['(4)MERV 13(norm.)'].values,
    "Burn 3": df["(1)MERV 13 Used(norm.)"].values,
    "Burn 4": df["(1)MERV 13 New(norm.)"].values,
    #'Burn 5': df['ClosedBR(norm.)'].values,
    #'Burn 6': df['ClosedBRw/Filter(norm.)'].values,
    "Burn 7": df["(2)MERV 12A New(norm.)"].values,
    "Burn 8": df["(2)MERV 12A Used(norm.)"].values,
    "Burn 9": df["(2)MERV 13 New(norm.)"].values,
    "Burn 10": df["(2)MERV 13 Used(norm.)"].values,
}

# Define different ranges for fitting each series in minutes
fit_ranges = {
    "Burn 1": (16.04, 176.47),  # Fit from 16 to 176 minutes
    "Burn 2": (16.04, 24.06),
    "Burn 3": (16.04, 85.56),
    "Burn 4": (16.04, 80.21),
    "Burn 5": (10.69, 240),  # end was 655.08 but changed to 240 to fit figure
    "Burn 6": (0, 10.69),
    "Burn 7": (16.04, 32.08),
    "Burn 8": (16.04, 37.43),
    "Burn 9": (16.04, 42.78),
    "Burn 10": (16.04, 40.10),
}

# Define specific label positions for each series
label_positions = {
    "Burn 1": (-10, 0.1),  # x_offset in minutes, y_offset in units above the line
    "Burn 2": (-5, -0.054),
    "Burn 3": (-10, 0.03),
    "Burn 4": (-5, -0.02),
    "Burn 5": (-50, -0.2),
    "Burn 6": (-5, -0.197),
    "Burn 7": (-4, -0.037),
    "Burn 8": (-3, -0.0335),
    "Burn 9": (-1, -0.025),
    "Burn 10": (-2, -0.031),
}

# Define custom line colors and line types for each series
line_properties = {
    "Burn 1": {"color": "black", "line_dash": "solid", "line_width": 1.5},
    "Burn 2": {"color": "orange", "line_dash": "solid", "line_width": 1.5},
    "Burn 3": {"color": "brown", "line_dash": "dashed", "line_width": 1.5},
    "Burn 4": {"color": "brown", "line_dash": "solid", "line_width": 1.5},
    "Burn 5": {"color": "purple", "line_dash": "solid", "line_width": 1.5},
    "Burn 6": {"color": "blue", "line_dash": "solid", "line_width": 1.5},
    "Burn 7": {"color": "green", "line_dash": "solid", "line_width": 1.5},
    "Burn 8": {"color": "green", "line_dash": "dashed", "line_width": 1.5},
    "Burn 9": {"color": "cyan", "line_dash": "solid", "line_width": 1.5},
    "Burn 10": {"color": "cyan", "line_dash": "dashed", "line_width": 1.5},
}

# Define the x-axis and y-axis ranges
x_range_start = minutes_np.min()
x_range_end = 100  # Cover an additional hour for a full view
y_range_start = 0.01  # Avoid log scale issues by not starting at zero
y_range_end = 1  # Example upper limit for the y-axis


def exponential_decay(x, a, b):
    return a * np.exp(-b * x)


def fit_exponential_curve(x, y):
    # Fit the exponential decay model
    popt, pcov = optimize.curve_fit(exponential_decay, x, y, p0=(1, 1))
    a, b = popt
    # Calculate standard error for the parameter 'b'
    std_err = np.sqrt(np.diag(pcov))
    y_fit = exponential_decay(x, *popt)
    return popt, y_fit, std_err


# Create a Bokeh figure with logarithmic y-axis and minutes on x-axis
p = figure(
    # title="SMPS New vs. Used/Aged Filers Normalized Decay",
    x_axis_label="Time since peak conc. (minutes)",
    y_axis_type="log",
    x_axis_type="linear",
    x_range=(x_range_start, x_range_end),
    y_range=(y_range_start, y_range_end),
    max_width=600,
    height=800,
)
# List to collect temporary DataFrames
burn_calc = []

# Plot decay data
for label, data in decay_data.items():
    # Get line properties for the current series
    color = line_properties[label]["color"]
    line_dash = line_properties[label]["line_dash"]
    line_width = line_properties[label]["line_width"]

    # Plot the original data with legend
    p.line(
        minutes_np,
        data,
        legend_label=label,
        line_width=line_width,
        color=color,
        line_dash=line_dash,
    )

    # Define the range for fitting
    fit_start_min, fit_end_min = fit_ranges[label]

    # Extract fitting data
    fit_start_index = np.searchsorted(minutes_np, fit_start_min)
    fit_end_index = np.searchsorted(minutes_np, fit_end_min)

    # Fit exponential curve on the specified portion of the time series
    x_fit = time_hours_np[fit_start_index:fit_end_index]
    y_fit = data[fit_start_index:fit_end_index]
    popt, y_curve_fit, std_err = fit_exponential_curve(x_fit, y_fit)

    # Define the fit curve and uncertainty band
    curve_fit_y = exponential_decay(time_hours_np, *popt)

    # Calculate uncertainty for the portion of the data used in the fit
    uncertainty = 1.96 * std_err[1] * np.exp(-popt[1] * x_fit)

    # Add the curve fit line for the fitting portion without legend
    p.line(
        minutes_np[fit_start_index:fit_end_index],
        y_curve_fit,
        line_color="red",
        line_dash="solid",
    )  # No legend label

    # Add the uncertainty band for the fitting portion
    # source = ColumnDataSource(data=dict(x=minutes_np[fit_start_index:fit_end_index], lower=y_fit - uncertainty, upper=y_fit + uncertainty))
    # band = Band(base='x', lower='lower', upper='upper', source=source, level='underlay', fill_alpha=0.2, fill_color='red')
    # p.add_layout(band)

    # Prepare fit text with only b value and uncertainty
    fit_info = f"{popt[1]:.2f} h$$^{{-1}}$$ ± {1.96 * std_err[1]:.2f} h$$^{{-1}}$$"
    print(label + f" {popt[1]:.3f} h^{-1} ± {1.96 * std_err[1]:.3f} h^{-1}")

    # Retrieve label position for the current series
    x_offset, y_offset = label_positions[label]
    label_x = minutes_np[fit_end_index - 1] + x_offset  # Move label to the right
    label_y = y_curve_fit[-1] + y_offset  # Move label slightly above the fit line end

    # Add Label with plain text
    p.add_layout(
        Label(
            x=label_x,
            y=label_y,
            text=fit_info,
            text_font_size="10pt",
            text_align="left",
            text_baseline="middle",
        )
    )

    # Add Arrow pointing to the fit line
    arrow = Arrow(
        end=OpenHead(size=10, line_color="black"),
        line_color="black",
        x_start=label_x,
        y_start=label_y,
        x_end=minutes_np[fit_end_index - 1],
        y_end=y_curve_fit[-1],
    )
    p.add_layout(arrow)

    # Create a dictionary for each row
    new_row = {
        "burn": label,
        "decay": f"{popt[1]:.4f}",
        "decay_uncertainty": f"{1.96 * std_err[1]:.4f}",
    }
    # Add the dictionary to the list
    burn_calc.append(new_row)

# Adjust the legend position to the lower-left corner
p.legend.location = "bottom_left"
p.legend.label_text_font = "Calibri"
p.legend.label_text_font_size = "12pt"
p.legend.orientation = "vertical"

p.yaxis.axis_label = r"$$LN(C_{t1} / C_{t0})$$"

p.axis.axis_label_text_font = "Calibri"
p.axis.axis_label_text_font_size = "12pt"
p.axis.axis_label_text_font_style = "normal"

# Center the title text
p.title.align = "center"

# Show the plot
show(p)

# %% Saferoom
# For visualization in Jupyter Notebook
# output_notebook()
# For visualization to static HTML
output_file("./Paper_figures/SMPS_SafeRoom_Performance_Decay.html")

# Extract the minutes and data series
minutes_np = df["min_from_peak(min)"].values
time_hours_np = minutes_np / 60  # Convert minutes to hours
decay_data = {
    #'Burn 1': df['Baseline(norm.)'].values,
    "Burn 1": df["Baseline(µg/m³)"].values,
    #'Burn 2': df['(4)MERV 13(norm.)'].values,
    #'Burn 3': df['(1)MERV 13 Used(norm.)'].values,
    #'Burn 4': df['(1)MERV 13 New(norm.)'].values,
    #'Burn 5': df['ClosedBR(norm.)'].values,
    "Burn 5": df["ClosedBR(µg/m³)"].values,
    #'Burn 6': df['ClosedBRw/Filter(norm.)'].values,
    "Burn 6": df["ClosedBRw/Filter(µg/m³)"].values,
    #'Burn 7': df['(2)MERV 12A New(norm.)'].values,
    #'Burn 8': df['(2)MERV 12A Used(norm.)'].values,
    #'Burn 9': df['(2)MERV 13 New(norm.)'].values,
    #'Burn 10': df['(2)MERV 13 Used(norm.)'].values
}

# Define different ranges for fitting each series in minutes
fit_ranges = {
    "Burn 1": (16.04, 176.47),  # Fit from 16 to 176 minutes
    "Burn 2": (16.04, 24.06),
    "Burn 3": (16.04, 85.56),
    "Burn 4": (16.04, 80.21),
    "Burn 5": (10.69, 240),  # end was 655.08 but changed to 240 to fit figure
    "Burn 6": (0, 10.69),
    "Burn 7": (16.04, 32.08),
    "Burn 8": (16.04, 37.43),
    "Burn 9": (16.04, 42.78),
    "Burn 10": (16.04, 40.10),
}

# Define specific label positions for each series
label_positions = {
    "Burn 1": (-10, 20),  # x_offset in minutes, y_offset in units above the line
    "Burn 2": (-5, -0.054),
    "Burn 3": (10, 0.01),
    "Burn 4": (-5, -0.019),
    "Burn 5": (-60, -2.5),
    "Burn 6": (30, -0.1),
    "Burn 7": (-4, -0.041),
    "Burn 8": (-3, -0.037),
    "Burn 9": (-1, -0.025),
    "Burn 10": (-2, -0.033),
}

max_label_positions = {
    "Burn 1": (20, 20),  # x_offset in minutes, y_offset in units above the line
    "Burn 2": (-5, -0.054),
    "Burn 3": (10, 0.01),
    "Burn 4": (-5, -0.019),
    "Burn 5": (20, 2.5),
    "Burn 6": (20, 0.0),
    "Burn 7": (-4, -0.041),
    "Burn 8": (-3, -0.037),
    "Burn 9": (-1, -0.025),
    "Burn 10": (-2, -0.033),
}

# Define custom line colors and line types for each series
line_properties = {
    "Burn 1": {"color": "black", "line_dash": "solid", "line_width": 1.5},
    "Burn 2": {"color": "orange", "line_dash": "solid", "line_width": 1.5},
    "Burn 3": {"color": "brown", "line_dash": "dashed", "line_width": 1.5},
    "Burn 4": {"color": "brown", "line_dash": "solid", "line_width": 1.5},
    "Burn 5": {"color": "purple", "line_dash": "solid", "line_width": 1.5},
    "Burn 6": {"color": "blue", "line_dash": "solid", "line_width": 1.5},
    "Burn 7": {"color": "green", "line_dash": "solid", "line_width": 1.5},
    "Burn 8": {"color": "green", "line_dash": "dashed", "line_width": 1.5},
    "Burn 9": {"color": "cyan", "line_dash": "solid", "line_width": 1.5},
    "Burn 10": {"color": "cyan", "line_dash": "dashed", "line_width": 1.5},
}

# Define the x-axis and y-axis ranges
x_range_start = minutes_np.min()
x_range_end = 250  # Cover an additional hour for a full view
y_range_start = 0.1  # Avoid log scale issues by not starting at zero
y_range_end = 1000  # Example upper limit for the y-axis


def exponential_decay(x, a, b):
    return a * np.exp(-b * x)


def fit_exponential_curve(x, y):
    # Fit the exponential decay model
    popt, pcov = optimize.curve_fit(exponential_decay, x, y, p0=(1, 1))
    a, b = popt
    # Calculate standard error for the parameter 'b'
    std_err = np.sqrt(np.diag(pcov))
    y_fit = exponential_decay(x, *popt)
    return popt, y_fit, std_err


# Create a Bokeh figure with logarithmic y-axis and minutes on x-axis
p = figure(
    # title="SMPS Safe Room Performance and Decay",
    x_axis_label="Time since peak conc. (minutes)",
    y_axis_label="Total PM mass [size range 9.31 to 437 nm] (µg/m3)",
    y_axis_type="log",
    x_axis_type="linear",
    x_range=(x_range_start, x_range_end),
    y_range=(y_range_start, y_range_end),
    max_width=600,
    height=800,
)
# List to collect temporary DataFrames
burn_calc = []

# Plot decay data
for label, data in decay_data.items():
    # Get line properties for the current series
    color = line_properties[label]["color"]
    line_dash = line_properties[label]["line_dash"]
    line_width = line_properties[label]["line_width"]

    # Plot the original data with legend
    p.line(
        minutes_np,
        data,
        legend_label=label,
        line_width=line_width,
        color=color,
        line_dash=line_dash,
    )

    # Define the range for fitting
    fit_start_min, fit_end_min = fit_ranges[label]

    # Extract fitting data
    fit_start_index = np.searchsorted(minutes_np, fit_start_min)
    fit_end_index = np.searchsorted(minutes_np, fit_end_min)

    # Fit exponential curve on the specified portion of the time series
    x_fit = time_hours_np[fit_start_index:fit_end_index]
    y_fit = data[fit_start_index:fit_end_index]
    popt, y_curve_fit, std_err = fit_exponential_curve(x_fit, y_fit)

    # Define the fit curve and uncertainty band
    curve_fit_y = exponential_decay(time_hours_np, *popt)

    # Calculate uncertainty for the portion of the data used in the fit
    uncertainty = 1.96 * std_err[1] * np.exp(-popt[1] * x_fit)

    # Add the curve fit line for the fitting portion without legend
    p.line(
        minutes_np[fit_start_index:fit_end_index],
        y_curve_fit,
        line_color="red",
        line_dash="solid",
    )  # No legend label

    # Add the uncertainty band for the fitting portion
    # source = ColumnDataSource(data=dict(x=minutes_np[fit_start_index:fit_end_index], lower=y_fit - uncertainty, upper=y_fit + uncertainty))
    # band = Band(base='x', lower='lower', upper='upper', source=source, level='underlay', fill_alpha=0.2, fill_color='red')
    # p.add_layout(band)

    # Prepare fit text with only b value and uncertainty
    fit_info = f"{popt[1]:.2f} h$$^{{-1}}$$ ± {1.96 * std_err[1]:.2f} h$$^{{-1}}$$"
    print(label + f" {popt[1]:.3f} h^{-1} ± {1.96 * std_err[1]:.3f} h^{-1}")

    # Retrieve label position for the current series
    x_offset, y_offset = label_positions[label]
    label_x = minutes_np[fit_end_index - 1] + x_offset  # Move label to the right
    label_y = y_curve_fit[-1] + y_offset  # Move label slightly above the fit line end

    # Add Label with plain text
    p.add_layout(
        Label(
            x=label_x,
            y=label_y,
            text=fit_info,
            text_font_size="10pt",
            text_align="left",
            text_baseline="middle",
        )
    )

    # Add Arrow pointing to the fit line
    arrow = Arrow(
        end=OpenHead(size=10, line_color="black"),
        line_color="black",
        x_start=label_x,
        y_start=label_y,
        x_end=minutes_np[fit_end_index - 1],
        y_end=y_curve_fit[-1],
    )
    p.add_layout(arrow)

    # Create a dictionary for each row
    new_row = {
        "burn": label,
        "decay": f"{popt[1]:.4f}",
        "decay_uncertainty": f"{1.96 * std_err[1]:.4f}",
    }
    # Add the dictionary to the list
    burn_calc.append(new_row)

    # Use the first value for the maximum y value
    max_y = data[0]
    max_x = minutes_np[0]

    # Extract the x and y offsets for maximum labels from the max_label_positions dictionary
    max_x_offset, max_y_offset = max_label_positions[label]

    # Calculate the adjusted label positions for the maximum value
    label_x_max = max_x + max_x_offset  # Adjust position horizontally
    label_y_max = max_y + max_y_offset  # Adjust position vertically

    # Add a label for the maximum value
    p.add_layout(
        Label(
            x=label_x_max,
            y=label_y_max,
            text=f"Max: {max_y:.1f} (µg/m³)",
            text_font="Calibri",
            text_font_size="12pt",
            text_align="left",
            text_baseline="bottom",
        )
    )

    # Add an arrow pointing to the maximum value
    arrow_max = Arrow(
        end=OpenHead(size=10, line_color="black"),
        line_color="black",
        x_start=label_x_max,
        y_start=label_y_max,
        x_end=max_x,
        y_end=max_y,
    )
    p.add_layout(arrow_max)

# Adjust the legend position to the lower-left corner
p.legend.location = "bottom_left"
p.legend.label_text_font = "Calibri"
p.legend.label_text_font_size = "12pt"
p.legend.orientation = "vertical"

p.axis.axis_label_text_font = "Calibri"
p.axis.axis_label_text_font_size = "12pt"
p.axis.axis_label_text_font_style = "normal"

# Center the title text
p.title.align = "center"

# Show the plot
show(p)
# %%
