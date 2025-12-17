"""
WUI CADR Calculation and Analysis: Comprehensive Multi-Instrument Processing

This script is the main analysis pipeline for calculating Clean Air Delivery Rates
(CADR) for portable air cleaners tested during wildland-urban interface (WUI) smoke
experiments. It processes data from multiple particle measurement instruments and
generates decay curve analyses across all experimental burns.

Key Features:
    - Multi-instrument data processing (AeroTrak, DustTrak, PurpleAir, QuantAQ, SMPS)
    - Automated baseline correction and quality control
    - Exponential decay fitting with uncertainty quantification
    - CADR calculation from decay rates: CADR = λ × V (where V is room volume)
    - Comprehensive visualization with interactive Bokeh plots
    - Batch processing across all burns (burn2-burn10)

Methodology:
    - Decay region identification based on CR Box activation timing
    - Exponential model: C(t) = C₀ × exp(-λt) + C_background
    - Relative Standard Deviation (RSD) filtering for quality assurance
    - 95% confidence intervals calculated using curve_fit covariance
    - Instrument-specific time shifts for synchronization

Instrument Configurations:
    - AeroTrakB: Bedroom location, PM size fractions
    - AeroTrakK: Kitchen location, PM size fractions
    - DustTrak: PM1, PM2.5, PM4, PM10, PM15
    - PurpleAir: PM2.5 (Burns 6-10)
    - QuantAQ: PM1, PM2.5, PM10 (Burns 4-10)
    - SMPS: Size-resolved particle distributions

Data Processing Steps:
    1. Load and clean instrument data
    2. Apply instrument-specific time shifts
    3. Baseline correction using pre-experiment measurements
    4. Filter data to decay region (CR Box on to end)
    5. Fit exponential decay curves
    6. Calculate CADR with uncertainty propagation
    7. Generate visualization plots

Outputs:
    - CADR values with uncertainties for each instrument and burn
    - Interactive HTML plots saved to Paper_figures directory
    - Console output with fitting diagnostics and quality metrics
    - CSV exports of processed decay parameters

Quality Control:
    - RSD threshold (typically 0.1) for accepting decay fits
    - Visual inspection plots for each fit
    - Exclusion of outliers and poor quality data

Dependencies:
    - pandas: Data manipulation and Excel I/O
    - numpy: Numerical operations and array processing
    - scipy: Nonlinear curve fitting (optimize.curve_fit)
    - bokeh: Interactive web-based visualization
    - datetime: Time series handling

Configuration:
    - absolute_path: Project directory path (user-defined)
    - dataset: Selected instrument for processing
    - Burn-specific parameters in burn_log.xlsx

Notes:
    - This script processes one instrument dataset at a time (set via 'dataset' variable)
    - Results are compiled across all burns for comparative analysis
    - Time shifts account for instrument clock differences
    - Volume-normalized concentrations used for CADR calculations

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
    Div,
)
from bokeh.layouts import gridplot, row, column

# from bokeh.io import output_notebook#, export_png
from functools import reduce

# %% RUN User defines directory path for datset, dataset used, and dataset final location
# User set absolute_path
absolute_path = "C:/Users/nml/OneDrive - NIST/Documents/NIST/WUI_smoke/"  # USER ENTERED PROJECT PATH
os.chdir(absolute_path)

# use only one dataset at a time
dataset = "QuantAQb"  # USER ENTERED selected
print("dataset selected: " + dataset)

# Set up Datasets that are used
print(dataset + " dataset selected and final file path defined")
if dataset == "AeroTrakB":
    # Define the columns to drop and the renaming mapping
    columns_to_drop = [
        "Date and Time",
        "Ch1 Size (µm)",
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
        "total_count (#)",
        "Unnamed: 21",
        "Unnamed: 22",
        "Unnamed: 23",
    ]

    def rename_columns(sheet_name):
        return {
            "Ch1 Diff (#)": f"{sheet_name} Ch1 0.3µm (#/cm³)",
            "Ch2 Diff (#)": f"{sheet_name} Ch2 0.5µm (#/cm³)",
            "Ch3 Diff (#)": f"{sheet_name} Ch3 1.0µm (#/cm³)",
            "Ch4 Diff (#)": f"{sheet_name} Ch4 3.0µm (#/cm³)",
            "Ch5 Diff (#)": f"{sheet_name} Ch5 5.0µm (#/cm³)",
            "Ch6 Diff (#)": f"{sheet_name} Ch6 10.0µm (#/cm³)",
            "Ch1 0.3µm (#)": f"{sheet_name} total_count (#/cm³)",
        }

    # Initialize an empty list to hold the data frames
    data_frames = []

    # Iterate over the sheet names
    for i in range(2, 11):
        sheet_name = f"burn{i}"
        # Read and process the sheet
        df = pd.read_excel(
            "./burn_dates_decay_aerotraks_bedroom.xlsx", sheet_name=sheet_name
        )
        df = df.drop(columns=columns_to_drop)

        # Perform unit conversion before renaming
        df["Ch1 Diff (#)"] /= 2830
        df["Ch2 Diff (#)"] /= 2830
        df["Ch3 Diff (#)"] /= 2830
        df["Ch4 Diff (#)"] /= 2830
        df["Ch5 Diff (#)"] /= 2830
        df["Ch6 Diff (#)"] /= 2830
        df["Ch1 0.3µm (#)"] /= 2830

        df.rename(columns=rename_columns(sheet_name), inplace=True)
        data_frames.append(df)

    # Merge all data frames
    df = reduce(
        lambda left, right: pd.merge(left, right, on=["min_since_peak"], how="outer"),
        data_frames,
    )

    # Set the index and interpolate
    df = df.set_index("min_since_peak")
    df = df.interpolate("index")
    df["min_from_peak"] = df.index
    df = df[df["min_from_peak"] >= -10]

elif dataset == "AeroTrakK":
    # Define the columns to drop and the renaming mapping
    columns_to_drop = [
        "Date and Time",
        "Ch1 Size (µm)",
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
        "total_count (#)",
        "Unnamed: 21",
        "Unnamed: 22",
        "Unnamed: 23",
    ]

    def rename_columns(sheet_name):
        return {
            "Ch1 Diff (#)": f"{sheet_name} Ch1 0.3µm (#/cm³)",
            "Ch2 Diff (#)": f"{sheet_name} Ch2 0.5µm (#/cm³)",
            "Ch3 Diff (#)": f"{sheet_name} Ch3 1.0µm (#/cm³)",
            "Ch4 Diff (#)": f"{sheet_name} Ch4 3.0µm (#/cm³)",
            "Ch5 Diff (#)": f"{sheet_name} Ch5 5.0µm (#/cm³)",
            "Ch6 Diff (#)": f"{sheet_name} Ch6 10.0µm (#/cm³)",
            "Ch1 0.3µm (#)": f"{sheet_name} total_count (#/cm³)",
        }

    # Initialize an empty list to hold the data frames
    data_frames = []

    # Iterate over the sheet names
    for i in range(2, 11):
        sheet_name = f"burn{i}"
        # Read and process the sheet
        df = pd.read_excel(
            "./burn_dates_decay_aerotraks_kitchen.xlsx", sheet_name=sheet_name
        )
        df = df.drop(columns=columns_to_drop)

        # Perform unit conversion before renaming
        df["Ch1 Diff (#)"] /= 2830
        df["Ch2 Diff (#)"] /= 2830
        df["Ch3 Diff (#)"] /= 2830
        df["Ch4 Diff (#)"] /= 2830
        df["Ch5 Diff (#)"] /= 2830
        df["Ch6 Diff (#)"] /= 2830
        df["Ch1 0.3µm (#)"] /= 2830

        df.rename(columns=rename_columns(sheet_name), inplace=True)
        data_frames.append(df)

    # Merge all data frames
    df = reduce(
        lambda left, right: pd.merge(left, right, on=["min_since_peak"], how="outer"),
        data_frames,
    )

    # Set the index and interpolate
    df = df.set_index("min_since_peak")
    df = df.interpolate("index")
    df["min_from_peak"] = df.index
    df = df[df["min_from_peak"] >= -120]

elif dataset == "DustTrak":
    df = pd.read_excel("./burn_dates_decay_dusttrak.xlsx", sheet_name="plotdata")

elif dataset == "PurpleAir":
    # Define file paths
    file_garage = "./burn_dates_decay_purpleair_garage.xlsx"
    file_kitchen = "./burn_dates_decay_purpleair_kitchen.xlsx"

    # Initialize dictionaries to hold DataFrames for each sheet
    dfs_garage = {}
    dfs_kitchen = {}

    # Iterate over the sheet names generated from the range
    for i in range(6, 11):
        sheet_name = f"Burn {i}"

        # Read data from the garage file
        dfg = pd.read_excel(file_garage, sheet_name=sheet_name)
        # Read data from the kitchen file
        dfk = pd.read_excel(file_kitchen, sheet_name=sheet_name)

        # Drop the DateTime column
        dfg = dfg.drop(columns=["DateTime"])
        dfk = dfk.drop(columns=["DateTime"])

        # Remove the last row from each DataFrame
        dfg = dfg[:-1]
        dfk = dfk[:-1]

        # Store DataFrames in dictionaries with sheet names as keys
        dfs_garage[sheet_name] = dfg
        dfs_kitchen[sheet_name] = dfk

elif dataset in ("QuantAQb", "QuantAQk"):
    # Function to load and process CSV files
    def load_and_process_csv(file_path):
        try:
            df = pd.read_csv(file_path)
            df["timestamp_local"] = pd.to_datetime(
                df["timestamp_local"], errors="coerce"
            )
            df = df[::-1].reset_index(drop=True)
            return df
        except Exception as e:
            print(f"Error loading or processing {file_path}: {e}")
            return pd.DataFrame()  # Return an empty DataFrame on error

    # Load the CSV files into DataFrames
    bedroom_df = load_and_process_csv(
        "./burn_data/quantaq/MOD-PM-00194-b0fc215029fa4852b926bc50b28fda5a.csv"
    )
    kitchen_df = load_and_process_csv(
        "./burn_data/quantaq/MOD-PM-00197-a6dd467a147a4d95a7b98a8a10ab4ea3.csv"
    )

    # Define the dates for each Burn
    burn_dates = {
        "Burn 4": "2024-05-09",
        "Burn 5": "2024-05-13",
        "Burn 6": "2024-05-17",
        "Burn 7": "2024-05-20",
        "Burn 8": "2024-05-23",
        "Burn 9": "2024-05-28",
        "Burn 10": "2024-05-31",
    }

    # Function to filter data by whole day
    def filter_by_date(df, date):
        date = pd.to_datetime(date).date()
        if "timestamp_local" in df.columns:
            df["date_only"] = df["timestamp_local"].dt.date
            filtered_df = df[df["date_only"] == date].copy()
            df.drop(columns=["date_only"], inplace=True)
            if filtered_df.empty:
                print(f"No data found for date {date}.")
            return filtered_df
        else:
            print("Column 'timestamp_local' is missing.")
            return pd.DataFrame()

    # Function to add PM_total column
    def add_pm_total_column(df):
        required_columns = ["pm1", "pm25", "pm10"]
        if all(col in df.columns for col in required_columns):
            df["PM_total"] = df[required_columns].sum(axis=1)
        else:
            print(
                f"One or more of the required columns {required_columns} are missing in DataFrame."
            )

    # Function to add min_since_peak column
    def add_min_since_peak_column(df):
        if not df.empty and "PM_total" in df.columns:
            peak_index = df["PM_total"].idxmax()
            peak_time = df.loc[peak_index, "timestamp_local"]
            df["min_since_peak"] = (
                df["timestamp_local"] - peak_time
            ).dt.total_seconds() / 60
            df.loc[df["timestamp_local"] == peak_time, "min_since_peak"] = 0
        else:
            if df.empty:
                print("DataFrame is empty.")
            if "PM_total" not in df.columns:
                print("PM_total column is missing in DataFrame.")

    # Create a dictionary to hold filtered and processed data for each Burn
    burn_data = {
        burn: {"bedroom": pd.DataFrame(), "kitchen": pd.DataFrame()}
        for burn in burn_dates.keys()
    }

    # Filter, process, and add columns for each Burn
    for burn, date in burn_dates.items():
        print(f"Processing {burn} for date {date}...")
        burn_data[burn]["bedroom"] = filter_by_date(bedroom_df, date)
        burn_data[burn]["kitchen"] = filter_by_date(kitchen_df, date)

        # Add PM_total and min_since_peak columns to each filtered DataFrame
        add_pm_total_column(burn_data[burn]["bedroom"])
        add_pm_total_column(burn_data[burn]["kitchen"])

        add_min_since_peak_column(burn_data[burn]["bedroom"])
        add_min_since_peak_column(burn_data[burn]["kitchen"])

    # Optionally, print a summary of the processed data
    for burn, data in burn_data.items():
        print(f"\n{burn}:")
        for room, df in data.items():
            print(f"\n{room} data:")
            print(df.head())

elif dataset == "SMPS":
    df = pd.read_excel("./burn_dates_decay_smps.xlsx", sheet_name="plotdata")

# %% RUN for plot
if dataset == "AeroTrakB":
    # For visualization in Jupyter Notebook
    # output_notebook()
    # For visualization to static HTML
    output_file("./Paper_figures/AeroTrak_Bedroom_Normalized_Decacy.html")

    # Extract the minutes and data series
    minutes_np = df["min_from_peak"].values
    time_hours_np = minutes_np / 60  # Convert minutes to hours
    decay_data = {
        #'Burn 1': df['Baseline(norm.)'].values,
        "Burn 2": df["burn2 total_count (#/cm³)"].values
        / df["burn2 total_count (#/cm³)"].max(),
        "Burn 3": df["burn3 total_count (#/cm³)"].values
        / df["burn3 total_count (#/cm³)"].max(),
        "Burn 4": df["burn4 total_count (#/cm³)"].values
        / df["burn4 total_count (#/cm³)"].max(),
        "Burn 5": df["burn5 total_count (#/cm³)"].values
        / df["burn5 total_count (#/cm³)"].max(),
        "Burn 6": df["burn6 total_count (#/cm³)"].values
        / df["burn6 total_count (#/cm³)"].max(),
        "Burn 7": df["burn7 total_count (#/cm³)"].values
        / df["burn7 total_count (#/cm³)"].max(),
        "Burn 8": df["burn8 total_count (#/cm³)"].values
        / df["burn8 total_count (#/cm³)"].max(),
        "Burn 9": df["burn9 total_count (#/cm³)"].values
        / df["burn9 total_count (#/cm³)"].max(),
        "Burn 10": df["burn10 total_count (#/cm³)"].values
        / df["burn10 total_count (#/cm³)"].max(),
    }

    # Define different ranges for fitting each series in minutes
    fit_ranges = {
        #'Burn 1': (16.04, 176.47),  # Fit from 16 to 176 minutes
        "Burn 2": (8, 37),
        "Burn 3": (16, 133),
        "Burn 4": (30, 176),
        "Burn 5": (33, 237),  # end was 655.08 but changed to 240 to fit figure
        "Burn 6": (0, 11),
        "Burn 7": (20, 92),
        "Burn 8": (10, 98),
        "Burn 9": (27, 115),
        "Burn 10": (25, 99),
    }

    # Define specific label positions for each series
    label_positions = {
        #'Burn 1': (-10, 0.1),  # x_offset in minutes, y_offset in units above the line
        "Burn 2": (-20, -0.07),
        "Burn 3": (15, 0.0),
        "Burn 4": (12, -0.01),
        "Burn 5": (-50, -0.15),
        "Burn 6": (0, -0.16),
        "Burn 7": (-30, -0.013),
        "Burn 8": (-5, -0.016),
        "Burn 9": (8, 0.002),
        "Burn 10": (10, -0.0125),
    }

    # Define custom line colors and line types for each series
    line_properties = {
        #'Burn 1': {'color': 'black', 'line_dash': 'solid','line_width': 1.5},
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
    x_range_start = 0
    x_range_end = 250  # Cover an additional hour for a full view
    y_range_start = 0.001  # Avoid log scale issues by not starting at zero
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
        # title="AeroTrak Bedroom Two Normalized Decay",
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
        label_y = (
            y_curve_fit[-1] + y_offset
        )  # Move label slightly above the fit line end

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

    # Convert the list of dictionaries to a DataFrame
    burn_calc = pd.DataFrame(burn_calc)

elif dataset == "AeroTrakK":
    # For visualization in Jupyter Notebook
    # output_notebook()
    # For visualization to static HTML
    output_file("./Paper_figures/AeroTrak_Kitchen_Normalized_Decacy.html")

    df = df[df["min_from_peak"] >= -10]

    # Extract the minutes and data series
    minutes_np = df["min_from_peak"].values
    time_hours_np = minutes_np / 60  # Convert minutes to hours
    decay_data = {
        #'Burn 1': df['Baseline(norm.)'].values,
        "Burn 2": df["burn2 total_count (#/cm³)"].values
        / df["burn2 total_count (#/cm³)"].max(),
        "Burn 3": df["burn3 total_count (#/cm³)"].values
        / df["burn3 total_count (#/cm³)"].max(),
        "Burn 4": df["burn4 total_count (#/cm³)"].values
        / df["burn4 total_count (#/cm³)"].max(),
        "Burn 5": df["burn5 total_count (#/cm³)"].values
        / df["burn5 total_count (#/cm³)"].max(),
        "Burn 6": df["burn6 total_count (#/cm³)"].values
        / df["burn6 total_count (#/cm³)"].max(),
        "Burn 7": df["burn7 total_count (#/cm³)"].values
        / df["burn7 total_count (#/cm³)"].max(),
        "Burn 8": df["burn8 total_count (#/cm³)"].values
        / df["burn8 total_count (#/cm³)"].max(),
        "Burn 9": df["burn9 total_count (#/cm³)"].values
        / df["burn9 total_count (#/cm³)"].max(),
        "Burn 10": df["burn10 total_count (#/cm³)"].values
        / df["burn10 total_count (#/cm³)"].max(),
    }

    # Define different ranges for fitting each series in minutes
    fit_ranges = {
        #'Burn 1': (16.04, 176.47),  # Fit from 16 to 176 minutes
        "Burn 2": (8, 52),
        "Burn 3": (16, 104),
        "Burn 4": (33, 120),
        "Burn 5": (19, 101),
        "Burn 6": (23, 81),
        "Burn 7": (3, 76),
        "Burn 8": (2, 75),
        "Burn 9": (19, 78),
        "Burn 10": (18, 76),
    }

    # Define specific label positions for each series
    label_positions = {
        #'Burn 1': (-10, 0.1),  # x_offset in minutes, y_offset in units above the line
        "Burn 2": (-35, -0.022),
        "Burn 3": (45, 0.06),
        "Burn 4": (65, 0.03),
        "Burn 5": (20, 0.05),
        "Burn 6": (10, 0.1),
        "Burn 7": (25, -0.013),
        "Burn 8": (25, -0.03),
        "Burn 9": (20, 0.002),
        "Burn 10": (-25, -0.045),
    }

    # Define custom line colors and line types for each series
    line_properties = {
        #'Burn 1': {'color': 'black', 'line_dash': 'solid','line_width': 1.5},
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
    x_range_start = 0
    x_range_end = 250  # Cover an additional hour for a full view
    y_range_start = 0.001  # Avoid log scale issues by not starting at zero
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
        # title="AeroTrak Morning Room Normalized Decay",
        x_axis_label="Time since peak conc. (minutes)",
        # y_axis_label='-Ln(C_t1 / C_t0)',
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
        print(label + f" {popt[1]:.13f} h^{-1} ± {1.96 * std_err[1]:.13f} h^{-1}")

        # Retrieve label position for the current series
        x_offset, y_offset = label_positions[label]
        label_x = minutes_np[fit_end_index - 1] + x_offset  # Move label to the right
        label_y = (
            y_curve_fit[-1] + y_offset
        )  # Move label slightly above the fit line end

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

    # Convert the list of dictionaries to a DataFrame
    burn_calc = pd.DataFrame(burn_calc)

elif dataset == "DustTrak":
    # For visualization in Jupyter Notebook
    # output_notebook()
    # For visualization to static HTML
    output_file("./Paper_figures/DustTrak_Normalized_Decacy.html")

    # Extract the minutes and data series
    minutes_np = df["min_from_peak"].values
    time_hours_np = minutes_np / 60  # Convert minutes to hours
    decay_data = {
        "Burn 1": df["Baseline(norm.)"].values,
        #'Burn 2': df['(4)MERV 13(norm.)'].values,
        "Burn 3": df["(1)MERV 13 Used(norm.)"].values,
        "Burn 4": df["(1)MERV 13 New(norm.)"].values,
        "Burn 5": df["ClosedBR(norm.)"].values,
        "Burn 6": df["ClosedBRw/Filter(norm.)"].values,
        "Burn 7": df["(2)MERV 12A New(norm.)"].values,
        "Burn 8": df["(2)MERV 12A Used(norm.)"].values,
        "Burn 9": df["(2)MERV 13 New(norm.)"].values,
        "Burn 10": df["(2)MERV 13 Used(norm.)"].values,
    }

    # Define different ranges for fitting each series in minutes
    fit_ranges = {
        "Burn 1": (15, 170),  # Fit from 15 to 170 minutes
        #'Burn 2': (15, 24.06),
        "Burn 3": (15, 76),
        "Burn 4": (15, 73),
        "Burn 5": (15, 240),  # end was 817.64 but changed to 240 to fit figure
        "Burn 6": (0, 10),
        "Burn 7": (15, 42),
        "Burn 8": (15, 32),
        "Burn 9": (15, 32),
        "Burn 10": (15, 43),
    }

    # Define specific label positions for each series
    label_positions = {
        "Burn 1": (-10, 0.07),  # x_offset in minutes, y_offset in units above the line
        #'Burn 2': (-5, -0.054),
        "Burn 3": (30, -0.015),
        "Burn 4": (-15, -0.025),
        "Burn 5": (-60, -0.15),
        "Burn 6": (0, -0.165),
        "Burn 7": (-2, -0.02),
        "Burn 8": (2, -0.041),
        "Burn 9": (-10, -0.041),
        "Burn 10": (5, -0.025),
    }

    # Define custom line colors and line types for each series
    line_properties = {
        "Burn 1": {"color": "black", "line_dash": "solid", "line_width": 1.5},
        #'Burn 2': {'color': 'orange', 'line_dash': 'solid','line_width': 1.5},
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
    y_range_start = 0.0001  # Avoid log scale issues by not starting at zero
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
        # title="DustTrak Normalized Decay",
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
        label_y = (
            y_curve_fit[-1] + y_offset
        )  # Move label slightly above the fit line end

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

    # Adjust the legend position to the lower-left corner
    p.legend.location = "bottom_left"
    p.legend.orientation = "vertical"

    # Center the title text
    p.title.align = "center"

    # Show the plot
    show(p)

    # Convert the list of dictionaries to a DataFrame
    burn_calc = pd.DataFrame(burn_calc)

elif dataset == "PurpleAir":
    # For visualization in Jupyter Notebook
    output_notebook()
    # For visualization to static HTML
    # output_file('./Paper_figures/PurpleAir_Kitchen_Normalized_Decay.html')

    # Function to calculate exponential decay
    def exponential_decay(x, a, b):
        return a * np.exp(-b * x)

    # Function to fit the exponential curve
    def fit_exponential_curve(x, y):
        popt, pcov = optimize.curve_fit(exponential_decay, x, y, p0=(1, 1))
        a, b = popt
        std_err = np.sqrt(np.diag(pcov))
        y_fit = exponential_decay(x, *popt)
        return popt, y_fit, std_err

    # Define the x-axis and y-axis ranges
    x_range_start = 0
    x_range_end = 250  # Cover an additional hour for a full view
    y_range_start = 0.001  # Avoid log scale issues by not starting at zero
    y_range_end = 1  # Example upper limit for the y-axis

    # Define different ranges for fitting each series in minutes
    fit_ranges = {
        "Burn 6": (15, 237),
        "Burn 7": (15, 58),
        "Burn 8": (15, 56),
        "Burn 9": (15, 66),
        "Burn 10": (15, 66),
    }

    # Define specific label positions for each series
    label_positions = {
        "Burn 6": (-50, 0.1),
        "Burn 7": (-30, -0.02),
        "Burn 8": (40, -0.02),
        "Burn 9": (30, -0.007),
        "Burn 10": (30, -0.013),
    }

    # Define custom line colors and line types for each series
    line_properties = {
        "Burn 6": {"color": "blue", "line_dash": "solid", "line_width": 1.5},
        "Burn 7": {"color": "green", "line_dash": "solid", "line_width": 1.5},
        "Burn 8": {"color": "green", "line_dash": "dashed", "line_width": 1.5},
        "Burn 9": {"color": "cyan", "line_dash": "solid", "line_width": 1.5},
        "Burn 10": {"color": "cyan", "line_dash": "dashed", "line_width": 1.5},
    }

    # Create a Bokeh figure with logarithmic y-axis and minutes on x-axis
    p = figure(
        # title="Purple Air Morning Room Normalized Decay",
        x_axis_label="Time since peak conc. (minutes)",
        y_axis_type="log",
        x_axis_type="linear",
        x_range=(x_range_start, x_range_end),
        y_range=(y_range_start, y_range_end),
        max_width=600,
        height=800,
    )

    # List to collect results for later use
    burn_calc = []

    # Loop through each burn data
    for sheet_name in ["Burn 6", "Burn 7", "Burn 8", "Burn 9", "Burn 10"]:
        # Access the DataFrame from the dictionaries
        dfk = dfs_kitchen[sheet_name]

        # Extract the minutes and data series
        minutes_np = dfk["min_since_peak"].values
        time_hours_np = minutes_np / 60  # Convert minutes to hours

        # Normalize the data for plotting
        data = dfk["Average Concentration(µg/m³)"].values
        normalized_data = data / data.max()

        # Get line properties for the current series
        color = line_properties[sheet_name]["color"]
        line_dash = line_properties[sheet_name]["line_dash"]
        line_width = line_properties[sheet_name]["line_width"]

        # Plot the original data with legend
        p.line(
            minutes_np,
            normalized_data,
            legend_label=sheet_name,
            line_width=line_width,
            color=color,
            line_dash=line_dash,
        )

        # Define the range for fitting
        fit_start_min, fit_end_min = fit_ranges[sheet_name]

        # Extract fitting data
        fit_start_index = np.searchsorted(minutes_np, fit_start_min)
        fit_end_index = np.searchsorted(minutes_np, fit_end_min)

        # Fit exponential curve on the specified portion of the time series
        x_fit = time_hours_np[fit_start_index:fit_end_index]
        y_fit = normalized_data[fit_start_index:fit_end_index]
        popt, y_curve_fit, std_err = fit_exponential_curve(x_fit, y_fit)

        # Define the fit curve and uncertainty band
        curve_fit_y = exponential_decay(time_hours_np, *popt)

        # Add the curve fit line for the fitting portion without legend
        p.line(
            minutes_np[fit_start_index:fit_end_index],
            y_curve_fit,
            line_color="red",
            line_dash="solid",
        )  # No legend label

        # Prepare fit text with only b value and uncertainty
        fit_info = f"{popt[1]:.2f} h$$^{{-1}}$$ ± {1.96 * std_err[1]:.2f} h$$^{{-1}}$$"
        print(f"{sheet_name} {popt[1]:.3f} h^{-1} ± {1.96 * std_err[1]:.3f} h^{-1}")

        # Retrieve label position for the current series
        x_offset, y_offset = label_positions[sheet_name]
        label_x = minutes_np[fit_end_index - 1] + x_offset  # Move label to the right
        label_y = (
            y_curve_fit[-1] + y_offset
        )  # Move label slightly above the fit line end

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
            "burn": sheet_name,
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

    # Adjust the legend position to the lower-left corner
    p.legend.location = "bottom_left"
    p.legend.orientation = "vertical"

    # Center the title text
    p.title.align = "center"

    # Show the plot
    show(p)

    # Convert the list of dictionaries to a DataFrame
    burn_calc = pd.DataFrame(burn_calc)

elif dataset == "QuantAQb":
    # For visualization in Jupyter Notebook
    # output_notebook()
    # For visualization to static HTML
    output_file("./Paper_figures/QuantAQ_Bedroom_Normalized_Decay.html")

    # Preprocess data to find the max concentration, 5% threshold, the first instance below this threshold, and the corresponding min_since_peak
    max_concentrations = {}
    pm_total_at_15_min = {}
    thresholds = {}
    first_below_threshold = {}
    min_since_peak_at_threshold = {}

    for burn in burn_dates.keys():
        # Access the DataFrame for the current burn
        dk = burn_data[burn][
            "bedroom"
        ]  # Assuming 'bedroom' is the data you want to analyze

        # Get the maximum value of PM_total and calculate 5% of this maximum
        max_concentration = dk["PM_total"].max()
        threshold_value = 0.05 * max_concentration

        # Find the PM_total value at min_since_peak of 15 minutes
        pm_total_at_15 = dk[dk["min_since_peak"] == 15]["PM_total"]
        if not pm_total_at_15.empty:
            pm_total_at_15_value = pm_total_at_15.values[0]
        else:
            pm_total_at_15_value = (
                np.nan
            )  # Handle case where there is no data at exactly 15 minutes

        # Find the first instance where PM_total falls to or below the threshold after 15 minutes
        post_15_min_data = dk[dk["min_since_peak"] >= 15]

        if post_15_min_data.empty:
            print(f"No data available after 15 minutes for {burn}.")
            continue

        # Find the first row where PM_total is less than or equal to the threshold
        threshold_row = post_15_min_data[
            post_15_min_data["PM_total"] <= threshold_value
        ]

        if threshold_row.empty:
            print(
                f"No PM_total values below 5% of maximum after 15 minutes for {burn}."
            )
            continue

        # Get the first occurrence of this row
        first_row_index = threshold_row.index.min()
        min_since_peak_at_threshold_value = dk.loc[first_row_index, "min_since_peak"]

        # Store the results
        max_concentrations[burn] = max_concentration
        thresholds[burn] = threshold_value
        first_below_threshold[burn] = dk.loc[first_row_index, "PM_total"]
        min_since_peak_at_threshold[burn] = min_since_peak_at_threshold_value
        pm_total_at_15_min[burn] = pm_total_at_15_value

        # Print the results for each burn
        print(f"{burn}:")
        print(f"  Max PM_total: {max_concentration:.2f}")
        print(f"  PM_total at 15 minutes: {pm_total_at_15_min[burn]:.2f}")
        print(f"  5% of max PM_total: {threshold_value:.2f}")
        print(f"  First PM_total ≤ 5% of max: {first_below_threshold[burn]:.2f}")
        print(
            f"  Corresponding min_since_peak: {min_since_peak_at_threshold[burn]:.2f}"
        )
        print()

    # Function to calculate exponential decay
    def exponential_decay(x, a, b):
        return a * np.exp(-b * x)

    # Function to fit the exponential curve
    def fit_exponential_curve(x, y):
        popt, pcov = optimize.curve_fit(exponential_decay, x, y, p0=(1, 1))
        a, b = popt
        std_err = np.sqrt(np.diag(pcov))
        y_fit = exponential_decay(x, *popt)
        return popt, y_fit, std_err

    # Define the x-axis and y-axis ranges
    x_range_start = 0
    x_range_end = 250  # Cover an additional hour for a full view
    y_range_start = 0.001  # Avoid log scale issues by not starting at zero
    y_range_end = 1  # Example upper limit for the y-axis

    # Define specific label positions for each series
    label_positions = {
        "Burn 4": (30, 0.05),
        "Burn 5": (-70, -0.1),
        "Burn 6": (30, 0.1),
        "Burn 7": (-40, -0.034),
        "Burn 8": (-25, -0.037),
        "Burn 9": (40, 0.09),
        "Burn 10": (20, -0.025),
    }

    # Define custom line colors and line types for each series
    line_properties = {
        "Burn 4": {"color": "brown", "line_dash": "solid", "line_width": 1.5},
        "Burn 5": {"color": "purple", "line_dash": "solid", "line_width": 1.5},
        "Burn 6": {"color": "blue", "line_dash": "solid", "line_width": 1.5},
        "Burn 7": {"color": "green", "line_dash": "solid", "line_width": 1.5},
        "Burn 8": {"color": "green", "line_dash": "dashed", "line_width": 1.5},
        "Burn 9": {"color": "cyan", "line_dash": "solid", "line_width": 1.5},
        "Burn 10": {"color": "cyan", "line_dash": "dashed", "line_width": 1.5},
    }

    # Update fit_ranges based on min_since_peak_at_threshold values
    fit_ranges = {
        "Burn 5": (15, 250),
        "Burn 6": (0, 15),
        **{
            burn: (15, min_since_peak_at_threshold[burn])
            for burn in burn_dates.keys()
            if burn not in ["Burn 5", "Burn 6"]
        },
    }

    # Create a Bokeh figure with logarithmic y-axis and minutes on x-axis
    p = figure(
        # title="QuantAQ Bedroom Two Normalized Decay",
        x_axis_label="Time since peak conc. (minutes)",
        y_axis_type="log",
        x_axis_type="linear",
        x_range=(x_range_start, x_range_end),
        y_range=(y_range_start, y_range_end),
        max_width=600,
        height=800,
    )

    # List to collect results for later use
    burn_calc = []

    # Loop through each burn data
    for burn in burn_dates.keys():
        # Access the DataFrame from the dictionaries
        dk = burn_data[burn][
            "bedroom"
        ]  # Assuming 'bedroom' is the data you want to plot

        # Extract the minutes and data series
        minutes_np = dk["min_since_peak"].values
        time_hours_np = minutes_np / 60  # Convert minutes to hours

        # Normalize the data for plotting
        data = dk["PM_total"].values  # Use 'PM_total' for bedroom data
        normalized_data = data / data.max()

        # Get line properties for the current series
        color = line_properties[burn]["color"]
        line_dash = line_properties[burn]["line_dash"]
        line_width = line_properties[burn]["line_width"]

        # Plot the original data with legend
        p.line(
            minutes_np,
            normalized_data,
            legend_label=burn,
            line_width=line_width,
            color=color,
            line_dash=line_dash,
        )

        # Define the range for fitting
        fit_start_min, fit_end_min = fit_ranges[burn]

        # Extract fitting data
        fit_start_index = np.searchsorted(minutes_np, fit_start_min)
        fit_end_index = np.searchsorted(minutes_np, fit_end_min)

        # Fit exponential curve on the specified portion of the time series
        x_fit = time_hours_np[fit_start_index:fit_end_index]
        y_fit = normalized_data[fit_start_index:fit_end_index]
        if len(x_fit) > 0 and len(y_fit) > 0:
            popt, y_curve_fit, std_err = fit_exponential_curve(x_fit, y_fit)

            # Define the fit curve and uncertainty band
            curve_fit_y = exponential_decay(time_hours_np, *popt)

            # Add the curve fit line for the fitting portion without legend
            p.line(
                minutes_np[fit_start_index:fit_end_index],
                y_curve_fit,
                line_color="red",
                line_dash="solid",
            )  # No legend label

            # Prepare fit text with only b value and uncertainty
            fit_info = (
                f"{popt[1]:.2f} h$$^{{-1}}$$ ± {1.96 * std_err[1]:.2f} h$$^{{-1}}$$"
            )
            print(f"{burn} {popt[1]:.3f} h^{-1} ± {1.96 * std_err[1]:.3f} h^{-1}")

            # Retrieve label position for the current series
            x_offset, y_offset = label_positions[burn]
            label_x = (
                minutes_np[fit_end_index - 1] + x_offset
            )  # Move label to the right
            label_y = (
                y_curve_fit[-1] + y_offset
            )  # Move label slightly above the fit line end

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
                "burn": burn,
                "decay": f"{popt[1]:.4f}",
                "decay_uncertainty": f"{1.96 * std_err[1]:.4f}",
            }
            # Add the dictionary to the list
            burn_calc.append(new_row)
        else:
            print(
                f"No data to fit for {burn} between {fit_start_min} and {fit_end_min} minutes."
            )

    # Adjust the legend position to the lower-left corner
    p.legend.location = "bottom_left"
    p.legend.label_text_font = "Calibri"
    p.legend.label_text_font_size = "12pt"
    p.legend.orientation = "vertical"

    p.yaxis.axis_label = r"$$LN(C_{t1} / C_{t0})$$"

    p.axis.axis_label_text_font = "Calibri"
    p.axis.axis_label_text_font_size = "12pt"
    p.axis.axis_label_text_font_style = "normal"

    # Adjust the legend position to the lower-left corner
    p.legend.location = "bottom_left"
    p.legend.orientation = "vertical"

    # Center the title text
    p.title.align = "center"

    # Show the plot
    show(p)

    # Convert the list of dictionaries to a DataFrame
    burn_calc = pd.DataFrame(burn_calc)

elif dataset == "QuantAQk":
    # For visualization in Jupyter Notebook
    # output_notebook()
    # For visualization to static HTML
    output_file("./Paper_figures/QuantAQ_Kitchen_Normalized_Decay.html")

    # Preprocess data to find the max concentration, 5% threshold, the first instance below this threshold, and the corresponding min_since_peak
    max_concentrations = {}
    pm_total_at_15_min = {}
    thresholds = {}
    first_below_threshold = {}
    min_since_peak_at_threshold = {}

    for burn in burn_dates.keys():
        # Access the DataFrame for the current burn
        dk = burn_data[burn][
            "kitchen"
        ]  # Assuming 'kitchen' is the data you want to analyze

        # Get the maximum value of PM_total and calculate 5% of this maximum
        max_concentration = dk["PM_total"].max()
        threshold_value = 0.05 * max_concentration

        # Find the PM_total value at min_since_peak of 15 minutes
        pm_total_at_15 = dk[dk["min_since_peak"] == 15]["PM_total"]
        if not pm_total_at_15.empty:
            pm_total_at_15_value = pm_total_at_15.values[0]
        else:
            pm_total_at_15_value = (
                np.nan
            )  # Handle case where there is no data at exactly 15 minutes

        # Find the first instance where PM_total falls to or below the threshold after 15 minutes
        post_15_min_data = dk[dk["min_since_peak"] >= 15]

        if post_15_min_data.empty:
            print(f"No data available after 15 minutes for {burn}.")
            continue

        # Find the first row where PM_total is less than or equal to the threshold
        threshold_row = post_15_min_data[
            post_15_min_data["PM_total"] <= threshold_value
        ]

        if threshold_row.empty:
            print(
                f"No PM_total values below 5% of maximum after 15 minutes for {burn}."
            )
            continue

        # Get the first occurrence of this row
        first_row_index = threshold_row.index.min()
        min_since_peak_at_threshold_value = dk.loc[first_row_index, "min_since_peak"]

        # Store the results
        max_concentrations[burn] = max_concentration
        thresholds[burn] = threshold_value
        first_below_threshold[burn] = dk.loc[first_row_index, "PM_total"]
        min_since_peak_at_threshold[burn] = min_since_peak_at_threshold_value
        pm_total_at_15_min[burn] = pm_total_at_15_value

        # Print the results for each burn
        print(f"{burn}:")
        print(f"  Max PM_total: {max_concentration:.2f}")
        print(f"  PM_total at 15 minutes: {pm_total_at_15_min[burn]:.2f}")
        print(f"  5% of max PM_total: {threshold_value:.2f}")
        print(f"  First PM_total ≤ 5% of max: {first_below_threshold[burn]:.2f}")
        print(
            f"  Corresponding min_since_peak: {min_since_peak_at_threshold[burn]:.2f}"
        )
        print()

    # Function to calculate exponential decay
    def exponential_decay(x, a, b):
        return a * np.exp(-b * x)

    # Function to fit the exponential curve
    def fit_exponential_curve(x, y):
        popt, pcov = optimize.curve_fit(exponential_decay, x, y, p0=(1, 1))
        a, b = popt
        std_err = np.sqrt(np.diag(pcov))
        y_fit = exponential_decay(x, *popt)
        return popt, y_fit, std_err

    # Define the x-axis and y-axis ranges
    x_range_start = 0
    x_range_end = 250  # Cover an additional hour for a full view
    y_range_start = 0.001  # Avoid log scale issues by not starting at zero
    y_range_end = 1  # Example upper limit for the y-axis

    # Define specific label positions for each series
    label_positions = {
        "Burn 4": (30, -0.03),
        "Burn 5": (-30, -0.02),
        "Burn 6": (-50, 0.1),
        "Burn 7": (-30, -0.04),
        "Burn 8": (37, -0.02),
        "Burn 9": (32, -0.018),
        "Burn 10": (30, -0.013),
    }

    # Define custom line colors and line types for each series
    line_properties = {
        "Burn 4": {"color": "brown", "line_dash": "solid", "line_width": 1.5},
        "Burn 5": {"color": "purple", "line_dash": "solid", "line_width": 1.5},
        "Burn 6": {"color": "blue", "line_dash": "solid", "line_width": 1.5},
        "Burn 7": {"color": "green", "line_dash": "solid", "line_width": 1.5},
        "Burn 8": {"color": "green", "line_dash": "dashed", "line_width": 1.5},
        "Burn 9": {"color": "cyan", "line_dash": "solid", "line_width": 1.5},
        "Burn 10": {"color": "cyan", "line_dash": "dashed", "line_width": 1.5},
    }

    # Update fit_ranges based on min_since_peak_at_threshold values
    fit_ranges = {
        burn: (15, min_since_peak_at_threshold[burn]) for burn in burn_dates.keys()
    }

    # Create a Bokeh figure with logarithmic y-axis and minutes on x-axis
    p = figure(
        # title="QuantAQ Morning Room Normalized Decay",
        x_axis_label="Time since peak conc. (minutes)",
        y_axis_type="log",
        x_axis_type="linear",
        x_range=(x_range_start, x_range_end),
        y_range=(y_range_start, y_range_end),
        max_width=600,
        height=800,
    )

    # List to collect results for later use
    burn_calc = []

    # Loop through each burn data
    for burn in burn_dates.keys():
        # Access the DataFrame from the dictionaries
        dk = burn_data[burn][
            "kitchen"
        ]  # Assuming 'kitchen' is the data you want to plot

        # Extract the minutes and data series
        minutes_np = dk["min_since_peak"].values
        time_hours_np = minutes_np / 60  # Convert minutes to hours

        # Normalize the data for plotting
        data = dk["PM_total"].values  # Use 'PM_total' for kitchen data
        normalized_data = data / data.max()

        # Get line properties for the current series
        color = line_properties[burn]["color"]
        line_dash = line_properties[burn]["line_dash"]
        line_width = line_properties[burn]["line_width"]

        # Plot the original data with legend
        p.line(
            minutes_np,
            normalized_data,
            legend_label=burn,
            line_width=line_width,
            color=color,
            line_dash=line_dash,
        )

        # Define the range for fitting
        fit_start_min, fit_end_min = fit_ranges[burn]

        # Extract fitting data
        fit_start_index = np.searchsorted(minutes_np, fit_start_min)
        fit_end_index = np.searchsorted(minutes_np, fit_end_min)

        # Fit exponential curve on the specified portion of the time series
        x_fit = time_hours_np[fit_start_index:fit_end_index]
        y_fit = normalized_data[fit_start_index:fit_end_index]
        if len(x_fit) > 0 and len(y_fit) > 0:
            popt, y_curve_fit, std_err = fit_exponential_curve(x_fit, y_fit)

            # Define the fit curve and uncertainty band
            curve_fit_y = exponential_decay(time_hours_np, *popt)

            # Add the curve fit line for the fitting portion without legend
            p.line(
                minutes_np[fit_start_index:fit_end_index],
                y_curve_fit,
                line_color="red",
                line_dash="solid",
            )  # No legend label

            # Prepare fit text with only b value and uncertainty
            fit_info = (
                f"{popt[1]:.2f} h$$^{{-1}}$$ ± {1.96 * std_err[1]:.2f} h$$^{{-1}}$$"
            )
            print(f"{burn} {popt[1]:.3f} h^{-1} ± {1.96 * std_err[1]:.3f} h^{-1}")

            # Retrieve label position for the current series
            x_offset, y_offset = label_positions[burn]
            label_x = (
                minutes_np[fit_end_index - 1] + x_offset
            )  # Move label to the right
            label_y = (
                y_curve_fit[-1] + y_offset
            )  # Move label slightly above the fit line end

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
                "burn": burn,
                "decay": f"{popt[1]:.4f}",
                "decay_uncertainty": f"{1.96 * std_err[1]:.4f}",
            }
            # Add the dictionary to the list
            burn_calc.append(new_row)
        else:
            print(
                f"No data to fit for {burn} between {fit_start_min} and {fit_end_min} minutes."
            )

    # Adjust the legend position to the lower-left corner
    p.legend.location = "bottom_left"
    p.legend.label_text_font = "Calibri"
    p.legend.label_text_font_size = "12pt"
    p.legend.orientation = "vertical"

    p.yaxis.axis_label = r"$$LN(C_{t1} / C_{t0})$$"

    p.axis.axis_label_text_font = "Calibri"
    p.axis.axis_label_text_font_size = "12pt"
    p.axis.axis_label_text_font_style = "normal"

    # Adjust the legend position to the lower-left corner
    p.legend.location = "bottom_left"
    p.legend.orientation = "vertical"

    # Center the title text
    p.title.align = "center"

    # Show the plot
    show(p)

    # Convert the list of dictionaries to a DataFrame
    burn_calc = pd.DataFrame(burn_calc)

elif dataset == "SMPS":
    # For visualization in Jupyter Notebook
    output_notebook()
    # For visualization to static HTML
    output_file("./Paper_figures/SMPS_Dwelling_Normalized_Decacy.html")

    # Extract the minutes and data series
    minutes_np = df["min_from_peak(min)"].values
    time_hours_np = minutes_np / 60  # Convert minutes to hours
    decay_data = {
        "Burn 1": df["Baseline(norm.)"].values,
        "Burn 2": df["(4)MERV 13(norm.)"].values,
        "Burn 3": df["(1)MERV 13 Used(norm.)"].values,
        "Burn 4": df["(1)MERV 13 New(norm.)"].values,
        "Burn 5": df["ClosedBR(norm.)"].values,
        "Burn 6": df["ClosedBRw/Filter(norm.)"].values,
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
        "Burn 3": (10, 0.01),
        "Burn 4": (-5, -0.019),
        "Burn 5": (-60, -0.2),
        "Burn 6": (-5, -0.197),
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
    y_range_start = 0.0001  # Avoid log scale issues by not starting at zero
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
        # title="SMPS Dwelling Normalized Decay",
        x_axis_label="Time since peak conc. (minutes)",
        y_axis_label=" ",
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
        label_y = (
            y_curve_fit[-1] + y_offset
        )  # Move label slightly above the fit line end

        # Add Label with plain text
        p.add_layout(
            Label(
                x=label_x,
                y=label_y,
                text=fit_info,
                text_font="Calibri",
                text_font_size="12pt",
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

    # Convert the list of dictionaries to a DataFrame
    burn_calc = pd.DataFrame(burn_calc)
# %% RUN for CADR Calculations
if dataset == "AeroTrakB":
    # Convert 'decay' and 'decay_uncertainty' columns to numeric in case they are strings
    burn_calc["decay"] = pd.to_numeric(burn_calc["decay"], errors="coerce")
    burn_calc["decay_uncertainty"] = pd.to_numeric(
        burn_calc["decay_uncertainty"], errors="coerce"
    )

    # Dictionary mapping burn values (as strings) to the number of CR boxes
    crbox_mapping = {
        "Burn 1": 0,
        "Burn 2": 4,
        "Burn 3": 1,
        "Burn 4": 1,
        "Burn 5": 0,
        "Burn 6": 1,
        "Burn 7": 2,
        "Burn 8": 2,
        "Burn 9": 2,
        "Burn 10": 2,
    }

    # Function to determine CR boxes based on 'burn' value using the dictionary
    def get_crboxes(burn_value):
        return crbox_mapping.get(burn_value, 0)

    burn_calc["CRboxes"] = burn_calc["burn"].apply(get_crboxes)

    # these values are from the results AeroTrakK
    baseline_decay = 0.3074
    baseline_decay_uncertainty = 0.0063

    # Volume (replace with the actual volume)
    volume1 = 324
    volume2 = 33

    # Calculate CADR and CADR uncertainty with special handling for 'Burn 5' and 'Burn 6'
    def calculate_cadr_and_uncertainty(row):
        if row["burn"] in ["Burn 5"]:
            # Special calculations for Burn 5 and Burn 6
            cadr = pd.NA
            cadr_uncertainty = pd.NA
        if row["burn"] in ["Burn 6"]:
            cadr = volume2 * (row["decay"] - baseline_decay)
            cadr_uncertainty = volume2 * (
                (row["decay"] + row["decay_uncertainty"])
                - (baseline_decay - baseline_decay_uncertainty)
            ) - volume2 * (
                (row["decay"] - row["decay_uncertainty"])
                - (baseline_decay + baseline_decay_uncertainty)
            )
        else:
            # Standard calculations for other burns
            cadr = volume1 * (row["decay"] - baseline_decay)
            cadr_uncertainty = volume1 * (
                (row["decay"] + row["decay_uncertainty"])
                - (baseline_decay - baseline_decay_uncertainty)
            ) - volume1 * (
                (row["decay"] - row["decay_uncertainty"])
                - (baseline_decay + baseline_decay_uncertainty)
            )
        return pd.Series([cadr, cadr_uncertainty])

    burn_calc[["CADR", "CADR_uncertainty"]] = burn_calc.apply(
        calculate_cadr_and_uncertainty, axis=1
    )

    # Calculate CADR per CRbox and uncertainty per CRbox
    # Handle division by zero for CR boxes
    burn_calc["CADR_per_CRbox"] = burn_calc["CADR"] / burn_calc["CRboxes"].replace(
        0, pd.NA
    )
    burn_calc["CADR_per_CRbox_uncertainty"] = burn_calc["CADR_uncertainty"] / burn_calc[
        "CRboxes"
    ].replace(0, pd.NA)

    print(burn_calc)

elif dataset == "AeroTrakK":
    # Convert 'decay' and 'decay_uncertainty' columns to numeric in case they are strings
    burn_calc["decay"] = pd.to_numeric(burn_calc["decay"], errors="coerce")
    burn_calc["decay_uncertainty"] = pd.to_numeric(
        burn_calc["decay_uncertainty"], errors="coerce"
    )

    # Dictionary mapping burn values (as strings) to the number of CR boxes
    crbox_mapping = {
        "Burn 1": 0,
        "Burn 2": 4,
        "Burn 3": 1,
        "Burn 4": 1,
        "Burn 5": 0,
        "Burn 6": 1,
        "Burn 7": 2,
        "Burn 8": 2,
        "Burn 9": 2,
        "Burn 10": 2,
    }

    # Function to determine CR boxes based on 'burn' value using the dictionary
    def get_crboxes(burn_value):
        return crbox_mapping.get(burn_value, 0)

    burn_calc["CRboxes"] = burn_calc["burn"].apply(get_crboxes)
    baseline_decay = (burn_calc["decay"].iloc[3] + burn_calc["decay"].iloc[4]) / 2
    baseline_decay_uncertainty = (
        burn_calc["decay_uncertainty"].iloc[3] ** 2
        + burn_calc["decay_uncertainty"].iloc[4] ** 2
    ) ** 0.5

    # Volume (replace with the actual volume)
    volume1 = 324
    volume2 = 33

    # Calculate CADR and CADR uncertainty with special handling for 'Burn 5' and 'Burn 6'
    def calculate_cadr_and_uncertainty(row):
        if row["burn"] in ["Burn 5", "Burn 6"]:
            # Special calculations for Burn 5 and Burn 6
            cadr = pd.NA
            cadr_uncertainty = pd.NA
        else:
            # Standard calculations for other burns
            cadr = volume1 * (row["decay"] - baseline_decay)
            cadr_uncertainty = volume1 * (
                (row["decay"] + row["decay_uncertainty"])
                - (baseline_decay - baseline_decay_uncertainty)
            ) - volume1 * (
                (row["decay"] - row["decay_uncertainty"])
                - (baseline_decay + baseline_decay_uncertainty)
            )
        return pd.Series([cadr, cadr_uncertainty])

    burn_calc[["CADR", "CADR_uncertainty"]] = burn_calc.apply(
        calculate_cadr_and_uncertainty, axis=1
    )

    # Calculate CADR per CRbox and uncertainty per CRbox
    # Handle division by zero for CR boxes
    burn_calc["CADR_per_CRbox"] = burn_calc["CADR"] / burn_calc["CRboxes"].replace(
        0, pd.NA
    )
    burn_calc["CADR_per_CRbox_uncertainty"] = burn_calc["CADR_uncertainty"] / burn_calc[
        "CRboxes"
    ].replace(0, pd.NA)

    print(burn_calc)

elif dataset == "DustTrak":
    # Convert 'decay' and 'decay_uncertainty' columns to numeric in case they are strings
    burn_calc["decay"] = pd.to_numeric(burn_calc["decay"], errors="coerce")
    burn_calc["decay_uncertainty"] = pd.to_numeric(
        burn_calc["decay_uncertainty"], errors="coerce"
    )

    # Dictionary mapping burn values (as strings) to the number of CR boxes
    crbox_mapping = {
        "Burn 1": 0,
        "Burn 2": 4,
        "Burn 3": 1,
        "Burn 4": 1,
        "Burn 5": 0,
        "Burn 6": 1,
        "Burn 7": 2,
        "Burn 8": 2,
        "Burn 9": 2,
        "Burn 10": 2,
    }

    # Function to determine CR boxes based on 'burn' value using the dictionary
    def get_crboxes(burn_value):
        return crbox_mapping.get(burn_value, 0)

    burn_calc["CRboxes"] = burn_calc["burn"].apply(get_crboxes)

    # Volume (replace with the actual volume)
    volume1 = 324
    volume2 = 33

    # Calculate CADR and CADR uncertainty with special handling for 'Burn 5' and 'Burn 6'
    def calculate_cadr_and_uncertainty(row):
        if row["burn"] in ["Burn 5", "Burn 6"]:
            # Special calculations for Burn 5 and Burn 6
            # Example: Adjust formulas as needed for these specific burns
            cadr = volume2 * (row["decay"] - burn_calc["decay"].iloc[3])
            cadr_uncertainty = volume2 * (
                (row["decay"] + row["decay_uncertainty"])
                - (burn_calc["decay"].iloc[3] - burn_calc["decay_uncertainty"].iloc[3])
            ) - volume2 * (
                (row["decay"] - row["decay_uncertainty"])
                - (burn_calc["decay"].iloc[3] + burn_calc["decay_uncertainty"].iloc[3])
            )
        else:
            # Standard calculations for other burns
            cadr = volume1 * (row["decay"] - burn_calc["decay"].iloc[0])
            cadr_uncertainty = volume1 * (
                (row["decay"] + row["decay_uncertainty"])
                - (burn_calc["decay"].iloc[0] - burn_calc["decay_uncertainty"].iloc[0])
            ) - volume1 * (
                (row["decay"] - row["decay_uncertainty"])
                - (burn_calc["decay"].iloc[0] + burn_calc["decay_uncertainty"].iloc[0])
            )
        return pd.Series([cadr, cadr_uncertainty])

    burn_calc[["CADR", "CADR_uncertainty"]] = burn_calc.apply(
        calculate_cadr_and_uncertainty, axis=1
    )

    # Calculate CADR per CRbox and uncertainty per CRbox
    # Handle division by zero for CR boxes
    burn_calc["CADR_per_CRbox"] = burn_calc["CADR"] / burn_calc["CRboxes"].replace(
        0, pd.NA
    )
    burn_calc["CADR_per_CRbox_uncertainty"] = burn_calc["CADR_uncertainty"] / burn_calc[
        "CRboxes"
    ].replace(0, pd.NA)

    print(burn_calc)

elif dataset == "PurpleAir":
    # Convert 'decay' and 'decay_uncertainty' columns to numeric in case they are strings
    burn_calc["decay"] = pd.to_numeric(burn_calc["decay"], errors="coerce")
    burn_calc["decay_uncertainty"] = pd.to_numeric(
        burn_calc["decay_uncertainty"], errors="coerce"
    )

    # Dictionary mapping burn values (as strings) to the number of CR boxes
    crbox_mapping = {
        "Burn 1": 0,
        "Burn 2": 4,
        "Burn 3": 1,
        "Burn 4": 1,
        "Burn 5": 0,
        "Burn 6": 1,
        "Burn 7": 2,
        "Burn 8": 2,
        "Burn 9": 2,
        "Burn 10": 2,
    }

    # Function to determine CR boxes based on 'burn' value using the dictionary
    def get_crboxes(burn_value):
        return crbox_mapping.get(burn_value, 0)

    burn_calc["CRboxes"] = burn_calc["burn"].apply(get_crboxes)

    # Volume (replace with the actual volume)
    volume = 324

    # Calculate CADR and CADR uncertainty
    burn_calc["CADR"] = volume * (burn_calc["decay"] - burn_calc["decay"].iloc[0])
    burn_calc["CADR_uncertainty"] = volume * (
        (burn_calc["decay"] + burn_calc["decay_uncertainty"])
        - (burn_calc["decay"].iloc[0] - burn_calc["decay_uncertainty"].iloc[0])
    ) - volume * (
        (burn_calc["decay"] - burn_calc["decay_uncertainty"])
        - (burn_calc["decay"].iloc[0] + burn_calc["decay_uncertainty"].iloc[0])
    )

    # Calculate CADR per CRbox and uncertainty per CRbox
    burn_calc["CADR_per_CRbox"] = burn_calc["CADR"] / burn_calc["CRboxes"].replace(
        0, pd.NA
    )  # Avoid division by zero
    burn_calc["CADR_per_CRbox_uncertainty"] = burn_calc["CADR_uncertainty"] / burn_calc[
        "CRboxes"
    ].replace(
        0, pd.NA
    )  # Avoid division by zero

    print(burn_calc)

elif dataset == "QuantAQb":
    # Convert 'decay' and 'decay_uncertainty' columns to numeric in case they are strings
    burn_calc["decay"] = pd.to_numeric(burn_calc["decay"], errors="coerce")
    burn_calc["decay_uncertainty"] = pd.to_numeric(
        burn_calc["decay_uncertainty"], errors="coerce"
    )

    # Dictionary mapping burn values (as strings) to the number of CR boxes
    crbox_mapping = {
        "Burn 1": 0,
        "Burn 2": 4,
        "Burn 3": 1,
        "Burn 4": 1,
        "Burn 5": 0,
        "Burn 6": 1,
        "Burn 7": 2,
        "Burn 8": 2,
        "Burn 9": 2,
        "Burn 10": 2,
    }

    # Function to determine CR boxes based on 'burn' value using the dictionary
    def get_crboxes(burn_value):
        return crbox_mapping.get(burn_value, 0)

    burn_calc["CRboxes"] = burn_calc["burn"].apply(get_crboxes)

    # these values are from the results QuantAQk Burn 5 and Burn 6 average.
    baseline_decay = 0.6414
    baseline_decay_uncertainty = 0.0054

    # Volume (replace with the actual volume)
    volume1 = 324
    volume2 = 33

    # Calculate CADR and CADR uncertainty with special handling for 'Burn 5' and 'Burn 6'
    def calculate_cadr_and_uncertainty(row):
        if row["burn"] in ["Burn 5"]:
            # Special calculations for Burn 5 and Burn 6
            cadr = pd.NA
            cadr_uncertainty = pd.NA
        elif row["burn"] in ["Burn 6"]:
            cadr = volume2 * (row["decay"] - burn_calc["decay"].iloc[1])
            cadr_uncertainty = volume2 * (
                (row["decay"] + row["decay_uncertainty"])
                - (burn_calc["decay"].iloc[1] - burn_calc["decay_uncertainty"].iloc[1])
            ) - volume2 * (
                (row["decay"] - row["decay_uncertainty"])
                - (burn_calc["decay"].iloc[1] + burn_calc["decay_uncertainty"].iloc[1])
            )
        else:
            # Standard calculations for other burns
            cadr = volume1 * (row["decay"] - baseline_decay)
            cadr_uncertainty = volume1 * (
                (row["decay"] + row["decay_uncertainty"])
                - (baseline_decay - baseline_decay_uncertainty)
            ) - volume1 * (
                (row["decay"] - row["decay_uncertainty"])
                - (baseline_decay + baseline_decay_uncertainty)
            )
        return pd.Series([cadr, cadr_uncertainty])

    burn_calc[["CADR", "CADR_uncertainty"]] = burn_calc.apply(
        calculate_cadr_and_uncertainty, axis=1
    )

    # Calculate CADR per CRbox and uncertainty per CRbox
    # Handle division by zero for CR boxes
    burn_calc["CADR_per_CRbox"] = burn_calc["CADR"] / burn_calc["CRboxes"].replace(
        0, pd.NA
    )
    burn_calc["CADR_per_CRbox_uncertainty"] = burn_calc["CADR_uncertainty"] / burn_calc[
        "CRboxes"
    ].replace(0, pd.NA)

    print(burn_calc)

elif dataset == "QuantAQk":
    # Convert 'decay' and 'decay_uncertainty' columns to numeric in case they are strings
    burn_calc["decay"] = pd.to_numeric(burn_calc["decay"], errors="coerce")
    burn_calc["decay_uncertainty"] = pd.to_numeric(
        burn_calc["decay_uncertainty"], errors="coerce"
    )

    # Dictionary mapping burn values (as strings) to the number of CR boxes
    crbox_mapping = {
        "Burn 1": 0,
        "Burn 2": 4,
        "Burn 3": 1,
        "Burn 4": 1,
        "Burn 5": 0,
        "Burn 6": 1,
        "Burn 7": 2,
        "Burn 8": 2,
        "Burn 9": 2,
        "Burn 10": 2,
    }

    # Function to determine CR boxes based on 'burn' value using the dictionary
    def get_crboxes(burn_value):
        return crbox_mapping.get(burn_value, 0)

    burn_calc["CRboxes"] = burn_calc["burn"].apply(get_crboxes)

    # Calculate baseline_decay and baseline_decay_uncertainty using values from the DataFrame
    x1 = burn_calc["decay"].iloc[1]
    u1 = burn_calc["decay_uncertainty"].iloc[1]
    x2 = burn_calc["decay"].iloc[2]
    u2 = burn_calc["decay_uncertainty"].iloc[2]

    # Calculate the weighted average and combined uncertainty
    weight1 = 1 / (u1**2)
    weight2 = 1 / (u2**2)

    baseline_decay = (x1 * weight1 + x2 * weight2) / (weight1 + weight2)
    baseline_decay_uncertainty = np.sqrt(1 / (weight1 + weight2))

    # Print resulting baseline_decay and baseline_decay_uncertainty for verification
    print(f"Baseline Decay: {baseline_decay:.4f} ± {baseline_decay_uncertainty:.4f}")

    # Volume (replace with the actual volume)
    volume1 = 324
    volume2 = 33

    # Calculate CADR and CADR uncertainty with special handling for 'Burn 5' and 'Burn 6'
    def calculate_cadr_and_uncertainty(row):
        if row["burn"] in ["Burn 5", "Burn 6"]:
            # Special calculations for Burn 5 and Burn 6
            cadr = pd.NA
            cadr_uncertainty = pd.NA
        else:
            # Standard calculations for other burns
            cadr = volume1 * (row["decay"] - baseline_decay)
            cadr_uncertainty = volume1 * (
                (row["decay"] + row["decay_uncertainty"])
                - (baseline_decay - baseline_decay_uncertainty)
            ) - volume1 * (
                (row["decay"] - row["decay_uncertainty"])
                - (baseline_decay + baseline_decay_uncertainty)
            )
        return pd.Series([cadr, cadr_uncertainty])

    burn_calc[["CADR", "CADR_uncertainty"]] = burn_calc.apply(
        calculate_cadr_and_uncertainty, axis=1
    )

    # Calculate CADR per CRbox and uncertainty per CRbox
    # Handle division by zero for CR boxes
    burn_calc["CADR_per_CRbox"] = burn_calc["CADR"] / burn_calc["CRboxes"].replace(
        0, pd.NA
    )
    burn_calc["CADR_per_CRbox_uncertainty"] = burn_calc["CADR_uncertainty"] / burn_calc[
        "CRboxes"
    ].replace(0, pd.NA)

    print(burn_calc)

elif dataset == "SMPS":
    # Convert 'decay' and 'decay_uncertainty' columns to numeric in case they are strings
    burn_calc["decay"] = pd.to_numeric(burn_calc["decay"], errors="coerce")
    burn_calc["decay_uncertainty"] = pd.to_numeric(
        burn_calc["decay_uncertainty"], errors="coerce"
    )

    # Dictionary mapping burn values (as strings) to the number of CR boxes
    crbox_mapping = {
        "Burn 1": 0,
        "Burn 2": 4,
        "Burn 3": 1,
        "Burn 4": 1,
        "Burn 5": 0,
        "Burn 6": 1,
        "Burn 7": 2,
        "Burn 8": 2,
        "Burn 9": 2,
        "Burn 10": 2,
    }

    # Function to determine CR boxes based on 'burn' value using the dictionary
    def get_crboxes(burn_value):
        return crbox_mapping.get(burn_value, 0)

    burn_calc["CRboxes"] = burn_calc["burn"].apply(get_crboxes)

    # Volume (replace with the actual volume)
    volume1 = 324
    volume2 = 33

    # Calculate CADR and CADR uncertainty with special handling for 'Burn 5' and 'Burn 6'
    def calculate_cadr_and_uncertainty(row):
        if row["burn"] in ["Burn 5", "Burn 6"]:
            # Special calculations for Burn 5 and Burn 6
            # Example: Adjust formulas as needed for these specific burns
            cadr = volume2 * (row["decay"] - burn_calc["decay"].iloc[4])
            cadr_uncertainty = volume2 * (
                (row["decay"] + row["decay_uncertainty"])
                - (burn_calc["decay"].iloc[4] - burn_calc["decay_uncertainty"].iloc[4])
            ) - volume2 * (
                (row["decay"] - row["decay_uncertainty"])
                - (burn_calc["decay"].iloc[4] + burn_calc["decay_uncertainty"].iloc[4])
            )
        else:
            # Standard calculations for other burns
            cadr = volume1 * (row["decay"] - burn_calc["decay"].iloc[0])
            cadr_uncertainty = volume1 * (
                (row["decay"] + row["decay_uncertainty"])
                - (burn_calc["decay"].iloc[0] - burn_calc["decay_uncertainty"].iloc[0])
            ) - volume1 * (
                (row["decay"] - row["decay_uncertainty"])
                - (burn_calc["decay"].iloc[0] + burn_calc["decay_uncertainty"].iloc[0])
            )
        return pd.Series([cadr, cadr_uncertainty])

    burn_calc[["CADR", "CADR_uncertainty"]] = burn_calc.apply(
        calculate_cadr_and_uncertainty, axis=1
    )

    # Calculate CADR per CRbox and uncertainty per CRbox
    # Handle division by zero for CR boxes
    burn_calc["CADR_per_CRbox"] = burn_calc["CADR"] / burn_calc["CRboxes"].replace(
        0, pd.NA
    )
    burn_calc["CADR_per_CRbox_uncertainty"] = burn_calc["CADR_uncertainty"] / burn_calc[
        "CRboxes"
    ].replace(0, pd.NA)

    print(burn_calc)

# %% used to make the PurpleAir burn plots
# For visualization in Jupyter Notebook
# output_notebook()
# For visualization to static HTML
output_file("./Paper_figures/PurpleAir_PM2.5_concentrations.html")

# Define fixed ranges for the axes
x_start = -50
x_end = 200  # Adjust as needed
y_start = 0
y_end = 2600  # Adjust as needed

# Create a list to hold the plots
plots = []

# Calculate time differences
time_differences = {}

# Generate a plot for each burn
for sheet_name in sorted(dfs_garage.keys()):
    dfg = dfs_garage[sheet_name]
    dfk = dfs_kitchen[sheet_name]

    # Convert DataFrames to ColumnDataSource
    source_garage = ColumnDataSource(dfg)
    source_kitchen = ColumnDataSource(dfk)

    # Create a new plot with specified x and y ranges
    p = figure(
        title=f"Purple Air {sheet_name}",
        # x_axis_label='Minutes Since Peak',
        # y_axis_label='Average Concentration (µg/m³)',
        width=300,
        height=300,
        x_range=(x_start, x_end),
        y_range=(y_start, y_end),
    )

    # Plot data for garage
    p.line(
        source_garage.data["min_since_peak"],
        source_garage.data["Average Concentration(µg/m³)"],
        legend_label="Garage",
        color="blue",
        line_width=1.5,
    )

    # Plot data for kitchen
    p.line(
        source_kitchen.data["min_since_peak"],
        source_kitchen.data["Average Concentration(µg/m³)"],
        legend_label="Morning Room",
        color="green",
        line_width=1.5,
    )

    # Add the plot to the list
    plots.append(p)

    # Find the peak values for both garage and kitchen
    peak_garage = dfg.loc[dfg["Average Concentration(µg/m³)"].idxmax()]
    peak_kitchen = dfk.loc[dfk["Average Concentration(µg/m³)"].idxmax()]

    # Calculate the time difference (in minutes) between peak values
    time_difference = peak_kitchen["min_since_peak"] - peak_garage["min_since_peak"]

    # Store the result in the dictionary
    time_differences[sheet_name] = time_difference

    # Print the time difference to the terminal
    print(f"{sheet_name} Peak Conc. time delay: {time_difference:.2f} minutes")

# Reorganize plots: shift all plots one position to the left
shifted_plots = plots[1:] + [plots[0]]

# Creating individual plot lists
row1_plots = shifted_plots[0:3]  # Plot 1, 2, 3 (originally 2, 3, 4)
column2_plots = [
    shifted_plots[3],
    shifted_plots[4],
]  # Plot 4 and Plot 5 (originally 5, 6)

# Create the layout
layout = column(
    row(*row1_plots), row(column2_plots[0], column2_plots[1], sizing_mode="scale_width")
)

# Show the layout
show(layout)

# export_png(layout, filename="./Paper_figures/PurpleAir_PM2.5_concentrations.png")

# %% RUN to plot the AeroTrakK data.
df_burn2 = df.filter(like="burn2")

# For visualization in Jupyter Notebook
# output_notebook()
# For visualization to static HTML
output_file("./Paper_figures/AeroTrak_Burn2_Kitchen_Decacy.html")

# Extract the minutes and data series
minutes_np = df["min_from_peak"].values
time_hours_np = minutes_np / 60  # Convert minutes to hours
decay_data = {
    "Ch1 0.3µm (#/cm³)": df_burn2["burn2 Ch1 0.3µm (#/cm³)"].values,
    "Ch2 0.5µm (#/cm³)": df_burn2["burn2 Ch2 0.5µm (#/cm³)"].values,
    "Ch3 1.0µm (#/cm³)": df_burn2["burn2 Ch3 1.0µm (#/cm³)"].values,
    "Ch4 3.0µm (#/cm³)": df_burn2["burn2 Ch4 3.0µm (#/cm³)"].values,
    "Ch5 5.0µm (#/cm³)": df_burn2["burn2 Ch5 5.0µm (#/cm³)"].values,
    "Ch6 10.0µm (#/cm³)": df_burn2["burn2 Ch6 10.0µm (#/cm³)"].values,
    "total_count (#/cm³)": df_burn2["burn2 total_count (#/cm³)"].values,
}

# Define different ranges for fitting each series in minutes
fit_ranges = {"total_count (#/cm³)": (8, 52)}

# Define specific label positions for each series
label_positions = {"total_count (#/cm³)": (40, 0)}

# Define custom line colors and line types for each series
line_properties = {
    "Ch1 0.3µm (#/cm³)": {"color": "#f8aa39", "line_dash": "solid", "line_width": 1},
    "Ch2 0.5µm (#/cm³)": {"color": "#f1ae58", "line_dash": "solid", "line_width": 1},
    "Ch3 1.0µm (#/cm³)": {"color": "#e8b274", "line_dash": "solid", "line_width": 1},
    "Ch4 3.0µm (#/cm³)": {"color": "#ddb78d", "line_dash": "solid", "line_width": 1},
    "Ch5 5.0µm (#/cm³)": {"color": "#d0bba7", "line_dash": "solid", "line_width": 1},
    "Ch6 10.0µm (#/cm³)": {"color": "#c0c0c0", "line_dash": "solid", "line_width": 1},
    "total_count (#/cm³)": {"color": "#ffa500", "line_dash": "solid", "line_width": 2},
}

# Define the x-axis and y-axis ranges
x_range_start = -60
x_range_end = 180  # Cover an additional hour for a full view
y_range_start = 0.0001  # Avoid log scale issues by not starting at zero
y_range_end = 10**4  # Example upper limit for the y-axis


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
    # title="AeroTrak Burn 2 Morning Room Decay",
    x_axis_label="Time since peak conc. after partical growth (minutes)",
    y_axis_label="Particulate Matter (number count)",
    y_axis_type="log",
    x_axis_type="linear",
    x_range=(x_range_start, x_range_end),
    y_range=(y_range_start, y_range_end),
    max_width=600,
    height=800,
)

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

    # Check if the current label is in fit_ranges
    if label in fit_ranges:
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
        x_offset, y_offset = label_positions.get(label, (0, 0))
        label_x = minutes_np[fit_end_index - 1] + x_offset  # Move label to the right
        label_y = (
            y_curve_fit[-1] + y_offset
        )  # Move label slightly above the fit line end

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

# Adjust the legend position to the lower-left corner
p.legend.location = "top_right"
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
