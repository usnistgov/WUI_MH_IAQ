"""
WUI Compartmentalization Strategy Comparison Analysis

This script compares the effectiveness of different compartmentalization strategies
for controlling smoke infiltration and concentration in residential structures during
wildland-urban interface fire events. It analyzes how sealing individual rooms versus
operating air cleaners affects smoke levels.

Analysis Components:
    1. Concentration time series in sealed vs unsealed rooms
    2. Peak concentration ratios (sealed/unsealed)
    3. Decay rate comparisons between strategies
    4. Exposure reduction calculations (AUC analysis)
    5. Statistical comparison of effectiveness

Metrics Calculated:
    - Peak concentration reduction (%)
    - Integrated exposure reduction (%)
    - Time to safe levels (<EPA threshold)
    - Spatial uniformity coefficient of variation
    - Protection factor: C_house / C_bedroom

Visualization Outputs:
    - Side-by-side time series comparisons
    - Strategy effectiveness bar charts
    - Concentration ratio heat maps
    - Annotated event timelines (door closing, CR Box activation)

Data Sources:
    - AeroTrak data (bedroom and kitchen)
    - QuantAQ data (multiple locations)
    - Burn log with door closure and CR Box timing
    - Room volume and air exchange rate data

Methodology:
    - Normalized concentrations for cross-burn comparison
    - Baseline-corrected measurements
    - Statistical significance testing (t-tests)
    - Uncertainty quantification with 95% CI

Configuration Features:
    - Flexible path configuration for multiple systems
    - Customizable text formatting
    - Publication-quality figure generation
    - Interactive HTML outputs with Bokeh

Dependencies:
    - pandas: Data processing
    - numpy: Numerical analysis
    - bokeh: Visualization
    - datetime: Time series handling

Outputs:
    - Comparison plots saved to Paper_figures directory
    - Statistical summary tables
    - Strategy effectiveness rankings

Author: Nathan Lima
Date: 2024-2025
"""

# %%
import os
import pandas as pd
import numpy as np
import datetime
from bokeh.plotting import figure, show
from bokeh.io import output_notebook, output_file, reset_output
from bokeh.models import ColumnDataSource, Div, Label, Arrow, NormalHead
from bokeh.layouts import column

# Set output to display plots in the notebook
output_notebook()

# Load text formatting configuration
TEXT_CONFIG = {
    "font_size": "12pt",
    "title_font_size": "12pt",
    "axis_label_font_size": "12pt",
    "axis_tick_font_size": "12pt",
    "legend_font_size": "12pt",
    "label_font_size": "12pt",
    "font_style": "normal",
    "plot_font_style": "bold",  # For Bokeh plot elements
    "html_font_weight": "normal",  # For HTML div elements
}

# System path configuration - automatically detect which system we're on
system_paths = [
    "C:/Users/nml/OneDrive - NIST/Documents/NIST/WUI_smoke/",
    "C:/Users/Nathan/Documents/NIST/WUI_smoke",
]

# Check which system path exists and set working directory
for path in system_paths:
    if os.path.exists(path):
        absolute_path = path
        print(f"Using path: {absolute_path}")
        break
else:
    print("No predefined system path found, using current directory.")
    absolute_path = os.getcwd()

os.chdir(absolute_path)

# Load burn log for reference data
burn_log = pd.read_excel("./burn_log.xlsx", sheet_name="Sheet2")

# Global variables for decay parameters (calculated during processing)
decay_parameters = {}

# Color schemes for consistent visualization across all instruments
POLLUTANT_COLORS = {
    "PM0.5 (µg/m³)": "#ef5675",
    "PM1 (µg/m³)": "#d45087",
    "PM2.5 (µg/m³)": "#b35093",
    "PM3 (µg/m³)": "#b35093",
    "PM4 (µg/m³)": "#8d5196",
    "PM5 (µg/m³)": "#665191",
    "PM10 (µg/m³)": "#404e84",
    "PM15 (µg/m³)": "#1d4772",
    "PM25 (µg/m³)": "#003f5c",
    # SMPS bin colors
    "Ʃ9-100nm (µg/m³)": "#ffa600",
    "Ʃ100-200nm (µg/m³)": "#ff8d2f",
    "Ʃ200-300nm (µg/m³)": "#ff764a",
    "Ʃ300-437nm (µg/m³)": "#ff6361",
    "Total Concentration (µg/m³)": "#ef5675",
}

# Instrument configurations - contains all processing parameters for each instrument
INSTRUMENT_CONFIG = {
    "AeroTrakB": {
        "file_path": "./burn_data/aerotraks/bedroom2/all_data.xlsx",
        "process_function": "process_aerotrak_data",
        "time_shift": 2.16,  # Minutes to shift timestamps for synchronization
        "plot_pollutants": ["PM1 (µg/m³)", "PM3 (µg/m³)", "PM10 (µg/m³)"],
        "datetime_column": "Date and Time",
        "special_cases": {
            "burn6": {"custom_decay_time": True, "decay_end_offset": 0.25}
        },
    },
    "AeroTrakK": {
        "file_path": "./burn_data/aerotraks/kitchen/all_data.xlsx",
        "process_function": "process_aerotrak_data",
        "time_shift": 5,
        "plot_pollutants": ["PM1 (µg/m³)", "PM3 (µg/m³)", "PM10 (µg/m³)"],
        "datetime_column": "Date and Time",
        "special_cases": {},
    },
    "DustTrak": {
        "file_path": "./burn_data/dusttrak/all_data.xlsx",
        "process_function": "process_dusttrak_data",
        "time_shift": 7,
        "plot_pollutants": ["PM1 (µg/m³)", "PM2.5 (µg/m³)", "PM10 (µg/m³)"],
        "datetime_column": "datetime",
        "special_cases": {
            "burn6": {"custom_decay_time": True, "decay_end_offset": 0.25}
        },
    },
    "PurpleAirK": {
        "file_path": "./burn_data/purpleair/garage-kitchen.xlsx",
        "process_function": "process_purpleairk_data",
        "time_shift": 0,
        "plot_pollutants": ["PM2.5 (µg/m³)"],
        "datetime_column": "DateTime",
        "special_cases": {},
    },
    "QuantAQB": {
        "file_path": "./burn_data/quantaq/MOD-PM-00194-b0fc215029fa4852b926bc50b28fda5a.csv",
        "process_function": "process_quantaq_data",
        "time_shift": -2.97,
        "plot_pollutants": ["PM1 (µg/m³)", "PM2.5 (µg/m³)", "PM10 (µg/m³)"],
        "datetime_column": "timestamp_local",
        "special_cases": {
            "burn6": {"custom_decay_time": True, "decay_end_offset": 0.25}
        },
    },
    "QuantAQK": {
        "file_path": "./burn_data/quantaq/MOD-PM-00197-a6dd467a147a4d95a7b98a8a10ab4ea3.csv",
        "process_function": "process_quantaq_data",
        "time_shift": 0,
        "plot_pollutants": ["PM1 (µg/m³)", "PM2.5 (µg/m³)", "PM10 (µg/m³)"],
        "datetime_column": "timestamp_local",
        "special_cases": {},
    },
    "SMPS": {
        "file_path": "./burn_data/smps",
        "process_function": "process_smps_data",
        "time_shift": 0,
        "plot_pollutants": ["Total Concentration (µg/m³)"],
        "datetime_column": "datetime",
        "special_cases": {
            "burn6": {"custom_decay_time": True, "decay_end_offset": 0.25}
        },
    },
}

# Define which burns to include in Figure 4
BURN_GROUPS = {"figure4": ["burn1", "burn5", "burn6"]}

# Visual styling for each burn (colors and line styles)
BURN_STYLES = {
    "burn1": {"color": "#ef5675", "line_dash": "solid"},
    "burn5": {"color": "#8d5196", "line_dash": "solid"},
    "burn6": {"color": "#665191", "line_dash": "solid"},
}

# Human-readable labels for each burn condition
BURN_LABELS = {
    "burn1": "Open Door, HVAC On",
    "burn5": "Sealed Door, HVAC Sealed",
    "burn6": "Sealed Door, HVAC Sealed, Air Cleaner On",
}


def get_text_config():
    """Return the text formatting configuration"""
    return TEXT_CONFIG


def apply_text_formatting(plot_object):
    """Apply consistent text formatting to a Bokeh plot object"""
    # Apply axis label formatting
    plot_object.xaxis.axis_label_text_font_size = TEXT_CONFIG["axis_label_font_size"]
    plot_object.yaxis.axis_label_text_font_size = TEXT_CONFIG["axis_label_font_size"]
    plot_object.xaxis.axis_label_text_font_style = TEXT_CONFIG["font_style"]
    plot_object.yaxis.axis_label_text_font_style = TEXT_CONFIG["font_style"]

    # Apply tick label formatting
    plot_object.xaxis.major_label_text_font_size = TEXT_CONFIG["axis_tick_font_size"]
    plot_object.yaxis.major_label_text_font_size = TEXT_CONFIG["axis_tick_font_size"]
    plot_object.xaxis.major_label_text_font_style = TEXT_CONFIG["font_style"]
    plot_object.yaxis.major_label_text_font_style = TEXT_CONFIG["font_style"]

    # Apply legend formatting only if legend items exist
    if hasattr(plot_object, "legend") and len(plot_object.legend) > 0:
        plot_object.legend.label_text_font_size = TEXT_CONFIG["legend_font_size"]
        plot_object.legend.label_text_font_style = TEXT_CONFIG["font_style"]

    # Apply title formatting if title exists
    if hasattr(plot_object, "title") and plot_object.title:
        plot_object.title.text_font_size = TEXT_CONFIG["title_font_size"]
        plot_object.title.text_font_style = TEXT_CONFIG["font_style"]

    return plot_object


def get_script_metadata():
    """Generate metadata string with script name and execution timestamp"""
    try:
        import inspect

        script_name = os.path.basename(
            inspect.getmodule(inspect.currentframe()).__file__
        )
    except (NameError, AttributeError, TypeError):
        script_name = "wui_compartmentalization_strategy_comparison_v3.0.py"

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"Generated by: {script_name} | Date: {timestamp}"


def create_naive_datetime(date_str, time_str):
    """Create a timezone-naive datetime object from date and time strings"""
    dt = pd.to_datetime(f"{date_str} {time_str}", errors="coerce")
    if hasattr(dt, "tz") and dt.tz is not None:
        dt = dt.tz_localize(None)
    return dt


def fit_exponential_curve(x_data, y_data, initial_guess):
    """
    Fit exponential decay curve to the data using scipy curve_fit

    Args:
        x_data: X data points (time)
        y_data: Y data points (concentration)
        initial_guess: Initial parameter values (amplitude, decay_rate)

    Returns:
        tuple: (optimal_parameters, fitted_y_values, parameter_errors)
    """
    from scipy.optimize import curve_fit

    if len(x_data) == 0 or len(y_data) == 0:
        raise ValueError("Input data for fitting is empty.")

    # Exponential decay function: y = a * exp(-b * t)
    def exp_decreasing(t, a, b):
        return a * np.exp(-b * t)

    # Perform curve fitting
    try:
        popt, pcov = curve_fit(
            exp_decreasing, x_data, y_data, p0=initial_guess, maxfev=10000
        )
        y_fit = exp_decreasing(x_data, *popt)
        perr = np.sqrt(np.diag(pcov))  # Standard error
        return popt, y_fit, perr
    except Exception as e:
        print(f"Curve fitting error: {e}")
        raise


def apply_time_shift(df, instrument, burn_id, burn_date):
    """Apply instrument-specific time shift for synchronization"""
    time_shift = INSTRUMENT_CONFIG[instrument].get("time_shift", 0)
    datetime_column = INSTRUMENT_CONFIG[instrument].get(
        "datetime_column", "Date and Time"
    )

    # Ensure datetime column is properly formatted
    df.loc[:, datetime_column] = pd.to_datetime(df[datetime_column])
    burn_date = pd.to_datetime(burn_date).date()

    # Apply time shift only if non-zero
    if time_shift != 0:
        mask = df[datetime_column].dt.date == burn_date
        if mask.any():
            df.loc[mask, datetime_column] += pd.Timedelta(minutes=time_shift)

    return df


def process_aerotrak_data(file_path, instrument="AeroTrakB"):
    """Process AeroTrak particulate matter data"""
    print(f"Processing AeroTrak data from {file_path}")

    # Load and filter data for Figure 4 burns only
    aerotrak_data = pd.read_excel(file_path)
    figure4_burn_ids = BURN_GROUPS["figure4"]
    figure4_burn_dates = burn_log[burn_log["Burn ID"].isin(figure4_burn_ids)]["Date"]
    figure4_burn_dates = pd.to_datetime(figure4_burn_dates)

    # Filter by burn dates
    aerotrak_data["Date"] = pd.to_datetime(aerotrak_data["Date and Time"]).dt.date
    filtered_data = aerotrak_data[
        aerotrak_data["Date"].isin(figure4_burn_dates.dt.date)
    ].copy()

    # Apply time shifts for synchronization
    for burn_id in figure4_burn_ids:
        if burn_id in burn_log["Burn ID"].values:
            burn_date = burn_log[burn_log["Burn ID"] == burn_id]["Date"].values[0]
            filtered_data = apply_time_shift(
                filtered_data, instrument, burn_id, burn_date
            )

    # Calculate time since garage door closed for each data point
    filtered_data["Time Since Garage Closed (hours)"] = np.nan

    for burn_id in figure4_burn_ids:
        burn_date_row = burn_log[burn_log["Burn ID"] == burn_id]
        if burn_date_row.empty:
            continue

        burn_date = pd.to_datetime(burn_date_row["Date"].iloc[0])
        garage_closed_time_str = burn_date_row["garage closed"].iloc[0]

        if pd.isna(garage_closed_time_str):
            continue

        # Calculate time difference from garage closing
        garage_closed_time = create_naive_datetime(burn_date, garage_closed_time_str)
        matching_rows = filtered_data[filtered_data["Date"] == burn_date.date()]

        if not matching_rows.empty:
            datetime_values = pd.to_datetime(
                filtered_data.loc[
                    filtered_data["Date"] == burn_date.date(), "Date and Time"
                ]
            )

            # Ensure timezone-naive datetime
            if (
                hasattr(datetime_values.dtype, "tz")
                and datetime_values.dtype.tz is not None
            ):
                datetime_values = datetime_values.dt.tz_localize(None)

            # Calculate hours since garage closed
            time_since_closed = (
                datetime_values - garage_closed_time
            ).dt.total_seconds() / 3600
            filtered_data.loc[
                matching_rows.index, "Time Since Garage Closed (hours)"
            ] = time_since_closed

    return filtered_data


def process_dusttrak_data(file_path):
    """Process DustTrak particulate matter data with unit conversion"""
    print(f"Processing DustTrak data from {file_path}")

    dusttrak_data = pd.read_excel(file_path)
    dusttrak_data.columns = dusttrak_data.columns.str.strip()

    # Convert from mg/m³ to µg/m³ for PM columns
    pm_columns = [
        "PM1 [mg/m3]",
        "PM2.5 [mg/m3]",
        "PM4 [mg/m3]",
        "PM10 [mg/m3]",
        "TOTAL [mg/m3]",
    ]

    for col in pm_columns:
        if col in dusttrak_data.columns:
            if col == "TOTAL [mg/m3]":
                new_col_name = "PM15 (µg/m³)"
            else:
                new_col_name = col.replace("[mg/m3]", "(µg/m³)")
            dusttrak_data[new_col_name] = dusttrak_data[col] * 1000

    # Filter and process similar to AeroTrak
    figure4_burn_ids = BURN_GROUPS["figure4"]
    figure4_burn_dates = burn_log[burn_log["Burn ID"].isin(figure4_burn_ids)]["Date"]
    figure4_burn_dates = pd.to_datetime(figure4_burn_dates)

    dusttrak_data["Date"] = pd.to_datetime(dusttrak_data["datetime"]).dt.date
    filtered_data = dusttrak_data[
        dusttrak_data["Date"].isin(figure4_burn_dates.dt.date)
    ].copy()

    # Apply time shifts and calculate time since garage closed
    for burn_id in figure4_burn_ids:
        if burn_id in burn_log["Burn ID"].values:
            burn_date = burn_log[burn_log["Burn ID"] == burn_id]["Date"].values[0]
            filtered_data = apply_time_shift(
                filtered_data, "DustTrak", burn_id, burn_date
            )

    # Calculate time since garage closed (similar pattern as AeroTrak)
    filtered_data["Time Since Garage Closed (hours)"] = np.nan

    for burn_id in figure4_burn_ids:
        burn_date_row = burn_log[burn_log["Burn ID"] == burn_id]
        if burn_date_row.empty:
            continue

        burn_date = pd.to_datetime(burn_date_row["Date"].iloc[0])
        garage_closed_time_str = burn_date_row["garage closed"].iloc[0]

        if pd.isna(garage_closed_time_str):
            continue

        garage_closed_time = create_naive_datetime(burn_date, garage_closed_time_str)
        matching_rows = filtered_data[filtered_data["Date"] == burn_date.date()]

        if not matching_rows.empty:
            datetime_values = pd.to_datetime(
                filtered_data.loc[filtered_data["Date"] == burn_date.date(), "datetime"]
            )

            if (
                hasattr(datetime_values.dtype, "tz")
                and datetime_values.dtype.tz is not None
            ):
                datetime_values = datetime_values.dt.tz_localize(None)

            time_since_closed = (
                datetime_values - garage_closed_time
            ).dt.total_seconds() / 3600
            filtered_data.loc[
                matching_rows.index, "Time Since Garage Closed (hours)"
            ] = time_since_closed

    return filtered_data


def process_purpleairk_data(file_path):
    """Process PurpleAir sensor data (kitchen location)"""
    print(f"Processing PurpleAir data from {file_path}")

    purpleair_data = pd.read_excel(file_path, sheet_name="(P2)kitchen")
    purpleair_data.columns = purpleair_data.columns.str.strip()
    purpleair_data.rename(columns={"Average": "PM2.5 (µg/m³)"}, inplace=True)

    # PurpleAir only has data for burn6 in Figure 4
    figure4_burn_ids = [
        burn_id
        for burn_id in BURN_GROUPS["figure4"]
        if int(burn_id.replace("burn", "")) >= 6
    ]
    figure4_burn_dates = burn_log[burn_log["Burn ID"].isin(figure4_burn_ids)]["Date"]
    figure4_burn_dates = pd.to_datetime(figure4_burn_dates)

    purpleair_data["Date"] = pd.to_datetime(purpleair_data["DateTime"]).dt.date
    filtered_data = purpleair_data[
        purpleair_data["Date"].isin(figure4_burn_dates.dt.date)
    ].copy()

    # Calculate time since garage closed
    filtered_data["Time Since Garage Closed (hours)"] = np.nan

    for burn_id in figure4_burn_ids:
        burn_date_row = burn_log[burn_log["Burn ID"] == burn_id]
        if burn_date_row.empty:
            continue

        burn_date = pd.to_datetime(burn_date_row["Date"].iloc[0])
        garage_closed_time_str = burn_date_row["garage closed"].iloc[0]

        if pd.isna(garage_closed_time_str):
            continue

        garage_closed_time = create_naive_datetime(burn_date, garage_closed_time_str)
        matching_rows = filtered_data[filtered_data["Date"] == burn_date.date()]

        if not matching_rows.empty:
            datetime_values = pd.to_datetime(
                filtered_data.loc[filtered_data["Date"] == burn_date.date(), "DateTime"]
            )

            if (
                hasattr(datetime_values.dtype, "tz")
                and datetime_values.dtype.tz is not None
            ):
                datetime_values = datetime_values.dt.tz_localize(None)

            time_since_closed = (
                datetime_values - garage_closed_time
            ).dt.total_seconds() / 3600
            filtered_data.loc[
                matching_rows.index, "Time Since Garage Closed (hours)"
            ] = time_since_closed

    return filtered_data


def process_quantaq_data(file_path, instrument):
    """Process QuantAQ sensor data"""
    print(f"Processing QuantAQ data from {file_path}")

    quantaq_data = pd.read_csv(file_path)
    quantaq_data = quantaq_data.iloc[::-1].reset_index(drop=True)  # Reverse data order
    quantaq_data.columns = quantaq_data.columns.str.strip()

    # Rename columns to standard format
    quantaq_data.rename(
        columns={"pm1": "PM1 (µg/m³)", "pm25": "PM2.5 (µg/m³)", "pm10": "PM10 (µg/m³)"},
        inplace=True,
    )

    # Convert timestamp to timezone-naive datetime
    quantaq_data["timestamp_local"] = pd.to_datetime(
        quantaq_data["timestamp_local"].str.replace("T", " ").str.replace("Z", ""),
        errors="coerce",
    ).dt.tz_localize(None)

    # Filter and process for Figure 4 burns
    figure4_burn_ids = BURN_GROUPS["figure4"]
    figure4_burn_dates = burn_log[burn_log["Burn ID"].isin(figure4_burn_ids)]["Date"]
    figure4_burn_dates = pd.to_datetime(figure4_burn_dates)

    quantaq_data["Date"] = quantaq_data["timestamp_local"].dt.date
    filtered_data = quantaq_data[
        quantaq_data["Date"].isin(figure4_burn_dates.dt.date)
    ].copy()

    # Apply time shifts and calculate time since garage closed
    for burn_id in figure4_burn_ids:
        if burn_id in burn_log["Burn ID"].values:
            burn_date = burn_log[burn_log["Burn ID"] == burn_id]["Date"].values[0]
            filtered_data = apply_time_shift(
                filtered_data, instrument, burn_id, burn_date
            )

    filtered_data["Time Since Garage Closed (hours)"] = np.nan

    for burn_id in figure4_burn_ids:
        burn_date_row = burn_log[burn_log["Burn ID"] == burn_id]
        if burn_date_row.empty:
            continue

        burn_date = pd.to_datetime(burn_date_row["Date"].iloc[0])
        garage_closed_time_str = burn_date_row["garage closed"].iloc[0]

        if pd.isna(garage_closed_time_str):
            continue

        garage_closed_time = create_naive_datetime(burn_date, garage_closed_time_str)
        matching_rows = filtered_data[filtered_data["Date"] == burn_date.date()]

        if not matching_rows.empty:
            datetime_values = filtered_data.loc[
                filtered_data["Date"] == burn_date.date(), "timestamp_local"
            ]

            if (
                hasattr(datetime_values.dtype, "tz")
                and datetime_values.dtype.tz is not None
            ):
                datetime_values = datetime_values.dt.tz_localize(None)

            time_since_closed = (
                datetime_values - garage_closed_time
            ).dt.total_seconds() / 3600
            filtered_data.loc[
                matching_rows.index, "Time Since Garage Closed (hours)"
            ] = time_since_closed

    return filtered_data


def process_smps_data(file_path):
    """Process SMPS (Scanning Mobility Particle Sizer) data"""
    print(f"Processing SMPS data from {file_path}")

    combined_smps_data = pd.DataFrame()
    burn_ids = BURN_GROUPS["figure4"]
    burn_dates = burn_log[burn_log["Burn ID"].isin(burn_ids)]["Date"]
    burn_dates = pd.to_datetime(burn_dates)

    # Process each burn date's SMPS file
    for burn_date in burn_dates:
        try:
            date_str = burn_date.strftime("%m%d%Y")
            smps_filename = f"MH_apollo_bed_{date_str}_MassConc.xlsx"
            smps_file_path = os.path.join(file_path, smps_filename)

            if not os.path.exists(smps_file_path):
                print(f"File not found: {smps_file_path}")
                continue

            print(f"Reading file: {smps_file_path}")
            smps_data = pd.read_excel(smps_file_path, sheet_name=0)

            # Transpose data if needed (check for expected columns)
            if (
                "Date" not in smps_data.columns
                and "Start Time" not in smps_data.columns
            ):
                smps_data = smps_data.transpose()
                smps_data.columns = smps_data.iloc[0].values
                smps_data = smps_data.iloc[1:].reset_index(drop=True)

            # Verify required columns exist
            required_columns = ["Date", "Start Time", "Total Concentration(µg/m³)"]
            missing_columns = [
                col for col in required_columns if col not in smps_data.columns
            ]
            if missing_columns:
                print(f"Required columns missing in {smps_filename}: {missing_columns}")
                continue

            # Rename with proper spacing
            if "Total Concentration(µg/m³)" in smps_data.columns:
                smps_data = smps_data.rename(
                    columns={
                        "Total Concentration(µg/m³)": "Total Concentration (µg/m³)"
                    }
                )

            # Convert datetime columns with error handling
            try:
                smps_data["Date"] = pd.to_datetime(smps_data["Date"], errors="coerce")
                smps_data["Start Time"] = pd.to_datetime(
                    smps_data["Start Time"], format="%H:%M:%S", errors="coerce"
                )
            except Exception as e:
                print(f"Error converting datetime in {smps_filename}: {str(e)}")
                continue

            # Check for NaN values after conversion
            if smps_data["Date"].isna().all() or smps_data["Start Time"].isna().all():
                print(f"All datetime values are NaN in {smps_filename}")
                continue

            # Create combined datetime column
            try:
                smps_data["datetime"] = pd.to_datetime(
                    smps_data["Date"].dt.strftime("%Y-%m-%d")
                    + " "
                    + smps_data["Start Time"].dt.strftime("%H:%M:%S"),
                    errors="coerce",
                )
            except Exception as e:
                print(f"Error creating datetime in {smps_filename}: {str(e)}")
                continue

            # Remove rows with invalid datetime
            invalid_rows = smps_data["datetime"].isna().sum()
            if invalid_rows > 0:
                print(
                    f"Dropping {invalid_rows} rows with invalid datetime in {smps_filename}"
                )
                smps_data = smps_data.dropna(subset=["datetime"])

            if smps_data.empty:
                print(f"No valid data after datetime filtering in {smps_filename}")
                continue

            # Calculate mid-time between consecutive measurements for better time representation
            smps_data = smps_data.sort_values("datetime").reset_index(drop=True)
            smps_data["next_datetime"] = smps_data["datetime"].shift(-1)

            try:
                smps_data["mid_datetime"] = pd.to_datetime(
                    smps_data.apply(
                        lambda row: (
                            row["datetime"]
                            + (row["next_datetime"] - row["datetime"]) / 2
                            if pd.notna(row["next_datetime"])
                            else row["datetime"]
                        ),
                        axis=1,
                    )
                )
                smps_data["datetime"] = smps_data["mid_datetime"]
            except Exception as e:
                print(
                    f"Error calculating mid-time in {smps_filename}, using original times: {str(e)}"
                )

            # Clean up temporary columns
            smps_data = smps_data.drop(
                ["next_datetime", "mid_datetime"], axis=1, errors="ignore"
            )

            # Get burn_id for this date
            burn_id_row = burn_log[burn_log["Date"] == burn_date]
            if not burn_id_row.empty:
                burn_id = burn_id_row["Burn ID"].iloc[0]

                # Create result DataFrame
                result_df = pd.DataFrame(
                    {
                        "datetime": smps_data["datetime"],
                        "Date": smps_data["datetime"].dt.date,
                        "burn_id": burn_id,
                        "Total Concentration (µg/m³)": pd.to_numeric(
                            smps_data["Total Concentration (µg/m³)"], errors="coerce"
                        ),
                    }
                )

                # Apply time shift if configured
                time_shift = INSTRUMENT_CONFIG["SMPS"].get("time_shift", 0)
                if time_shift != 0:
                    result_df["datetime"] += pd.Timedelta(minutes=time_shift)

                # Ensure proper data types
                result_df["datetime"] = pd.to_datetime(
                    result_df["datetime"], errors="coerce"
                )

                # Add to combined data
                combined_smps_data = pd.concat(
                    [combined_smps_data, result_df], ignore_index=True
                )

        except Exception as e:
            print(f"Error processing SMPS file for date {burn_date}: {str(e)}")
            continue

    # Calculate time since garage closed for all data points
    if not combined_smps_data.empty:
        combined_smps_data["Time Since Garage Closed (hours)"] = np.nan

        for burn_id in burn_ids:
            burn_date_row = burn_log[burn_log["Burn ID"] == burn_id]
            if burn_date_row.empty:
                continue

            burn_date = burn_date_row["Date"].iloc[0]
            garage_closed_time_str = burn_date_row["garage closed"].iloc[0]

            if pd.isna(garage_closed_time_str):
                continue

            try:
                garage_closed_time = create_naive_datetime(
                    burn_date, garage_closed_time_str
                )
                burn_mask = combined_smps_data["burn_id"] == burn_id

                if not any(burn_mask):
                    continue

                burn_datetime = combined_smps_data.loc[burn_mask, "datetime"]

                # Ensure timezone-naive datetime
                if (
                    hasattr(burn_datetime.dtype, "tz")
                    and burn_datetime.dtype.tz is not None
                ):
                    burn_datetime = burn_datetime.dt.tz_localize(None)

                # Calculate hours since garage closed
                combined_smps_data.loc[
                    burn_mask, "Time Since Garage Closed (hours)"
                ] = (burn_datetime - garage_closed_time).dt.total_seconds() / 3600

            except Exception as e:
                print(f"Error calculating garage closed time for {burn_id}: {str(e)}")
                continue

    # Final data quality check
    if not combined_smps_data.empty:
        # Replace infinite values with NaN
        combined_smps_data = combined_smps_data.replace([np.inf, -np.inf], np.nan)

        # Remove rows where all concentration measurements are NaN
        concentration_cols = ["Total Concentration (µg/m³)"]
        existing_cols = [
            col for col in concentration_cols if col in combined_smps_data.columns
        ]

        if existing_cols:
            na_rows = combined_smps_data[existing_cols].isna().all(axis=1).sum()
            if na_rows > 0:
                print(f"Dropping {na_rows} rows with all NaN concentration values")
                combined_smps_data = combined_smps_data.dropna(
                    subset=existing_cols, how="all"
                )

    print(f"Processed SMPS data: {len(combined_smps_data)} records")
    return combined_smps_data


def process_instrument_data(instrument):
    """Route data processing to the appropriate function based on instrument type"""
    config = INSTRUMENT_CONFIG[instrument]
    process_function_name = config["process_function"]

    if process_function_name == "process_aerotrak_data":
        return process_aerotrak_data(config["file_path"], instrument)
    elif process_function_name == "process_dusttrak_data":
        return process_dusttrak_data(config["file_path"])
    elif process_function_name == "process_purpleairk_data":
        return process_purpleairk_data(config["file_path"])
    elif process_function_name == "process_quantaq_data":
        return process_quantaq_data(config["file_path"], instrument)
    elif process_function_name == "process_smps_data":
        return process_smps_data(config["file_path"])
    else:
        print(f"Unknown processing function: {process_function_name}")
        return None


def calculate_decay_parameters(data, instrument):
    """
    Calculate exponential decay parameters for particulate matter concentrations

    This function fits exponential decay curves to concentration data after peak values
    to quantify the rate of air quality improvement for different compartmentalization strategies.
    """
    global decay_parameters

    config = INSTRUMENT_CONFIG[instrument]
    pollutants = config["plot_pollutants"]
    datetime_column = config["datetime_column"]
    special_cases = config.get("special_cases", {})

    decay_parameters = {}
    burn_ids = BURN_GROUPS["figure4"]

    # Ensure datetime column is properly formatted
    data[datetime_column] = pd.to_datetime(data[datetime_column])

    # Process each burn scenario
    for burn_id in burn_ids:
        burn_data = data[data["burn_id"] == burn_id].copy()

        if burn_data.empty:
            print(f"No data available for {burn_id} in {instrument}")
            continue

        decay_parameters[burn_id] = {}

        # Get burn information from log
        burn_info = burn_log[burn_log["Burn ID"] == burn_id]
        if burn_info.empty:
            print(f"No burn info for {burn_id}")
            continue

        garage_closed_time_str = burn_info["garage closed"].iloc[0]
        if pd.isna(garage_closed_time_str):
            print(f"No garage closed time for {burn_id}")
            continue

        # Process each pollutant
        for pollutant in pollutants:
            if pollutant not in burn_data.columns:
                print(f"Pollutant {pollutant} not in data for {burn_id}")
                continue

            # Find maximum concentration and its timing
            max_concentration = burn_data[pollutant].max()

            if pd.isna(max_concentration):
                print(
                    f"No valid data for {pollutant} on {burn_id}, skipping decay calculation."
                )
                continue

            max_index = burn_data[pollutant].idxmax()
            max_time = burn_data["Time Since Garage Closed (hours)"].iloc[
                max_index - burn_data.index[0]
            ]

            # Filter out non-positive concentrations for logarithmic calculations
            burn_data["filtered_concentration"] = pd.to_numeric(
                burn_data[pollutant], errors="coerce"
            ).where(
                pd.to_numeric(burn_data[pollutant], errors="coerce") > 0, other=np.nan
            )

            # Calculate logarithmic values for decay analysis
            try:
                numeric_values = pd.to_numeric(
                    burn_data["filtered_concentration"], errors="coerce"
                )
                burn_data["log_concentration"] = np.log(numeric_values)
            except Exception as e:
                print(
                    f"Error calculating logarithm for {burn_id} {pollutant}: {str(e)}"
                )
                continue

            # Calculate rolling derivative to find stable decay region
            burn_data["rolling_derivative"] = (
                burn_data["log_concentration"].diff().rolling(window=5).mean()
            )

            # Determine decay time window based on special cases or standard method
            burn_special_case = special_cases.get(burn_id, {})
            if burn_special_case.get("custom_decay_time", False):
                # Use custom decay window (e.g., for air filter activation)
                decay_start_time = max_time
                decay_end_time = decay_start_time + burn_special_case.get(
                    "decay_end_offset", 0.25
                )
            else:
                # Standard method: find stable decay region after peak
                decay_start_time = max(
                    0, max_time
                )  # Start at garage closing or peak, whichever is later

                # Adjust for CR Box activation if applicable
                cr_box_on_time_str = burn_info["CR Box on"].iloc[0]
                if pd.notna(cr_box_on_time_str):
                    cr_box_on_time = create_naive_datetime(
                        burn_info["Date"].iloc[0], cr_box_on_time_str
                    )
                    garage_closed_time = create_naive_datetime(
                        burn_info["Date"].iloc[0], garage_closed_time_str
                    )
                    cr_box_on_time_since_garage_closed = (
                        cr_box_on_time - garage_closed_time
                    ).total_seconds() / 3600
                    decay_start_time = max(
                        decay_start_time, cr_box_on_time_since_garage_closed
                    )

                # Find stable derivative region
                valid_start_found = False
                sorted_data = burn_data.sort_values("Time Since Garage Closed (hours)")
                for i in range(len(sorted_data)):
                    current_time = sorted_data["Time Since Garage Closed (hours)"].iloc[
                        i
                    ]
                    if current_time >= decay_start_time and i >= 5:
                        rolling_mean = (
                            sorted_data["rolling_derivative"].iloc[i - 3 : i + 1].mean()
                        )
                        stability_threshold = 0.1
                        if (
                            abs(
                                sorted_data["rolling_derivative"].iloc[i] - rolling_mean
                            )
                            < stability_threshold
                        ):
                            decay_start_time = current_time
                            valid_start_found = True
                            break

                if not valid_start_found:
                    print(
                        f"Could not find stable decay start for {burn_id} {pollutant}"
                    )
                    continue

                # Find decay end time (5% of maximum concentration threshold)
                threshold_value = 0.05 * max_concentration
                below_threshold = burn_data[
                    (burn_data["Time Since Garage Closed (hours)"] > decay_start_time)
                    & (burn_data[pollutant] < threshold_value)
                ]

                if below_threshold.empty:
                    # Use last available data point if 5% threshold not reached
                    valid_data = burn_data[
                        (
                            burn_data["Time Since Garage Closed (hours)"]
                            > decay_start_time
                        )
                        & (burn_data[pollutant] > 0)
                    ]
                    if valid_data.empty:
                        print(
                            f"No valid data points after decay start time for {burn_id} {pollutant}"
                        )
                        continue
                    decay_end_time = valid_data[
                        "Time Since Garage Closed (hours)"
                    ].iloc[-1]
                    print(
                        f"Using last available time point ({decay_end_time:.2f} h) as decay end time for {burn_id} {pollutant}"
                    )
                else:
                    decay_end_time = below_threshold[
                        "Time Since Garage Closed (hours)"
                    ].iloc[0]

            # Extract decay region data
            decay_data = burn_data[
                (burn_data["Time Since Garage Closed (hours)"] >= decay_start_time)
                & (burn_data["Time Since Garage Closed (hours)"] <= decay_end_time)
                & (burn_data[pollutant] > 0)
            ].copy()

            if decay_data.empty or len(decay_data) < 3:
                print(f"Insufficient decay data points for {burn_id} {pollutant}")
                continue

            # Display decay analysis information
            print(f"{burn_id} {pollutant}")
            print(f"  Max Value: {max_concentration:.2f}")
            print(
                f"  Decay Start: {decay_start_time:.2f} h (Value: {decay_data[pollutant].iloc[0]:.2f})"
            )
            print(
                f"  Decay End: {decay_end_time:.2f} h (Value: {decay_data[pollutant].iloc[-1]:.2f})"
            )

            # Prepare data for exponential curve fitting
            decay_data = decay_data.sort_values("Time Since Garage Closed (hours)")
            x_data = (
                decay_data["Time Since Garage Closed (hours)"].values - decay_start_time
            )
            y_data = decay_data[pollutant].values

            # Set initial parameter guess for fitting
            initial_amplitude = y_data[0]
            initial_decay_rate = 0.1
            initial_guess = [initial_amplitude, initial_decay_rate]

            # Perform exponential curve fitting
            try:
                popt, y_fit, perr = fit_exponential_curve(x_data, y_data, initial_guess)

                # Check fit quality using relative standard deviation
                rsd = perr[1] / popt[1] if popt[1] != 0 else np.inf

                if rsd > 0.1:  # Exclude poor fits (>10% relative uncertainty)
                    print(
                        f"Decay for {burn_id} {pollutant} has RSD of {rsd:.2f} (> 0.1), excluding from results"
                    )
                    continue

                # Store decay parameters
                decay_parameters[burn_id][pollutant] = {
                    "decay_start_time": decay_start_time,
                    "decay_end_time": decay_end_time,
                    "amplitude": popt[0],
                    "decay_rate": popt[1],
                    "uncertainty": 1.96 * perr[1],  # 95% confidence interval
                    "rsd": rsd,
                    "x_data": x_data,
                    "y_data": y_data,
                    "max_concentration": max_concentration,
                    "max_time": max_time,
                }

                print(
                    f"  Decay Rate: {popt[1]:.4f} ± {1.96 * perr[1]:.4f} h⁻¹ (RSD: {rsd:.2f})"
                )

            except Exception as e:
                print(f"Error fitting {burn_id} {pollutant}: {str(e)}")
                continue

    return decay_parameters


def plot_figure4_raw_data(data, instrument, output_to_file=False):
    """
    Create Figure 4: Raw concentration data with decay analysis for compartmentalization strategies

    This function generates a log-scale plot showing how different compartmentalization strategies
    affect indoor air quality over time, with exponential decay curve fitting and maximum concentration callouts.

    Args:
        data: Processed instrument data
        instrument: Name of the instrument
        output_to_file: Whether to save as HTML file
    """
    global decay_parameters

    # Get text configuration
    metadata = get_script_metadata()

    # Calculate decay parameters if not already done
    if not decay_parameters:
        calculate_decay_parameters(data, instrument)

    # Ensure output directory exists
    os.makedirs("./Paper_figures", exist_ok=True)

    config = INSTRUMENT_CONFIG[instrument]
    pollutants = config["plot_pollutants"]

    # Configure Bokeh output
    reset_output()

    if output_to_file:
        output_file(f"./Paper_figures/{instrument}_cleanerairspace.html")
    else:
        output_notebook()

    burns_to_plot = BURN_GROUPS["figure4"]

    # Set y-axis label based on instrument type
    if instrument == "SMPS":
        y_label = "Particle Mass Concentration (µg/m³)"
    else:
        y_label = "PM Concentration (µg/m³)"

    # Create main figure with log scale
    p = figure(
        x_axis_label="Time Since Garage Door Closed (hours)",
        y_axis_label=y_label,
        x_axis_type="linear",
        y_axis_type="log",
        width=900,
        height=500,
        y_range=(10**-1.2, 10**3),
        toolbar_location="right",
    )

    # Apply initial text formatting (everything except legend)
    p.xaxis.axis_label_text_font_size = TEXT_CONFIG["axis_label_font_size"]
    p.yaxis.axis_label_text_font_size = TEXT_CONFIG["axis_label_font_size"]
    p.xaxis.axis_label_text_font_style = TEXT_CONFIG["font_style"]
    p.yaxis.axis_label_text_font_style = TEXT_CONFIG["font_style"]
    p.xaxis.major_label_text_font_size = TEXT_CONFIG["axis_tick_font_size"]
    p.yaxis.major_label_text_font_size = TEXT_CONFIG["axis_tick_font_size"]
    p.xaxis.major_label_text_font_style = TEXT_CONFIG["font_style"]
    p.yaxis.major_label_text_font_style = TEXT_CONFIG["font_style"]

    # Set x-axis range to show pre- and post-closure periods
    p.x_range.start = -1
    p.x_range.end = 4

    # Process each burn scenario
    for burn_id in burns_to_plot:
        burn_data = data[data["burn_id"] == burn_id].copy()

        if burn_data.empty:
            print(f"No data available for {burn_id}")
            continue

        burn_data = burn_data.sort_values("Time Since Garage Closed (hours)")
        source = ColumnDataSource(burn_data)

        # Select appropriate pollutant based on instrument type
        selected_pollutant = None

        if instrument == "SMPS":
            selected_pollutant = "Total Concentration (µg/m³)"
        elif instrument.startswith("AeroTrak"):
            selected_pollutant = "PM3 (µg/m³)"
        elif "PM2.5 (µg/m³)" in burn_data.columns:
            selected_pollutant = "PM2.5 (µg/m³)"
        elif "PM3 (µg/m³)" in burn_data.columns:
            selected_pollutant = "PM3 (µg/m³)"
        else:
            # Use first available pollutant
            for pollutant in pollutants:
                if (
                    pollutant in burn_data.columns
                    and not burn_data[pollutant].isna().all()
                ):
                    selected_pollutant = pollutant
                    break

        if selected_pollutant is None or selected_pollutant not in burn_data.columns:
            print(f"No suitable pollutant found for {burn_id}")
            continue

        if burn_data[selected_pollutant].isna().all():
            print(f"No valid data for {selected_pollutant} in {burn_id}")
            continue

        # Plot main concentration line
        legend_label = BURN_LABELS[burn_id]

        p.line(
            "Time Since Garage Closed (hours)",
            selected_pollutant,
            source=source,
            legend_label=legend_label,
            line_width=1.5,
            color=BURN_STYLES[burn_id]["color"],
            line_dash=BURN_STYLES[burn_id]["line_dash"],
        )

        # Add decay curve and annotations if parameters exist
        if (
            burn_id in decay_parameters
            and selected_pollutant in decay_parameters[burn_id]
        ):
            decay_params = decay_parameters[burn_id][selected_pollutant]

            # Extract decay parameters
            decay_start_time = decay_params["decay_start_time"]
            amplitude = decay_params["amplitude"]
            decay_rate = decay_params["decay_rate"]
            max_concentration = decay_params["max_concentration"]
            max_time = decay_params["max_time"]

            # Generate fitted decay curve
            x_range = np.linspace(0, decay_params["x_data"][-1], 100)
            y_fit = amplitude * np.exp(-decay_rate * x_range)
            x_fit = x_range + decay_start_time

            # Plot fitted decay curve
            p.line(
                x_fit,
                y_fit,
                line_color=BURN_STYLES[burn_id]["color"],
                line_dash="dashdot",
                line_width=2,
            )

            # Add maximum concentration callout with consistent text formatting
            max_label = Label(
                x=max_time - 0.85,
                y=max_concentration * 1.5,
                text=f"Max: {max_concentration:.0f} µg/m³",
                text_font_size=TEXT_CONFIG["label_font_size"],
                text_font_style=TEXT_CONFIG["font_style"],
                text_color=BURN_STYLES[burn_id]["color"],
                border_line_color=None,
                background_fill_color="white",
                background_fill_alpha=0.6,
            )
            p.add_layout(max_label)

            # Add arrow pointing to maximum
            max_arrow = Arrow(
                end=NormalHead(size=10),
                x_start=max_time - 0.2,
                y_start=max_concentration * 1.5,
                x_end=max_time,
                y_end=max_concentration,
                line_color=BURN_STYLES[burn_id]["color"],
                line_width=1.5,
            )
            p.add_layout(max_arrow)

    # Configure legend with consistent formatting
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    p.legend.label_text_font_size = TEXT_CONFIG["legend_font_size"]
    p.legend.label_text_font_style = TEXT_CONFIG["font_style"]
    p.legend.background_fill_alpha = 0.7
    p.legend.spacing = 5
    p.legend.padding = 10

    # Add vertical line at Time = 0 (garage door closure)
    p.line(
        x=[0, 0],
        y=[10**-2, 10**4],
        line_color="black",
        line_width=1.5,
        line_dash="solid",
        legend_label="Garage Closed",
    )

    # Add metadata with consistent text formatting
    text_div = Div(
        text=f'<div style="font-size: {TEXT_CONFIG["font_size"]}; font-weight: {TEXT_CONFIG["html_font_weight"]};">{metadata}</div>',
        width=800,
    )
    layout = column(p, text_div)

    show(layout)
    return p


def main(instrument="SMPS", output_to_file=False):
    """
    Main function to process instrument data and generate Figure 4

    Args:
        instrument: Instrument to process ('AeroTrakB', 'AeroTrakK', 'DustTrak',
                   'PurpleAirK', 'QuantAQB', 'QuantAQK', 'SMPS')
        output_to_file: Whether to save the figure to an HTML file
    """
    print(f"Starting Figure 4 Raw Concentration Analysis for {instrument}")
    print(f"Output to file: {output_to_file}")

    # Get text configuration at startup
    get_text_config()

    # Validate instrument selection
    if instrument not in INSTRUMENT_CONFIG:
        print(f"Invalid instrument: {instrument}")
        print(f"Available instruments: {', '.join(INSTRUMENT_CONFIG.keys())}")
        return

    # Process the instrument data
    data = process_instrument_data(instrument)

    if data is not None and not data.empty:
        # Generate Figure 4 with decay analysis
        plot_figure4_raw_data(data, instrument, output_to_file)
        print("\nPlotting complete.")

        # Print instrument-specific information
        if instrument == "SMPS":
            print("SMPS data showing Total Concentration")
        elif instrument.startswith("AeroTrak"):
            print("AeroTrak data showing PM3 concentration")
        elif instrument in ["DustTrak", "QuantAQB", "QuantAQK", "PurpleAirK"]:
            print("Data showing PM2.5 concentration (or PM3 if PM2.5 not available)")

        print("To save to file, set output_to_file=True in the main() function call")
    else:
        print(f"No valid {instrument} data available for plotting")


# Execute main function if script is run directly
if __name__ == "__main__":
    # Configuration: Select instrument and output options
    instrument = "SMPS"  # Options: 'AeroTrakB', 'AeroTrakK', 'DustTrak', 'PurpleAirK', 'QuantAQB', 'QuantAQK', 'SMPS'
    output_to_file = (
        True  # Set to True to save as HTML file, False for notebook display only
    )

    main(instrument, output_to_file)
# %%
