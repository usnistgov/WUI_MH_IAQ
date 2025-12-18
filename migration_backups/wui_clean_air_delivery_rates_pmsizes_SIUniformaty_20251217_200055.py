"""
WUI CADR Analysis: PM Size-Dependent Comparison for Burn 4

This script analyzes and compares Clean Air Delivery Rates (CADR) for different
particle sizes (PM1, PM3, PM10) between two AeroTrak instruments located in the
bedroom and kitchen during Burn 4 experiments. The analysis focuses on evaluating
spatial uniformity of CADR measurements across different PM size fractions.

Key Features:
    - Processes AeroTrak particle count data and converts to mass concentrations
    - Calculates size-dependent PM concentrations (PM1, PM3, PM10)
    - Applies instrument-specific time shifts for synchronization
    - Fits exponential decay curves to determine decay rates
    - Generates comparison plots with logarithmic scale
    - Exports figures for Supporting Information (SI) documentation

Methodology:
    - Exponential decay fitting: C(t) = A * exp(-λt)
    - Decay rates (λ) calculated with 95% confidence intervals
    - Quality control: RSD threshold of 0.1 for accepting fits
    - Baseline correction applied using pre-experiment values

Inputs:
    - AeroTrak data from bedroom2 and kitchen locations
    - Burn log with experimental timing data
    - Instrument configurations with time shifts and baseline values

Outputs:
    - Interactive Bokeh plots showing PM concentration time series
    - HTML figure saved to ./Paper_figures/SI_Burn4_AeroTrak_comparison.html
    - Decay rate parameters with uncertainties

Dependencies:
    - pandas: Data manipulation
    - numpy: Numerical operations
    - bokeh: Interactive visualization
    - scipy: Curve fitting

Author: Nathan Lima
Date: 2024-2025
"""

# %%
import os
import sys
import pandas as pd
import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_notebook, output_file
from bokeh.models import ColumnDataSource, Div
from bokeh.layouts import column

# Set output to display plots in the notebook
output_notebook()

# Set the absolute path for the dataset
absolute_path = "C:/Users/nml/OneDrive - NIST/Documents/NIST/WUI_smoke/"
os.chdir(absolute_path)

# Import utils
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from scripts import get_script_metadata  # pylint: disable=import-error,wrong-import-position

# SAVE FLAG - Set to True when ready to save figures
SAVE_FIGURES = True

# Load burn log
burn_log_path = "./burn_log.xlsx"
burn_log = pd.read_excel(burn_log_path, sheet_name="Sheet2")

# Global variables
decay_parameters = {}

# Define color mapping for pollutants
POLLUTANT_COLORS = {
    "PM1 (µg/m³)": "#d45087",
    "PM3 (µg/m³)": "#b35093",
    "PM10 (µg/m³)": "#404e84",
}

# Define instrument configurations (only AeroTrak instruments)
INSTRUMENT_CONFIG = {
    "AeroTrakB": {
        "file_path": "./burn_data/aerotraks/bedroom2/all_data.xlsx",
        "time_shift": 2.16,
        "plot_pollutants": ["PM1 (µg/m³)", "PM3 (µg/m³)", "PM10 (µg/m³)"],
        "normalize_pollutant": "PM3 (µg/m³)",
        "datetime_column": "Date and Time",
        "baseline_values": {
            "PM1 (µg/m³)": (0.5492, 0.0116),
            "PM3 (µg/m³)": (1.0855, 0.0511),
            "PM10 (µg/m³)": (2.7994, 0.1160),
        },
    },
    "AeroTrakK": {
        "file_path": "./burn_data/aerotraks/kitchen/all_data.xlsx",
        "time_shift": 5,
        "plot_pollutants": ["PM1 (µg/m³)", "PM3 (µg/m³)", "PM10 (µg/m³)"],
        "normalize_pollutant": "PM3 (µg/m³)",
        "datetime_column": "Date and Time",
        "baseline_values": {
            "PM1 (µg/m³)": (0.5492, 0.0116),
            "PM3 (µg/m³)": (1.0855, 0.0511),
            "PM10 (µg/m³)": (2.7994, 0.1160),
        },
    },
}


def apply_time_shift(df, instrument, burn_id, burn_date):
    """Apply time shift based on instrument configuration"""
    time_shift = INSTRUMENT_CONFIG[instrument].get("time_shift", 0)
    datetime_column = INSTRUMENT_CONFIG[instrument].get(
        "datetime_column", "Date and Time"
    )

    df.loc[:, datetime_column] = pd.to_datetime(df[datetime_column])
    burn_date = pd.to_datetime(burn_date).date()

    if time_shift != 0:
        mask = df[datetime_column].dt.date == burn_date
        if mask.any():
            df.loc[mask, datetime_column] += pd.Timedelta(minutes=time_shift)

    return df


def create_naive_datetime(date_str, time_str):
    """Create a timezone-naive datetime object from date and time strings"""
    dt = pd.to_datetime(f"{date_str} {time_str}", errors="coerce")
    if hasattr(dt, "tz") and dt.tz is not None:
        dt = dt.tz_localize(None)
    return dt


def process_aerotrak_data(file_path, instrument):
    """Process AeroTrak data with instrument-specific settings"""
    # Load the AeroTrak data from the Excel file
    aerotrak_data = pd.read_excel(file_path)
    aerotrak_data.columns = aerotrak_data.columns.str.strip()

    # Define size channels and initialize a dictionary for size values
    size_channels = ["Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6"]
    size_values = {}

    # Extract size values for each channel
    for channel in size_channels:
        size_col = f"{channel} Size (µm)"
        if size_col in aerotrak_data.columns:
            size_value = aerotrak_data[size_col].iloc[0]
            if pd.notna(size_value):
                size_values[channel] = size_value

    # Check for the volume column and convert it to cm³
    volume_column = "Volume (L)"
    if volume_column in aerotrak_data.columns:
        aerotrak_data["Volume (cm³)"] = aerotrak_data[volume_column] * 1000
        volume_cm = aerotrak_data["Volume (cm³)"]

    def g_mean(x):
        a = np.log(x)
        return np.exp(a.mean())

    # Initialize new columns for mass concentration and calculate values
    pm_columns = []
    for i, channel in enumerate(size_channels):
        if channel in size_values:
            next_channel = size_channels[i + 1] if i < len(size_channels) - 1 else None
            next_size_value = size_values[next_channel] if next_channel else 25

            particle_size = g_mean([size_values[channel], next_size_value])
            particle_size_m = particle_size * 1e-6

            diff_col = f"{channel} Diff (#)"
            if diff_col in aerotrak_data.columns:
                particle_counts = aerotrak_data[diff_col]

                radius_m = particle_size_m / 2
                volume_per_particle = (4 / 3) * np.pi * (radius_m**3)
                particle_mass = volume_per_particle * 1e6 * 1e6

                new_diff_col_µg_m3 = (
                    f"PM{size_values[channel]}-{next_size_value} Diff (µg/m³)"
                )
                aerotrak_data[new_diff_col_µg_m3] = (
                    particle_counts / (volume_cm * 1e-6)
                ) * (particle_mass)
                pm_columns.append(new_diff_col_µg_m3)

                pm_column_name = f"PM{next_size_value} (µg/m³)"
                aerotrak_data[pm_column_name] = aerotrak_data[new_diff_col_µg_m3]

    # Define cumulative PM concentration columns
    cumulative_columns = [
        "PM0.5 (µg/m³)",
        "PM1 (µg/m³)",
        "PM3 (µg/m³)",
        "PM5 (µg/m³)",
        "PM10 (µg/m³)",
        "PM25 (µg/m³)",
    ]

    # Calculate cumulative PM concentrations
    for i in range(len(cumulative_columns)):
        if i == 0:
            aerotrak_data[cumulative_columns[i]] = aerotrak_data[pm_columns[i]]
        else:
            aerotrak_data[cumulative_columns[i]] = aerotrak_data[pm_columns[i]].add(
                aerotrak_data[cumulative_columns[i - 1]], fill_value=0
            )

    # Replace invalid entries with NaN for numeric columns only
    status_columns = ["Flow Status", "Laser Status"]
    valid_status = (aerotrak_data[status_columns] == "OK").all(axis=1)
    for col in aerotrak_data.columns:
        if pd.api.types.is_numeric_dtype(aerotrak_data[col]) and col not in [
            "Date and Time",
            "Sample Time",
            "Volume (L)",
        ]:
            aerotrak_data.loc[~valid_status, col] = pd.NA

    # Filter for burn4 only
    burn4_date = burn_log[burn_log["Burn ID"] == "burn4"]["Date"].iloc[0]
    burn4_date = pd.to_datetime(burn4_date)

    # Convert 'Date and Time' to date and filter for burn4
    aerotrak_data["Date"] = pd.to_datetime(aerotrak_data["Date and Time"]).dt.date
    filtered_aerotrak_data = aerotrak_data[
        aerotrak_data["Date"] == burn4_date.date()
    ].copy()

    # Apply time shift for burn4
    filtered_aerotrak_data = apply_time_shift(
        filtered_aerotrak_data, instrument, "burn4", burn4_date
    )

    # Calculate time since garage closed
    filtered_aerotrak_data["Date and Time"] = pd.to_datetime(
        filtered_aerotrak_data["Date and Time"]
    )
    filtered_aerotrak_data["Time Since Garage Closed (hours)"] = np.nan

    garage_closed_time_str = burn_log[burn_log["Burn ID"] == "burn4"][
        "garage closed"
    ].iloc[0]
    garage_closed_time = create_naive_datetime(
        burn4_date.date(), garage_closed_time_str
    )

    if pd.notna(garage_closed_time):
        datetime_values = filtered_aerotrak_data["Date and Time"]
        if (
            hasattr(datetime_values.dtype, "tz")
            and datetime_values.dtype.tz is not None
        ):
            datetime_values = datetime_values.dt.tz_localize(None)

        time_since_closed = (
            datetime_values - garage_closed_time
        ).dt.total_seconds() / 3600
        filtered_aerotrak_data["Time Since Garage Closed (hours)"] = time_since_closed

    return filtered_aerotrak_data


def fit_exponential_curve(x_data, y_data, initial_guess):
    from scipy.optimize import curve_fit

    if len(x_data) == 0 or len(y_data) == 0:
        raise ValueError("Input data for fitting is empty.")

    def exp_decreasing(t, a, b):
        return a * np.exp(-b * t)

    try:
        popt, pcov = curve_fit(
            exp_decreasing, x_data, y_data, p0=initial_guess, maxfev=10000
        )
    except Exception as e:
        print(f"Curve fitting error: {e}")
        raise

    y_fit = exp_decreasing(x_data, *popt)
    perr = np.sqrt(np.diag(pcov))

    return popt, y_fit, perr


def calculate_decay_parameters_burn4(data, instrument, burn_id="burn4"):
    """Calculate decay parameters for burn4 data"""
    global decay_parameters

    config = INSTRUMENT_CONFIG[instrument]
    pollutants = config["plot_pollutants"]
    datetime_column = config["datetime_column"]
    # baseline_values = config.get('baseline_values', {})

    # Initialize entry for this burn
    if burn_id not in decay_parameters:
        decay_parameters[burn_id] = {}
    if instrument not in decay_parameters[burn_id]:
        decay_parameters[burn_id][instrument] = {}

    # Get burn data
    burn_date = burn_log[burn_log["Burn ID"] == burn_id]["Date"].iloc[0]
    garage_closed_time_str = burn_log[burn_log["Burn ID"] == burn_id][
        "garage closed"
    ].iloc[0]
    garage_closed_time = create_naive_datetime(burn_date, garage_closed_time_str)

    if pd.isna(garage_closed_time):
        return decay_parameters

    # Calculate time since garage closed if not already done
    if "Time Since Garage Closed (hours)" not in data.columns:
        data_datetime = data[datetime_column]
        if hasattr(data_datetime.dtype, "tz") and data_datetime.dtype.tz is not None:
            data_datetime = data_datetime.dt.tz_localize(None)

        data["Time Since Garage Closed (hours)"] = (
            data_datetime - garage_closed_time
        ).dt.total_seconds() / 3600

    for pollutant in pollutants:
        if pollutant not in data.columns:
            continue

        # Filter out non-positive and NaN values first
        valid_data = (
            data[
                (data[pollutant].notna())
                & (data[pollutant] > 0)
                & (data["Time Since Garage Closed (hours)"].notna())
            ]
            .copy()
            .reset_index(drop=True)
        )

        if valid_data.empty:
            print(f"No valid data for {pollutant} in {burn_id}")
            continue

        # Find maximum concentration from valid data
        max_concentration = valid_data[pollutant].max()
        if pd.isna(max_concentration):
            continue

        # Get the index in the valid_data dataframe
        max_index_valid = valid_data[pollutant].idxmax()
        max_time = valid_data["Time Since Garage Closed (hours)"].loc[max_index_valid]

        # Filter out non-positive concentrations for log calculation
        valid_data["filtered_concentration"] = pd.to_numeric(
            valid_data[pollutant], errors="coerce"
        ).where(pd.to_numeric(valid_data[pollutant], errors="coerce") > 0, other=np.nan)

        try:
            numeric_values = pd.to_numeric(
                valid_data["filtered_concentration"], errors="coerce"
            )
            valid_data["log_concentration"] = np.log(numeric_values)
        except Exception as e:
            print(f"Error calculating logarithm for {burn_id} {pollutant}: {str(e)}")
            continue

        # Calculate rolling derivative
        valid_data["rolling_derivative"] = (
            valid_data["log_concentration"].diff().rolling(window=5).mean()
        )

        # Initialize decay start time
        decay_start_time = max(0, max_time)

        # Get CR Box on time
        cr_box_on_time_str = burn_log[burn_log["Burn ID"] == burn_id]["CR Box on"].iloc[
            0
        ]
        cr_box_on_time = create_naive_datetime(burn_date, cr_box_on_time_str)

        if pd.notna(cr_box_on_time):
            cr_box_on_time_since_garage_closed = (
                cr_box_on_time - garage_closed_time
            ).total_seconds() / 3600
            decay_start_time = max(decay_start_time, cr_box_on_time_since_garage_closed)

        # Search for stable derivative after max_index_valid
        valid_start_found = False
        for idx in range(max_index_valid + 1, len(valid_data)):
            if idx >= max_index_valid + 5:  # Ensure enough points for rolling mean
                rolling_mean = (
                    valid_data["rolling_derivative"].iloc[idx - 3 : idx + 1].mean()
                )
                stability_threshold = 0.1
                if (
                    abs(valid_data["rolling_derivative"].iloc[idx] - rolling_mean)
                    < stability_threshold
                ):
                    potential_start_time = valid_data[
                        "Time Since Garage Closed (hours)"
                    ].iloc[idx]
                    if potential_start_time >= decay_start_time:
                        decay_start_time = potential_start_time
                        valid_start_found = True
                        break

        if not valid_start_found:
            print(f"Could not find stable decay start for {burn_id} {pollutant}")
            continue

        # Find decay end time (5% threshold)
        threshold_value = 0.05 * max_concentration
        below_threshold = valid_data[
            (valid_data["Time Since Garage Closed (hours)"] > decay_start_time)
            & (valid_data[pollutant] < threshold_value)
        ]

        if below_threshold.empty:
            valid_end_data = valid_data[
                (valid_data["Time Since Garage Closed (hours)"] > decay_start_time)
                & (valid_data[pollutant] > 0)
            ]
            if valid_end_data.empty:
                continue
            decay_end_time = valid_end_data["Time Since Garage Closed (hours)"].iloc[-1]
        else:
            decay_end_time = below_threshold["Time Since Garage Closed (hours)"].iloc[0]

        # Extract decay data
        decay_data = valid_data[
            (valid_data["Time Since Garage Closed (hours)"] >= decay_start_time)
            & (valid_data["Time Since Garage Closed (hours)"] <= decay_end_time)
            & (valid_data[pollutant] > 0)
        ].copy()

        if decay_data.empty or len(decay_data) < 3:
            continue

        # Prepare data for exponential fitting
        x_data = (
            decay_data["Time Since Garage Closed (hours)"].values - decay_start_time
        )
        y_data = decay_data[pollutant].values

        initial_amplitude = y_data[0]
        initial_decay_rate = 0.1
        initial_guess = [initial_amplitude, initial_decay_rate]

        # Fit exponential curve
        try:
            popt, y_fit, perr = fit_exponential_curve(x_data, y_data, initial_guess)

            rsd = perr[1] / popt[1]
            if rsd > 0.1:
                print(
                    f"Decay for {burn_id} {pollutant} has RSD of {rsd:.2f} (> 0.1), excluding from results"
                )
                continue

            # Store parameters
            decay_parameters[burn_id][instrument][pollutant] = {
                "decay_start_time": decay_start_time,
                "decay_end_time": decay_end_time,
                "amplitude": popt[0],
                "decay_rate": popt[1],
                "uncertainty": 1.96 * perr[1],
                "rsd": rsd,
                "x_data": x_data,
                "y_data": y_data,
            }

        except Exception as e:
            print(f"Error fitting {burn_id} {pollutant}: {str(e)}")
            continue

    return decay_parameters


def create_burn4_comparison_plot():
    """Create comparison plot for burn4 showing both AeroTrakB and AeroTrakK"""
    # Process data for both instruments
    aerotrakb_data = process_aerotrak_data(
        INSTRUMENT_CONFIG["AeroTrakB"]["file_path"], "AeroTrakB"
    )
    aerotrakk_data = process_aerotrak_data(
        INSTRUMENT_CONFIG["AeroTrakK"]["file_path"], "AeroTrakK"
    )

    # Calculate decay parameters for both instruments
    calculate_decay_parameters_burn4(aerotrakb_data, "AeroTrakB")
    calculate_decay_parameters_burn4(aerotrakk_data, "AeroTrakK")

    # Get script metadata
    metadata = get_script_metadata()

    # Create plot
    p = figure(
        x_axis_label="Time Since Garage Closed (hours)",
        y_axis_label="PM Concentration (µg/m³)",
        x_axis_type="linear",
        y_axis_type="log",
        width=800,
        height=500,
        y_range=(10**-2, 10**4),
        title="Burn 4 AeroTrak Comparison PM-dependent-size mass-concentration",
    )

    # Set x-axis range
    p.x_range.start = -1
    p.x_range.end = 4

    pollutants = ["PM1 (µg/m³)", "PM3 (µg/m³)", "PM10 (µg/m³)"]

    # Plot AeroTrakB data (dashed lines)
    source_b = ColumnDataSource(aerotrakb_data)
    for pollutant in pollutants:
        if pollutant in aerotrakb_data.columns:
            # Change label format
            if pollutant == "PM1 (µg/m³)":
                legend_label = "PM1.0-B"
            elif pollutant == "PM3 (µg/m³)":
                legend_label = "PM3.0-B"
            elif pollutant == "PM10 (µg/m³)":
                legend_label = "PM10-B"

            color = POLLUTANT_COLORS.get(pollutant, "gray")
            p.line(
                "Time Since Garage Closed (hours)",
                pollutant,
                source=source_b,
                legend_label=legend_label,
                line_width=1.5,
                color=color,
                line_dash="dashed",
            )

    # Plot AeroTrakK data (solid lines)
    source_k = ColumnDataSource(aerotrakk_data)
    for pollutant in pollutants:
        if pollutant in aerotrakk_data.columns:
            # Change label format
            if pollutant == "PM1 (µg/m³)":
                legend_label = "PM1.0-K"
            elif pollutant == "PM3 (µg/m³)":
                legend_label = "PM3.0-K"
            elif pollutant == "PM10 (µg/m³)":
                legend_label = "PM10-K"

            color = POLLUTANT_COLORS.get(pollutant, "gray")
            p.line(
                "Time Since Garage Closed (hours)",
                pollutant,
                source=source_k,
                legend_label=legend_label,
                line_width=1.5,
                color=color,
                line_dash="solid",
            )

    # Add vertical line for garage closed
    p.line(
        x=[0] * 2,
        y=[10**-2, 10**4],
        line_color="black",
        line_width=1,
        line_dash="solid",
        legend_label="Garage Closed",
    )

    # Add vertical line for CR Box On
    burn4_date = burn_log[burn_log["Burn ID"] == "burn4"]["Date"].iloc[0]
    garage_closed_time_str = burn_log[burn_log["Burn ID"] == "burn4"][
        "garage closed"
    ].iloc[0]
    cr_box_on_time_str = burn_log[burn_log["Burn ID"] == "burn4"]["CR Box on"].iloc[0]

    garage_closed_time = create_naive_datetime(burn4_date, garage_closed_time_str)
    cr_box_on_time = create_naive_datetime(burn4_date, cr_box_on_time_str)

    if pd.notna(cr_box_on_time):
        cr_box_on_time_since_garage_closed = (
            cr_box_on_time - garage_closed_time
        ).total_seconds() / 3600
        p.line(
            x=[cr_box_on_time_since_garage_closed] * 2,
            y=[10**-2, 10**4],
            line_color="black",
            line_width=1,
            line_dash="dashed",
            legend_label="CR Box on",
        )

    # Plot decay curves for both instruments
    decay_rates_info = []

    for instrument in ["AeroTrakB", "AeroTrakK"]:
        if "burn4" in decay_parameters and instrument in decay_parameters["burn4"]:
            for pollutant in pollutants:
                if pollutant in decay_parameters["burn4"][instrument]:
                    color = POLLUTANT_COLORS.get(pollutant, "gray")
                    decay_params = decay_parameters["burn4"][instrument][pollutant]

                    try:
                        decay_start_time = decay_params["decay_start_time"]
                        amplitude = decay_params["amplitude"]
                        decay_rate = decay_params["decay_rate"]
                        uncertainty = decay_params["uncertainty"]
                        x_data = decay_params["x_data"]

                        # Create points for the fitted curve
                        x_fit = np.linspace(0, x_data[-1], 100)
                        y_fit = amplitude * np.exp(-decay_rate * x_fit)

                        # Plot the fitted curve (dashdot for both instruments)
                        # p.line(x_fit + decay_start_time, y_fit,
                        #       line_color=color, line_dash='dashdot', line_cap='round', line_width=2)

                        # Add decay rate info
                        if pollutant == "PM1 (µg/m³)":
                            pollutant_name = "PM1.0"
                        elif pollutant == "PM3 (µg/m³)":
                            pollutant_name = "PM3.0"
                        elif pollutant == "PM10 (µg/m³)":
                            pollutant_name = "PM10"

                        instrument_suffix = "B" if instrument == "AeroTrakB" else "K"
                        decay_rate_text = f"{pollutant_name}-{instrument_suffix} Decay Rate: {decay_rate:.1f} ± {uncertainty:.2f}  h⁻¹"
                        decay_rates_info.append(decay_rate_text)

                    except Exception as e:
                        print(
                            f"Error plotting decay curve for {pollutant} in {instrument}: {str(e)}"
                        )
                        continue

    # Customize legend
    p.legend.click_policy = "hide"
    p.legend.location = "top_right"

    # Create layout with decay information
    if decay_rates_info:
        text_div = Div(
            text="<br>".join(decay_rates_info) + f"<br><small>{metadata}</small>"
        )
        layout = column(p, text_div)
    else:
        layout = p

    # Show plot in terminal
    show(layout)

    # Save plot only if flag is set
    if SAVE_FIGURES:
        os.makedirs("./Paper_figures", exist_ok=True)
        html_filename = "./Paper_figures/SI_Burn4_AeroTrak_comparison.html"
        output_file(html_filename)
        show(layout)
        print(f"Figure saved to {html_filename}")
    else:
        print("Figure displayed in terminal only (SAVE_FIGURES = False)")


# Main execution
if __name__ == "__main__":
    create_burn4_comparison_plot()

# %%
