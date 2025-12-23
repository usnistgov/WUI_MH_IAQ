"""
WUI General Particle Count Comparison: Multi-Instrument Analysis

This script creates comprehensive comparison plots of particle number concentrations
from multiple instruments across all wildland-urban interface smoke experiments.

It generates separate visualizations for bedroom and kitchen locations to evaluate
spatial variation and instrument agreement.

Key Features:
    - Multi-instrument particle count overlay plots
    - Separate panels for bedroom and kitchen locations
    - All burns (1-10) processed automatically
    - Logarithmic y-axis for wide concentration range
    - Interactive Bokeh HTML outputs with synchronized time axes

Instruments Compared:
    Bedroom Instruments:
        - AeroTrakB: Size-resolved particle counts (0.3-25 µm, 6 channels)
        - QuantAQB: Optical particle counter (0.35-40 µm, 24 bins)
        - SMPS: Scanning Mobility Particle Sizer (9-437 nm)

    Kitchen Instruments:
        - AeroTrakK: Size-resolved particle counts (0.3-25 µm, 6 channels)
        - QuantAQK: Optical particle counter (0.35-40 µm, 24 bins)

Data Processing Pipeline:
    Uses centralized utility modules from scripts/ directory:
    - data_loaders: Instrument-specific data loading and processing
    - datetime_utils: Time synchronization and shift corrections
    - data_filters: Quality filtering and rolling averages
    - plotting_utils: Standardized figure creation and formatting
    - instrument_config: Bin definitions and instrument configurations

Visualization Features:
    - Logarithmic y-axis (1e-4 to 1e5 #/cm³)
    - Time axis: -1 to 4 hours relative to garage closed
    - Color-coded by instrument (Blues: AeroTrak, Greens: QuantAQ, Reds: SMPS)
    - Event markers for garage closed and CR Box activation
    - Interactive tooltips, zoom, and pan capabilities

Research Applications:
    - Cross-validate instrument consistency
    - Identify measurement artifacts or drift
    - Assess spatial uniformity between bedroom and kitchen
    - Evaluate sensor performance across concentration ranges
    - Compare ultrafine (SMPS) vs. larger particle (AeroTrak, QuantAQ) dynamics

Outputs:
    For each burn:
        - {burn}_bedroom_particle_count_comparison.html
        - {burn}_kitchen_particle_count_comparison.html

Configuration:
    - selected_instruments: List of instruments to include ('AeroTrak', 'QuantAQ', 'SMPS')
    - burn_numbers: Specific burns to process (default: all burns 1-10)
    - DEBUG: Enable detailed diagnostic output

Dependencies:
    - pandas: Data manipulation and time series handling
    - numpy: Numerical operations
    - bokeh: Interactive HTML visualization
    - Custom modules: src.data_paths, scripts.* utilities

Notes:
    - Instrument time shifts automatically applied for synchronization
    - QuantAQ data only available for burns 4-10
    - Burn 3 data includes 5-minute rolling average smoothing
    - Status flags (Flow Status, Laser Status) used for quality filtering

Author: Nathan Lima
Date: 2024-2025
"""

# %%
import sys
import traceback
from pathlib import Path
import pandas as pd
import numpy as np
from bokeh.plotting import show
from bokeh.io import output_notebook, output_file
from bokeh.layouts import column
from bokeh.palettes import Blues, Greens, Reds

# Add repository root to path for portable data access
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

# pylint: disable=import-error,wrong-import-position
from src.data_paths import get_data_root, get_instrument_path, get_common_file

# Import utility functions
from scripts.data_filters import split_data_by_nan
from scripts.plotting_utils import (
    create_standard_figure,
    get_script_metadata,
    add_event_markers,
    configure_legend,
    create_metadata_div,
)
from scripts.data_loaders import (
    load_burn_log,
    get_garage_closed_times,
    get_cr_box_times,
    process_aerotrak_data,
    process_quantaq_data,
    process_smps_data,
)
from scripts.instrument_config import get_burn_range_for_instrument
# pylint: enable=import-error,wrong-import-position

# %% Configuration parameters

# Set paths
data_root = get_data_root()  # Portable path - auto-configured
output_dir = get_common_file('output_figures')

# Burns to process (all burns by default)
burn_numbers = [f"burn{i}" for i in range(1, 11)]  # burn1 through burn10

# Select which instruments to include in the plot
# Options: 'AeroTrak', 'QuantAQ', 'SMPS'
selected_instruments = ["AeroTrak", "QuantAQ", "SMPS"]

# Enable debug mode for additional diagnostic output
DEBUG = False

# %% Load burn log
print("Loading burn log...")
burn_log_path = get_common_file('burn_log')
burn_log = load_burn_log(burn_log_path)

# Get garage closed times and CR Box activation times
garage_closed_times = get_garage_closed_times(burn_log, burn_numbers)
cr_box_hours = get_cr_box_times(burn_log, burn_numbers, relative_to_garage=True)

# %% Process AeroTrak data
aerotrak_data = {"Bedroom": {}, "Kitchen": {}}

if "AeroTrak" in selected_instruments:
    print("\nProcessing AeroTrak data...")

    aerotrak_b_path = get_instrument_path('aerotrak_bedroom') / 'all_data.xlsx'
    aerotrak_k_path = get_instrument_path('aerotrak_kitchen') / 'all_data.xlsx'

    # Process Bedroom AeroTrak
    print("Loading AeroTrakB data...")
    for burn_number in burn_numbers:
        burn_date_row = burn_log[burn_log["Burn ID"] == burn_number]
        if not burn_date_row.empty:
            burn_date = pd.to_datetime(burn_date_row["Date"].iloc[0])
            garage_time = garage_closed_times.get(burn_number)

            filtered_data = process_aerotrak_data(
                aerotrak_b_path,
                instrument="AeroTrakB",
                burn_date=burn_date,
                garage_closed_time=garage_time,
                burn_number=burn_number,
            )

            if not filtered_data.empty:
                aerotrak_data["Bedroom"][burn_number] = filtered_data
                print(f"  Processed AeroTrakB: {burn_number} ({len(filtered_data)} records)")

    # Process Kitchen AeroTrak
    print("Loading AeroTrakK data...")
    for burn_number in burn_numbers:
        burn_date_row = burn_log[burn_log["Burn ID"] == burn_number]
        if not burn_date_row.empty:
            burn_date = pd.to_datetime(burn_date_row["Date"].iloc[0])
            garage_time = garage_closed_times.get(burn_number)

            filtered_data = process_aerotrak_data(
                aerotrak_k_path,
                instrument="AeroTrakK",
                burn_date=burn_date,
                garage_closed_time=garage_time,
                burn_number=burn_number,
            )

            if not filtered_data.empty:
                aerotrak_data["Kitchen"][burn_number] = filtered_data
                print(f"  Processed AeroTrakK: {burn_number} ({len(filtered_data)} records)")

# %% Process QuantAQ data
quantaq_data = {"Bedroom": {}, "Kitchen": {}}

if "QuantAQ" in selected_instruments:
    print("\nProcessing QuantAQ data...")

    # Get valid burns for QuantAQ (burns 4-10)
    quantaq_burns = [b for b in burn_numbers if b in get_burn_range_for_instrument('QuantAQB')]

    if quantaq_burns:
        quantaq_b_path = get_instrument_path('quantaq_bedroom') / 'MOD-PM-00194-b0fc215029fa4852b926bc50b28fda5a.csv'
        quantaq_k_path = get_instrument_path('quantaq_kitchen') / 'MOD-PM-00197-a6dd467a147a4d95a7b98a8a10ab4ea3.csv'

        # Process Bedroom QuantAQ
        print("Loading QuantAQB data...")
        for burn_number in quantaq_burns:
            burn_date_row = burn_log[burn_log["Burn ID"] == burn_number]
            if not burn_date_row.empty:
                burn_date = pd.to_datetime(burn_date_row["Date"].iloc[0])
                garage_time = garage_closed_times.get(burn_number)

                filtered_data = process_quantaq_data(
                    quantaq_b_path,
                    instrument="QuantAQB",
                    burn_date=burn_date,
                    garage_closed_time=garage_time,
                    sum_bins=True,
                )

                if not filtered_data.empty:
                    quantaq_data["Bedroom"][burn_number] = filtered_data
                    print(f"  Processed QuantAQB: {burn_number} ({len(filtered_data)} records)")

        # Process Kitchen QuantAQ
        print("Loading QuantAQK data...")
        for burn_number in quantaq_burns:
            burn_date_row = burn_log[burn_log["Burn ID"] == burn_number]
            if not burn_date_row.empty:
                burn_date = pd.to_datetime(burn_date_row["Date"].iloc[0])
                garage_time = garage_closed_times.get(burn_number)

                filtered_data = process_quantaq_data(
                    quantaq_k_path,
                    instrument="QuantAQK",
                    burn_date=burn_date,
                    garage_closed_time=garage_time,
                    sum_bins=True,
                )

                if not filtered_data.empty:
                    quantaq_data["Kitchen"][burn_number] = filtered_data
                    print(f"  Processed QuantAQK: {burn_number} ({len(filtered_data)} records)")

# %% Process SMPS data
smps_data = {}

if "SMPS" in selected_instruments:
    print("\nProcessing SMPS data...")

    for burn_number in burn_numbers:
        burn_date_row = burn_log[burn_log["Burn ID"] == burn_number]
        if not burn_date_row.empty:
            burn_date = pd.to_datetime(burn_date_row["Date"].iloc[0])
            date_str = burn_date.strftime("%m%d%Y")
            SMPS_FILENAME = f"MH_apollo_bed_{date_str}_numConc.xlsx"
            smps_path = get_instrument_path('smps') / SMPS_FILENAME

            if smps_path.exists():
                garage_time = garage_closed_times.get(burn_number)

                filtered_data = process_smps_data(
                    smps_path,
                    garage_closed_time=garage_time,
                    sum_ranges=True,
                    debug=DEBUG,
                )

                if not filtered_data.empty:
                    smps_data[burn_number] = filtered_data
                    print(f"  Processed SMPS: {burn_number} ({len(filtered_data)} records)")
            else:
                print(f"  No SMPS file for {burn_number}")


# %% Create plots for each burn
def create_plots_for_burn(burn_number):
    """Create bedroom and kitchen plots for a specific burn"""
    print(f"\n{'-'*50}")
    print(f"Creating plots for {burn_number}")
    print(f"{'-'*50}")

    # Skip if no garage closed time available
    if burn_number not in garage_closed_times:
        print(f"Skipping {burn_number}: No garage closed time")
        return

    # Helper function to extract size from column name for sorting
    def extract_size(col_name):
        """Extract the first size value from a column name for sorting."""
        try:
            # Extract the size range (e.g., "Ʃ0.5-1.0µm" -> [0.5, 1.0])
            if "nm" in col_name:
                parts = (
                    col_name.replace("Ʃ", "")
                    .replace("nm (#/cm³)", "")
                    .split("-")
                )
            else:
                parts = (
                    col_name.replace("Ʃ", "")
                    .replace("µm (#/cm³)", "")
                    .split("-")
                )
            return float(parts[0])
        except (ValueError, IndexError):
            return float("inf")  # Put columns with parsing errors at the end

    # Define color palettes for each instrument type
    color_palettes = {"AeroTrak": Blues[9], "QuantAQ": Greens[9], "SMPS": Reds[9]}

    # Create Bedroom Plot
    bedroom_data = {}
    if "AeroTrak" in selected_instruments and burn_number in aerotrak_data["Bedroom"]:
        bedroom_data["AeroTrak"] = aerotrak_data["Bedroom"][burn_number]
    if "QuantAQ" in selected_instruments and burn_number in quantaq_data["Bedroom"]:
        bedroom_data["QuantAQ"] = quantaq_data["Bedroom"][burn_number]
    if "SMPS" in selected_instruments and burn_number in smps_data:
        bedroom_data["SMPS"] = smps_data[burn_number]

    if bedroom_data:
        # Create bedroom figure
        bedroom_title = f"{burn_number.upper()} Bedroom Particle Count Comparison"
        p_bedroom = create_standard_figure(
            bedroom_title,
            y_axis_label="Particulate Matter Particle Count (#/cm³)",
            x_range=(-1, 4),
            y_range=(1e-4, 1e5),
        )

        # Process and plot each instrument's data
        for instrument_name, instrument_data in bedroom_data.items():
            # Add instrument name to legend first (as a dummy line)
            p_bedroom.line([], [], legend_label=instrument_name, line_width=0)

            # Enhanced debug for SMPS
            if instrument_name == "SMPS" and DEBUG:
                print(f"\nDEBUG: Processing SMPS data for {burn_number}")
                print(f"Shape: {instrument_data.shape}")
                print(f"Columns: {instrument_data.columns.tolist()}")
                smps_time_col = "Time Since Garage Closed (hours)"
                if smps_time_col in instrument_data.columns:
                    time_stats = instrument_data[smps_time_col].describe()
                    print(
                        f"{smps_time_col} stats: min={time_stats['min']:.2f}, mean={time_stats['mean']:.2f}, max={time_stats['max']:.2f}"
                    )
                    # Check for NaN values
                    nan_count = instrument_data[smps_time_col].isna().sum()
                    print(
                        f"NaN count in {smps_time_col}: {nan_count}/{len(instrument_data)} ({nan_count/len(instrument_data)*100:.1f}%)"
                    )

            # Find columns with particle count data
            bedroom_particle_cols = []
            for col in instrument_data.columns:
                if isinstance(col, str) and "(#/cm³)" in col:
                    bedroom_particle_cols.append(col)

            # Sort columns by size
            bedroom_particle_cols.sort(key=extract_size)

            # Debug: Check if we have particle columns for this instrument
            if not bedroom_particle_cols:
                print(
                    f"Warning: No particle columns found for {instrument_name} in {burn_number} bedroom data"
                )
                if instrument_name == "SMPS":
                    # Print all columns for debugging SMPS data
                    print(f"SMPS columns: {instrument_data.columns.tolist()}")
                    # Try to force add some SMPS data columns if possible
                    for col in instrument_data.columns:
                        if isinstance(col, (int, float)) or (
                            isinstance(col, str) and col.replace(".", "", 1).isdigit()
                        ):
                            # This is likely a numeric size column from SMPS
                            numeric_col = float(col) if isinstance(col, str) else col
                            if 9 <= numeric_col <= 437:  # Within typical SMPS range
                                bedroom_particle_cols.append(col)
                continue

            # Plot each particle size range
            for idx, col in enumerate(bedroom_particle_cols):
                # Get color from palette
                color = color_palettes[instrument_name][
                    idx % len(color_palettes[instrument_name])
                ]

                # Special handling for SMPS data
                if instrument_name == "SMPS":
                    try:
                        # Get time and data columns with extra checks
                        time_column = "Time Since Garage Closed (hours)"

                        # Check if both columns exist
                        if time_column not in instrument_data.columns:
                            print(
                                f"Error: Time column missing for SMPS in {burn_number}"
                            )
                            continue
                        if col not in instrument_data.columns:
                            print(
                                f"Error: Data column {col} missing for SMPS in {burn_number}"
                            )
                            continue

                        # Get time and concentration values
                        time_values = instrument_data[time_column].values
                        concentration_values = instrument_data[col].values

                        # Check if all time values are NaN
                        if np.all(np.isnan(time_values)):
                            print(
                                f"Warning: All time values are NaN for {burn_number} SMPS data"
                            )
                            print(
                                "This might be due to an issue with datetime conversion."
                            )
                            print(
                                "Check the 'Date' and 'Start Time' formats in the source file."
                            )
                            continue

                        # Filter out NaN, zero, and negative values
                        valid_mask = (
                            (~np.isnan(time_values))
                            & (~np.isnan(concentration_values))
                            & (concentration_values > 0)
                        )

                        # Check if we have valid data
                        if np.sum(valid_mask) < 2:
                            print(
                                f"Warning: Not enough valid data points for SMPS {col} in {burn_number}"
                            )
                            if DEBUG:
                                print(f"Total points: {len(time_values)}")
                                print(f"NaN in time: {np.sum(np.isnan(time_values))}")
                                print(
                                    f"NaN in concentration: {np.sum(np.isnan(concentration_values))}"
                                )
                                print(
                                    f"Non-positive concentration: {np.sum(concentration_values <= 0)}"
                                )
                            continue

                        # Get valid data points
                        x_values = time_values[valid_mask]
                        y_values = concentration_values[valid_mask]

                        # Debug check if we have any x_values in the visible range
                        if DEBUG:
                            in_range = np.sum((x_values >= -1) & (x_values <= 4))
                            total = len(x_values)
                            print(
                                f"SMPS {col}: {in_range}/{total} points in visible range (-1 to 4 hours)"
                            )

                            # Print some sample values for verification
                            if len(x_values) > 0:
                                print("Sample values (time, concentration):")
                                for i in range(min(5, len(x_values))):
                                    print(f"  {x_values[i]:.2f}, {y_values[i]:.2f}")

                        # Only proceed if we have data points in range
                        in_range_mask = (x_values >= -1) & (x_values <= 4)
                        if np.sum(in_range_mask) < 2:
                            print(
                                f"Warning: Not enough in-range data points for SMPS {col} in {burn_number}"
                            )
                            continue

                        # Get in-range values
                        x_in_range = x_values[in_range_mask]
                        y_in_range = y_values[in_range_mask]

                        # Sort by x values
                        sort_indices = np.argsort(x_in_range)
                        x_sorted = x_in_range[sort_indices]
                        y_sorted = y_in_range[sort_indices]

                        # Format legend label
                        legend_label = col.replace(" (#/cm³)", "")

                        # Plot directly
                        p_bedroom.line(
                            x_sorted,
                            y_sorted,
                            legend_label=legend_label,
                            line_width=2,
                            color=color,
                            line_dash="solid",
                        )
                        # line_dash='dotted')

                        if DEBUG:
                            print(f"Successfully plotted SMPS {col} for {burn_number}")

                    except (ValueError, KeyError, IndexError) as e:
                        print(f"Error plotting SMPS {col} for {burn_number}: {str(e)}")
                        traceback.print_exc()
                        continue

                else:
                    # Standard handling for other instruments
                    try:
                        # Create a temporary dataframe with x, y columns
                        temp_df = pd.DataFrame(
                            {
                                "x": instrument_data[
                                    "Time Since Garage Closed (hours)"
                                ],
                                "y": instrument_data[col],
                            }
                        )

                        # Split data by NaNs to prevent lines connecting across gaps
                        segments = split_data_by_nan(temp_df, "x", "y")

                        # Format legend label (just the size range without units)
                        legend_label = col.replace(" (#/cm³)", "")

                        # Plot each segment
                        for j, (segment_x, segment_y) in enumerate(segments):
                            if (
                                len(segment_x) > 1
                            ):  # Only plot segments with multiple points
                                # Only add to legend for first segment of each column
                                if j == 0:
                                    p_bedroom.line(
                                        segment_x,
                                        segment_y,
                                        legend_label=legend_label,
                                        line_width=2,
                                        color=color,
                                        line_dash="solid",
                                    )
                                else:
                                    p_bedroom.line(
                                        segment_x,
                                        segment_y,
                                        line_width=2,
                                        color=color,
                                        line_dash="solid",
                                    )
                                # line_dash='solid' if instrument_name == 'AeroTrak' else 'dashed')
                    except (ValueError, KeyError, IndexError) as e:
                        print(
                            f"Error plotting {instrument_name} {col} for {burn_number}: {str(e)}"
                        )
                        continue

        # Add event markers
        events = {}
        if burn_number in garage_closed_times:
            events['Garage Closed'] = 0
        if burn_number in cr_box_hours:
            events['CR Boxes On'] = cr_box_hours[burn_number]
        add_event_markers(p_bedroom, events, y_range=(1e-4, 1e5))

        # Configure legend
        configure_legend(p_bedroom, location='top_right', click_policy='hide')

        # Add metadata
        metadata = get_script_metadata()
        bedroom_instruments_list = list(bedroom_data.keys())
        instruments_str = ", ".join(bedroom_instruments_list)
        bedroom_info_div = create_metadata_div(
            f"<p><b>{burn_number.upper()} Bedroom</b> | Instruments: {instruments_str}<br>"
            f"<small>{metadata}</small></p>",
            width=800,
        )

        # Create the bedroom layout
        bedroom_layout = column(p_bedroom, bedroom_info_div)

        # Show bedroom plot
        show(bedroom_layout)

        output_file_path = output_dir / f"{burn_number}_bedroom_particle_count_comparison.html"
        output_file(str(output_file_path))
        print(f"Bedroom figure would be saved to: {output_file_path}")

    # Create Kitchen Plot
    kitchen_data = {}
    if "AeroTrak" in selected_instruments and burn_number in aerotrak_data["Kitchen"]:
        kitchen_data["AeroTrak"] = aerotrak_data["Kitchen"][burn_number]
    if "QuantAQ" in selected_instruments and burn_number in quantaq_data["Kitchen"]:
        kitchen_data["QuantAQ"] = quantaq_data["Kitchen"][burn_number]

    if kitchen_data:
        # Create kitchen figure
        kitchen_title = f"{burn_number.upper()} Kitchen Particle Count Comparison"
        p_kitchen = create_standard_figure(
            kitchen_title,
            y_axis_label="Particulate Matter Particle Count (#/cm³)",
            x_range=(-1, 4),
            y_range=(1e-4, 1e5),
        )

        # Plot each kitchen instrument
        for instrument_name, instrument_data in kitchen_data.items():
            # Add instrument name to legend first (as a dummy line)
            p_kitchen.line([], [], legend_label=instrument_name, line_width=0)

            # Find columns with particle count data
            particle_cols = []
            for col in instrument_data.columns:
                if isinstance(col, str) and "(#/cm³)" in col:
                    particle_cols.append(col)

            # Sort particle columns by size
            particle_cols.sort(key=extract_size)

            # Debug: Check if we have particle columns for this instrument
            if not particle_cols:
                print(
                    f"Warning: No particle columns found for {instrument_name} in {burn_number} kitchen data"
                )
                print(f"Available columns: {instrument_data.columns.tolist()}")
                continue

            # Plot each particle size range
            for i, col in enumerate(particle_cols):
                # Get color from palette
                color = color_palettes[instrument_name][
                    i % len(color_palettes[instrument_name])
                ]

                # Create a temporary dataframe with x, y columns
                temp_df = pd.DataFrame(
                    {
                        "x": instrument_data["Time Since Garage Closed (hours)"],
                        "y": instrument_data[col],
                    }
                )

                # Split data by NaNs to prevent lines connecting across gaps
                segments = split_data_by_nan(temp_df, "x", "y")

                # Format legend label (just the size range)
                legend_label = col.replace(" (#/cm³)", "")

                # Plot each segment
                for j, (segment_x, segment_y) in enumerate(segments):
                    if len(segment_x) > 1:  # Only plot segments with multiple points
                        # Only add to legend for first segment of each column
                        if j == 0:
                            p_kitchen.line(
                                segment_x,
                                segment_y,
                                legend_label=legend_label,
                                line_width=2,
                                color=color,
                                line_dash="solid",
                            )
                        else:
                            p_kitchen.line(
                                segment_x,
                                segment_y,
                                line_width=2,
                                color=color,
                                line_dash="solid",
                            )
                        # line_dash='solid' if instrument_name == 'AeroTrak' else 'dashed')

        # Add event markers
        events = {}
        if burn_number in garage_closed_times:
            events['Garage Closed'] = 0
        if burn_number in cr_box_hours:
            events['CR Boxes On'] = cr_box_hours[burn_number]
        add_event_markers(p_kitchen, events, y_range=(1e-4, 1e5))

        # Configure legend
        configure_legend(p_kitchen, location='top_right', click_policy='hide')

        # Add metadata
        metadata = get_script_metadata()
        kitchen_instruments_list = list(kitchen_data.keys())
        instruments_str = ", ".join(kitchen_instruments_list)
        kitchen_info_div = create_metadata_div(
            f"<p><b>{burn_number.upper()} Kitchen</b> | Instruments: {instruments_str}<br>"
            f"<small>{metadata}</small></p>",
            width=800,
        )

        # Create the kitchen layout
        kitchen_layout = column(p_kitchen, kitchen_info_div)

        # Show kitchen plot
        show(kitchen_layout)

        output_file_path = output_dir / f"{burn_number}_kitchen_particle_count_comparison.html"
        output_file(str(output_file_path))
        print(f"Kitchen figure would be saved to: {output_file_path}")


# %% Main execution
# Prepare to output the plot in notebook
output_notebook()

# Process each burn
for burn_number in burn_numbers:
    create_plots_for_burn(burn_number)
# %%
