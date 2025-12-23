"""
Data Loading Utilities for NIST WUI MH IAQ Analysis

This module provides instrument-specific data loading and processing functions
that are commonly used across multiple analysis scripts.

Functions:
    - load_burn_log: Load and parse burn log Excel file
    - process_aerotrak_data: Process AeroTrak particle counter data
    - process_quantaq_data: Process QuantAQ sensor data
    - process_smps_data: Process SMPS (Scanning Mobility Particle Sizer) data
    - process_dusttrak_data: Process DustTrak PM monitor data
    - process_purpleair_data: Process PurpleAir sensor data
    - process_miniams_data: Process MiniAMS aerosol mass spectrometer data

Author: Nathan Lima
Date: 2024-2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import traceback

# Import utility functions from other modules
try:
    from scripts.datetime_utils import apply_time_shift, fix_smps_datetime
    from scripts.data_filters import (
        filter_by_status_columns,
        calculate_rolling_average_burn3,
    )
    from scripts.instrument_config import (
        AEROTRAK_CHANNELS,
        sum_quantaq_bins,
        sum_smps_ranges,
        convert_unit,
    )
except ImportError:
    # Fallback for when imported from different location
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.datetime_utils import apply_time_shift, fix_smps_datetime
    from scripts.data_filters import (
        filter_by_status_columns,
        calculate_rolling_average_burn3,
    )
    from scripts.instrument_config import (
        AEROTRAK_CHANNELS,
        sum_quantaq_bins,
        sum_smps_ranges,
        convert_unit,
    )


def load_burn_log(burn_log_path, sheet_name="Sheet2"):
    """
    Load burn log Excel file with experiment metadata.

    Parameters:
    -----------
    burn_log_path : str or Path
        Path to burn log Excel file
    sheet_name : str, optional
        Sheet name to read (default: "Sheet2")

    Returns:
    --------
    pd.DataFrame
        Burn log with columns: Burn ID, Date, garage closed, CR Box on, etc.

    Examples:
    ---------
    >>> from src.data_paths import get_common_file
    >>> burn_log_path = get_common_file('burn_log')
    >>> burn_log = load_burn_log(burn_log_path)
    >>> burn_log['Burn ID'].tolist()
    ['burn1', 'burn2', ..., 'burn10']
    """
    try:
        burn_log = pd.read_excel(burn_log_path, sheet_name=sheet_name)
        return burn_log
    except Exception as e:
        print(f"Error loading burn log: {str(e)}")
        raise


def process_aerotrak_data(
    file_path,
    instrument="AeroTrakB",
    burn_date=None,
    garage_closed_time=None,
    burn_number=None,
    apply_rolling_avg=True,
):
    """
    Process AeroTrak particle counter data.

    This function loads AeroTrak data, extracts size channel information,
    calculates particle concentrations, applies quality filtering, and
    optionally applies time shifts and rolling averages.

    Parameters:
    -----------
    file_path : str or Path
        Path to AeroTrak data file (Excel format)
    instrument : str, optional
        Instrument name for time shift ('AeroTrakB' or 'AeroTrakK')
        Default: 'AeroTrakB'
    burn_date : datetime or str, optional
        Date to filter data (if None, returns all data)
    garage_closed_time : datetime, optional
        Reference time for calculating elapsed time
    burn_number : str, optional
        Burn ID (e.g., 'burn3') - used to determine if rolling average needed
    apply_rolling_avg : bool, optional
        Whether to apply rolling average for burn3 (default: True)

    Returns:
    --------
    pd.DataFrame
        Processed AeroTrak data with columns:
        - Date and Time: Datetime (with time shift applied)
        - Time Since Garage Closed (hours): If garage_closed_time provided
        - Ʃ{size1}-{size2}µm (#/cm³): Particle count density for each channel
        - Status columns: Flow Status, Laser Status

    Examples:
    ---------
    >>> from src.data_paths import get_instrument_path
    >>> path = get_instrument_path('aerotrak_bedroom') / 'all_data.xlsx'
    >>> df = process_aerotrak_data(path, instrument='AeroTrakB',
    ...                             burn_date='2024-01-15',
    ...                             garage_closed_time=garage_time,
    ...                             burn_number='burn1')

    Notes:
    ------
    - Automatically filters by instrument status (Flow Status, Laser Status)
    - Converts particle counts from total to #/cm³
    - Applies 5-minute rolling average for burn3 if apply_rolling_avg=True
    - Size channel values extracted from first row of data
    """
    try:
        # Load data
        data = pd.read_excel(file_path)
        data.columns = data.columns.str.strip()

        # Extract size values for each channel
        size_values = {}
        for channel in AEROTRAK_CHANNELS:
            size_col = f"{channel} Size (µm)"
            if size_col in data.columns:
                size_value = data[size_col].iloc[0]
                if pd.notna(size_value):
                    size_values[channel] = size_value

        # Filter by burn date if provided
        if burn_date is not None:
            burn_date = pd.to_datetime(burn_date)
            data["Date"] = pd.to_datetime(data["Date and Time"]).dt.date
            data = data[data["Date"] == burn_date.date()].copy()

        if data.empty:
            print(f"No data found for {instrument} on {burn_date}")
            return pd.DataFrame()

        # Apply time shift
        data = apply_time_shift(data, instrument, "Date and Time")

        # Calculate time since garage closed if provided
        if garage_closed_time is not None:
            from scripts.datetime_utils import calculate_time_since_garage_closed
            data = calculate_time_since_garage_closed(
                data, "Date and Time", garage_closed_time
            )

        # Filter by status columns
        data = filter_by_status_columns(
            data, status_columns=["Flow Status", "Laser Status"]
        )

        # Convert Diff counts to #/cm³
        volume_column = "Volume (L)"
        if volume_column in data.columns:
            data["Volume (cm³)"] = convert_unit(data[volume_column], "liter_to_cm3")

            # Process each channel
            for i, channel in enumerate(AEROTRAK_CHANNELS):
                if channel in size_values:
                    # Get next channel's size value
                    next_channel = (
                        AEROTRAK_CHANNELS[i + 1]
                        if i < len(AEROTRAK_CHANNELS) - 1
                        else None
                    )
                    next_size_value = size_values.get(next_channel, 25)

                    diff_col = f"{channel} Diff (#)"
                    if diff_col in data.columns:
                        # Create column for particle count density
                        new_col_name = (
                            f"Ʃ{size_values[channel]}-{next_size_value}µm (#/cm³)"
                        )
                        data[new_col_name] = data[diff_col] / data["Volume (cm³)"]

        # Apply rolling average for burn3
        if apply_rolling_avg and burn_number == "burn3":
            data = calculate_rolling_average_burn3(data, "Date and Time")

        return data

    except Exception as e:
        print(f"Error processing {instrument} data: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()


def process_quantaq_data(
    file_path,
    instrument="QuantAQB",
    burn_date=None,
    garage_closed_time=None,
    sum_bins=True,
):
    """
    Process QuantAQ air quality sensor data.

    Parameters:
    -----------
    file_path : str or Path
        Path to QuantAQ data file (CSV format)
    instrument : str, optional
        Instrument name for time shift ('QuantAQB' or 'QuantAQK')
        Default: 'QuantAQB'
    burn_date : datetime or str, optional
        Date to filter data (if None, returns all data)
    garage_closed_time : datetime, optional
        Reference time for calculating elapsed time
    sum_bins : bool, optional
        Whether to sum bin columns into size ranges (default: True)

    Returns:
    --------
    pd.DataFrame
        Processed QuantAQ data with columns:
        - timestamp_local: Datetime (with time shift applied)
        - Time Since Garage Closed (hours): If garage_closed_time provided
        - PM1 (µg/m³), PM2.5 (µg/m³), PM10 (µg/m³): Mass concentrations
        - bin0-bin23: Raw bin counts
        - Ʃ{size_range}µm (#/cm³): Summed bin ranges (if sum_bins=True)

    Examples:
    ---------
    >>> from src.data_paths import get_instrument_path
    >>> path = get_instrument_path('quantaq_bedroom') / 'data.csv'
    >>> df = process_quantaq_data(path, instrument='QuantAQB',
    ...                            burn_date='2024-01-15')

    Notes:
    ------
    - Automatically reverses row order (QuantAQ files are reverse chronological)
    - Removes 'T' and 'Z' from timestamp format
    - Renames PM columns to standard format
    - Optionally sums bins into size ranges using instrument_config.QUANTAQ_BINS
    """
    try:
        # Load data
        data = pd.read_csv(file_path)

        # Reverse row order (QuantAQ files are in reverse chronological order)
        data = data.iloc[::-1].reset_index(drop=True)

        # Convert timestamp to datetime
        data["timestamp_local"] = pd.to_datetime(
            data["timestamp_local"].str.replace("T", " ").str.replace("Z", ""),
            errors="coerce",
        ).dt.tz_localize(None)

        # Rename PM columns to standard format
        rename_dict = {
            "pm1": "PM1 (µg/m³)",
            "pm25": "PM2.5 (µg/m³)",
            "pm10": "PM10 (µg/m³)",
        }
        data.rename(columns=rename_dict, inplace=True)

        # Filter by burn date if provided
        if burn_date is not None:
            burn_date = pd.to_datetime(burn_date)
            data["Date"] = data["timestamp_local"].dt.date
            data = data[data["Date"] == burn_date.date()].copy()

        if data.empty:
            print(f"No data found for {instrument} on {burn_date}")
            return pd.DataFrame()

        # Apply time shift
        data = apply_time_shift(data, instrument, "timestamp_local")

        # Calculate time since garage closed if provided
        if garage_closed_time is not None:
            from scripts.datetime_utils import calculate_time_since_garage_closed
            data = calculate_time_since_garage_closed(
                data, "timestamp_local", garage_closed_time
            )

        # Sum bins into size ranges if requested
        if sum_bins:
            data = sum_quantaq_bins(data)

        return data

    except Exception as e:
        print(f"Error processing {instrument} data: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()


def process_smps_data(
    file_path,
    instrument="SMPS",
    garage_closed_time=None,
    sum_ranges=True,
    debug=False,
):
    """
    Process SMPS (Scanning Mobility Particle Sizer) data.

    Parameters:
    -----------
    file_path : str or Path
        Path to SMPS data file (Excel format)
    instrument : str, optional
        Instrument name for time shift (default: 'SMPS')
    garage_closed_time : datetime, optional
        Reference time for calculating elapsed time
    sum_ranges : bool, optional
        Whether to sum size columns into ranges (default: True)
    debug : bool, optional
        Print debug information (default: False)

    Returns:
    --------
    pd.DataFrame
        Processed SMPS data with columns:
        - datetime: Combined date and time
        - Time Since Garage Closed (hours): If garage_closed_time provided
        - {size_nm}: Concentration for each size bin (column names are floats)
        - Ʃ{start}-{end}nm (#/cm³): Summed size ranges (if sum_ranges=True)

    Examples:
    ---------
    >>> from src.data_paths import get_instrument_path
    >>> path = get_instrument_path('smps') / 'MH_apollo_bed_01152024_numConc.xlsx'
    >>> df = process_smps_data(path, garage_closed_time=garage_time)

    Notes:
    ------
    - Handles transposed data formats automatically
    - Combines Date and Start Time columns into datetime
    - Drops 'Total Concentration(#/cm³)' column if present
    - Sums columns by size ranges defined in instrument_config.SMPS_BIN_RANGES
    """
    try:
        # Try to load from different sheet names
        try:
            data = pd.read_excel(file_path, sheet_name="all_data")
        except (ValueError, KeyError):
            try:
                data = pd.read_excel(file_path, sheet_name="sheet1")
            except (ValueError, KeyError):
                data = pd.read_excel(file_path)

        if debug:
            print(f"  SMPS raw data shape: {data.shape}")
            print(f"  First few columns: {data.columns[:5].tolist()}")

        # Check if data needs to be transposed
        if "Date" not in data.columns or "Start Time" not in data.columns:
            if isinstance(data.iloc[0].values, np.ndarray):
                if "Date" in data.iloc[0].values and "Start Time" in data.iloc[0].values:
                    if debug:
                        print("  Transposing SMPS data...")
                    data = data.transpose()
                    data.columns = data.iloc[0].values
                    data = data.iloc[1:].reset_index(drop=True)

        # Drop Total Concentration column if present
        if "Total Concentration(#/cm³)" in data.columns:
            data.drop(columns=["Total Concentration(#/cm³)"], inplace=True)

        # Fix datetime
        data = fix_smps_datetime(data, debug=debug)

        # Apply time shift
        data = apply_time_shift(data, instrument, "datetime")

        # Calculate time since garage closed if provided
        if garage_closed_time is not None:
            from scripts.datetime_utils import calculate_time_since_garage_closed
            data = calculate_time_since_garage_closed(
                data, "datetime", garage_closed_time
            )

        # Sum size ranges if requested
        if sum_ranges:
            data = sum_smps_ranges(data)

        return data

    except Exception as e:
        print(f"Error processing SMPS data: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()


def process_dusttrak_data(
    file_path,
    instrument="DustTrak",
    burn_date=None,
    garage_closed_time=None,
):
    """
    Process DustTrak PM monitor data.

    Parameters:
    -----------
    file_path : str or Path
        Path to DustTrak data file
    instrument : str, optional
        Instrument name for time shift (default: 'DustTrak')
    burn_date : datetime or str, optional
        Date to filter data (if None, returns all data)
    garage_closed_time : datetime, optional
        Reference time for calculating elapsed time

    Returns:
    --------
    pd.DataFrame
        Processed DustTrak data with columns:
        - datetime: Timestamp
        - Time Since Garage Closed (hours): If garage_closed_time provided
        - PM15 (µg/m³): Converted from mg/m³

    Examples:
    ---------
    >>> df = process_dusttrak_data(path, burn_date='2024-01-15',
    ...                             garage_closed_time=garage_time)

    Notes:
    ------
    - Converts TOTAL [mg/m3] to PM15 (µg/m³) by multiplying by 1000
    """
    try:
        # Load data (format may vary - adjust as needed)
        data = pd.read_excel(file_path)

        # Convert mg/m³ to µg/m³
        if "TOTAL [mg/m3]" in data.columns:
            data["PM15 (µg/m³)"] = convert_unit(
                data["TOTAL [mg/m3]"], "mg_m3_to_ug_m3"
            )

        # Filter by burn date if provided and datetime column exists
        if burn_date is not None and "datetime" in data.columns:
            burn_date = pd.to_datetime(burn_date)
            data["Date"] = pd.to_datetime(data["datetime"]).dt.date
            data = data[data["Date"] == burn_date.date()].copy()

        if data.empty:
            print(f"No data found for {instrument} on {burn_date}")
            return pd.DataFrame()

        # Apply time shift if datetime column exists
        datetime_col = "datetime" if "datetime" in data.columns else "Date and Time"
        if datetime_col in data.columns:
            data = apply_time_shift(data, instrument, datetime_col)

            # Calculate time since garage closed if provided
            if garage_closed_time is not None:
                from scripts.datetime_utils import calculate_time_since_garage_closed
                data = calculate_time_since_garage_closed(
                    data, datetime_col, garage_closed_time
                )

        return data

    except Exception as e:
        print(f"Error processing DustTrak data: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()


def process_purpleair_data(
    file_path,
    instrument="PurpleAir",
    burn_date=None,
    garage_closed_time=None,
):
    """
    Process PurpleAir sensor data.

    Parameters:
    -----------
    file_path : str or Path
        Path to PurpleAir data file
    instrument : str, optional
        Instrument name (default: 'PurpleAir')
    burn_date : datetime or str, optional
        Date to filter data (if None, returns all data)
    garage_closed_time : datetime, optional
        Reference time for calculating elapsed time

    Returns:
    --------
    pd.DataFrame
        Processed PurpleAir data with columns:
        - datetime: Timestamp
        - Time Since Garage Closed (hours): If garage_closed_time provided
        - PM2.5 (µg/m³): Renamed from 'Average' column

    Examples:
    ---------
    >>> df = process_purpleair_data(path, burn_date='2024-01-15')

    Notes:
    ------
    - Renames 'Average' column to 'PM2.5 (µg/m³)'
    - Typically used for burns 6-10
    """
    try:
        # Load data
        data = pd.read_excel(file_path)

        # Rename Average to PM2.5
        if "Average" in data.columns:
            data.rename(columns={"Average": "PM2.5 (µg/m³)"}, inplace=True)

        # Filter by burn date if provided
        if burn_date is not None and "datetime" in data.columns:
            burn_date = pd.to_datetime(burn_date)
            data["Date"] = pd.to_datetime(data["datetime"]).dt.date
            data = data[data["Date"] == burn_date.date()].copy()

        if data.empty:
            print(f"No data found for {instrument} on {burn_date}")
            return pd.DataFrame()

        # Apply time shift if datetime column exists
        datetime_col = "datetime" if "datetime" in data.columns else "Date and Time"
        if datetime_col in data.columns:
            data = apply_time_shift(data, instrument, datetime_col)

            # Calculate time since garage closed if provided
            if garage_closed_time is not None:
                from scripts.datetime_utils import calculate_time_since_garage_closed
                data = calculate_time_since_garage_closed(
                    data, datetime_col, garage_closed_time
                )

        return data

    except Exception as e:
        print(f"Error processing {instrument} data: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()


def process_miniams_data(
    file_path,
    instrument="MiniAMS",
    burn_date=None,
    garage_closed_time=None,
):
    """
    Process MiniAMS (Aerosol Mass Spectrometer) data.

    Parameters:
    -----------
    file_path : str or Path
        Path to MiniAMS data file
    instrument : str, optional
        Instrument name (default: 'MiniAMS')
    burn_date : datetime or str, optional
        Date to filter data (if None, returns all data)
    garage_closed_time : datetime, optional
        Reference time for calculating elapsed time

    Returns:
    --------
    pd.DataFrame
        Processed MiniAMS data with columns:
        - datetime: Timestamp
        - Time Since Garage Closed (hours): If garage_closed_time provided
        - Organic (µg/m³): Organic aerosol mass
        - Nitrate (µg/m³): Nitrate mass
        - Sulfate (µg/m³): Sulfate mass
        - Ammonium (µg/m³): Ammonium mass
        - Chloride (µg/m³): Chloride mass

    Examples:
    ---------
    >>> df = process_miniams_data(path, burn_date='2024-01-15')

    Notes:
    ------
    - Renames Org→Organic, NO3→Nitrate, SO4→Sulfate, NH4→Ammonium, Chl→Chloride
    - Typically only available for burns 1-3
    """
    try:
        # Load data
        data = pd.read_excel(file_path)

        # Rename columns to standard format
        rename_dict = {
            "Org": "Organic (µg/m³)",
            "NO3": "Nitrate (µg/m³)",
            "SO4": "Sulfate (µg/m³)",
            "NH4": "Ammonium (µg/m³)",
            "Chl": "Chloride (µg/m³)",
        }
        data.rename(columns=rename_dict, inplace=True)

        # Filter by burn date if provided
        if burn_date is not None and "datetime" in data.columns:
            burn_date = pd.to_datetime(burn_date)
            data["Date"] = pd.to_datetime(data["datetime"]).dt.date
            data = data[data["Date"] == burn_date.date()].copy()

        if data.empty:
            print(f"No data found for {instrument} on {burn_date}")
            return pd.DataFrame()

        # Apply time shift if datetime column exists
        datetime_col = "datetime" if "datetime" in data.columns else "Date and Time"
        if datetime_col in data.columns:
            data = apply_time_shift(data, instrument, datetime_col)

            # Calculate time since garage closed if provided
            if garage_closed_time is not None:
                from scripts.datetime_utils import calculate_time_since_garage_closed
                data = calculate_time_since_garage_closed(
                    data, datetime_col, garage_closed_time
                )

        return data

    except Exception as e:
        print(f"Error processing MiniAMS data: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()


def get_garage_closed_times(burn_log, burn_numbers=None):
    """
    Extract garage closed times from burn log for specified burns.

    Parameters:
    -----------
    burn_log : pd.DataFrame
        Burn log DataFrame
    burn_numbers : list of str, optional
        List of burn IDs to process (e.g., ['burn1', 'burn2'])
        If None, processes all burns in the log

    Returns:
    --------
    dict
        Dictionary mapping burn_number to garage_closed_time (pd.Timestamp)

    Examples:
    ---------
    >>> burn_log = load_burn_log(burn_log_path)
    >>> times = get_garage_closed_times(burn_log, ['burn1', 'burn2'])
    >>> times['burn1']
    Timestamp('2024-01-15 14:30:00')
    """
    if burn_numbers is None:
        burn_numbers = burn_log["Burn ID"].tolist()

    garage_closed_times = {}

    for burn_number in burn_numbers:
        burn_date_row = burn_log[burn_log["Burn ID"] == burn_number]
        if not burn_date_row.empty:
            burn_date = pd.to_datetime(burn_date_row["Date"].iloc[0])

            # Get garage closed time
            garage_closed_time_str = burn_date_row["garage closed"].iloc[0]
            if pd.notna(garage_closed_time_str):
                garage_closed_time = pd.to_datetime(
                    f"{burn_date.strftime('%Y-%m-%d')} {garage_closed_time_str}"
                )
                garage_closed_times[burn_number] = garage_closed_time

    return garage_closed_times


def get_cr_box_times(burn_log, burn_numbers=None, relative_to_garage=True):
    """
    Extract CR Box activation times from burn log.

    Parameters:
    -----------
    burn_log : pd.DataFrame
        Burn log DataFrame
    burn_numbers : list of str, optional
        List of burn IDs to process
        If None, processes all burns in the log
    relative_to_garage : bool, optional
        If True, returns hours since garage closed
        If False, returns absolute timestamps
        Default: True

    Returns:
    --------
    dict
        Dictionary mapping burn_number to CR Box time
        (hours since garage closed if relative_to_garage=True,
         pd.Timestamp if relative_to_garage=False)

    Examples:
    ---------
    >>> burn_log = load_burn_log(burn_log_path)
    >>> times = get_cr_box_times(burn_log, relative_to_garage=True)
    >>> times['burn1']  # Hours since garage closed
    1.5
    """
    if burn_numbers is None:
        burn_numbers = burn_log["Burn ID"].tolist()

    garage_closed_times = get_garage_closed_times(burn_log, burn_numbers)
    cr_box_times = {}

    for burn_number in burn_numbers:
        burn_date_row = burn_log[burn_log["Burn ID"] == burn_number]
        if not burn_date_row.empty:
            burn_date = pd.to_datetime(burn_date_row["Date"].iloc[0])

            # Get CR Box activation time
            cr_box_on_time_str = burn_date_row["CR Box on"].iloc[0]
            if pd.notna(cr_box_on_time_str):
                cr_box_on_time = pd.to_datetime(
                    f"{burn_date.strftime('%Y-%m-%d')} {cr_box_on_time_str}"
                )

                if relative_to_garage and burn_number in garage_closed_times:
                    # Calculate hours since garage closed
                    cr_box_times[burn_number] = (
                        cr_box_on_time - garage_closed_times[burn_number]
                    ).total_seconds() / 3600
                else:
                    cr_box_times[burn_number] = cr_box_on_time

    return cr_box_times
