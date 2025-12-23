"""
DateTime Utility Functions for NIST WUI MH IAQ Analysis

This module provides common datetime handling functions used across
multiple analysis scripts in the NIST WUI MH IAQ repository.

Functions:
    - create_naive_datetime: Create timezone-naive datetime from date and time strings
    - apply_time_shift: Apply instrument-specific time corrections
    - calculate_time_since_event: Calculate elapsed time since a reference event
    - fix_smps_datetime: Fix datetime creation for SMPS data

Constants:
    - TIME_SHIFTS: Dictionary of instrument-specific time corrections (in minutes)

Author: Nathan Lima
Date: 2024-2025
"""

import pandas as pd
import numpy as np
import traceback

# Instrument-specific time shift corrections (in minutes)
# These values synchronize data from different instruments
TIME_SHIFTS = {
    "AeroTrakB": 2.16,
    "AeroTrakK": 5,
    "QuantAQB": -2.97,
    "QuantAQK": 0,
    "SMPS": 0,
    "DustTrak": 0,
    "PurpleAir": 0,
    "PurpleAirK": 0,
    "MiniAMS": 0,
}


def create_naive_datetime(date_str, time_str):
    """
    Create a timezone-naive datetime object from separate date and time strings.

    This function is useful when combining date and time columns from data files
    that may have timezone information that needs to be removed for consistency.

    Parameters:
    -----------
    date_str : str
        Date string in a format parseable by pd.to_datetime
    time_str : str
        Time string in a format parseable by pd.to_datetime

    Returns:
    --------
    pd.Timestamp or pd.NaT
        Timezone-naive datetime object, or NaT if conversion fails

    Examples:
    ---------
    >>> create_naive_datetime("2024-01-15", "14:30:00")
    Timestamp('2024-01-15 14:30:00')

    >>> create_naive_datetime("2024-01-15", "2:30 PM")
    Timestamp('2024-01-15 14:30:00')
    """
    # Combine date and time strings
    dt = pd.to_datetime(f"{date_str} {time_str}", errors="coerce")

    # Remove timezone information if present
    if pd.notna(dt) and hasattr(dt, "tz") and dt.tz is not None:
        dt = dt.tz_localize(None)

    return dt


def apply_time_shift(df, instrument, datetime_column):
    """
    Apply instrument-specific time shift to synchronize data across instruments.

    Different instruments may have clock offsets that need correction for
    proper temporal alignment. This function applies predefined time shifts
    based on the instrument type.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing data with datetime column
    instrument : str
        Instrument name (must be a key in TIME_SHIFTS dictionary)
        Options: "AeroTrakB", "AeroTrakK", "QuantAQB", "QuantAQK", "SMPS", etc.
    datetime_column : str
        Name of the column containing datetime values

    Returns:
    --------
    pd.DataFrame
        DataFrame with shifted datetime column (returns copy if shift applied)

    Examples:
    ---------
    >>> df = pd.DataFrame({
    ...     'datetime': pd.date_range('2024-01-01', periods=3, freq='1H'),
    ...     'value': [1, 2, 3]
    ... })
    >>> shifted_df = apply_time_shift(df, 'AeroTrakB', 'datetime')
    >>> # Datetime column shifted by 2.16 minutes

    Notes:
    ------
    - A copy of the DataFrame is created if a shift is applied to avoid
      SettingWithCopyWarning
    - If the instrument is not in TIME_SHIFTS, no shift is applied
    - Time shifts are defined in the TIME_SHIFTS constant dictionary
    """
    # Get time shift for the instrument (default to 0 if not found)
    time_shift = TIME_SHIFTS.get(instrument, 0)

    # Only apply shift if non-zero
    if time_shift != 0:
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        # Apply time shift
        df[datetime_column] = df[datetime_column] + pd.Timedelta(minutes=time_shift)

    return df


def calculate_time_since_event(df, datetime_column, event_time, output_column=None):
    """
    Calculate elapsed time in hours since a reference event.

    This function is commonly used to calculate time since garage closed,
    time since burn start, or other reference events in the experiments.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing data with datetime column
    datetime_column : str
        Name of the column containing datetime values
    event_time : pd.Timestamp or datetime
        Reference event datetime
    output_column : str, optional
        Name for the output column. If None, defaults to
        "Time Since Event (hours)"

    Returns:
    --------
    pd.DataFrame
        DataFrame with added time-since-event column

    Examples:
    ---------
    >>> df = pd.DataFrame({
    ...     'datetime': pd.date_range('2024-01-01 12:00', periods=3, freq='1H')
    ... })
    >>> event = pd.Timestamp('2024-01-01 12:00')
    >>> result = calculate_time_since_event(df, 'datetime', event, 'Hours')
    >>> result['Hours'].tolist()
    [0.0, 1.0, 2.0]

    Notes:
    ------
    - Returns original DataFrame if event_time is None
    - Automatically removes timezone information for consistency
    - Time is calculated in hours (fractional hours for sub-hour precision)
    """
    # Return original dataframe if no event time provided
    if event_time is None:
        return df

    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Ensure datetime column is properly formatted
    df[datetime_column] = pd.to_datetime(df[datetime_column])

    # Make datetime timezone-naive if it has timezone info
    if (
        hasattr(df[datetime_column].dtype, "tz")
        and df[datetime_column].dtype.tz is not None
    ):
        df[datetime_column] = df[datetime_column].dt.tz_localize(None)

    # Set default output column name if not provided
    if output_column is None:
        output_column = "Time Since Event (hours)"

    # Calculate time since event in hours
    df[output_column] = (
        df[datetime_column] - event_time
    ).dt.total_seconds() / 3600

    return df


def calculate_time_since_garage_closed(df, datetime_column, garage_closed_time):
    """
    Calculate time since garage door was closed.

    This is a convenience wrapper around calculate_time_since_event()
    specifically for the garage closed event, which is a common reference
    point in the WUI experiments.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing data with datetime column
    datetime_column : str
        Name of the column containing datetime values
    garage_closed_time : pd.Timestamp or datetime
        Datetime when garage door was closed

    Returns:
    --------
    pd.DataFrame
        DataFrame with added 'Time Since Garage Closed (hours)' column

    Examples:
    ---------
    >>> df = pd.DataFrame({
    ...     'datetime': pd.date_range('2024-01-01 12:00', periods=5, freq='30min')
    ... })
    >>> garage_time = pd.Timestamp('2024-01-01 12:00')
    >>> result = calculate_time_since_garage_closed(df, 'datetime', garage_time)
    >>> result['Time Since Garage Closed (hours)'].tolist()
    [0.0, 0.5, 1.0, 1.5, 2.0]
    """
    return calculate_time_since_event(
        df,
        datetime_column,
        garage_closed_time,
        output_column="Time Since Garage Closed (hours)"
    )


def fix_smps_datetime(smps_data, debug=False):
    """
    Fix datetime creation for SMPS (Scanning Mobility Particle Sizer) data.

    SMPS data files often have Date and Start Time in separate columns with
    inconsistent formats that require special handling for proper datetime
    creation.

    Parameters:
    -----------
    smps_data : pd.DataFrame
        DataFrame containing SMPS data with 'Date' and 'Start Time' columns
    debug : bool, optional
        If True, print diagnostic information during processing
        Default is False

    Returns:
    --------
    pd.DataFrame
        DataFrame with added 'datetime' column containing properly formatted
        datetime values

    Notes:
    ------
    - Tries multiple approaches to handle different SMPS file formats
    - If columns are missing, returns original DataFrame with error message
    - Creates new 'datetime' column; does not modify original columns
    - Reports number of successfully converted datetime values

    Examples:
    ---------
    >>> smps_df = pd.DataFrame({
    ...     'Date': ['2024-01-01', '2024-01-01'],
    ...     'Start Time': ['12:00:00', '13:00:00']
    ... })
    >>> result = fix_smps_datetime(smps_df)
    >>> 'datetime' in result.columns
    True
    """
    # Check if required columns exist
    if "Date" not in smps_data.columns or "Start Time" not in smps_data.columns:
        print("  Error: Date or Start Time columns missing from SMPS data")
        return smps_data

    # Make a copy to avoid modifying the original
    data = smps_data.copy()

    # Print sample values for debugging
    if debug:
        print(f"  Sample Date value: {data['Date'].iloc[0]}")
        print(f"  Sample Start Time value: {data['Start Time'].iloc[0]}")

    try:
        # First try direct approach with string operations
        datetime_strings = []
        for i in range(len(data)):
            try:
                row_date_str = str(data["Date"].iloc[i])
                time_str = str(data["Start Time"].iloc[i])
                if (
                    row_date_str != "nan"
                    and time_str != "nan"
                    and row_date_str != "NaT"
                    and time_str != "NaT"
                ):
                    datetime_strings.append(f"{row_date_str} {time_str}")
                else:
                    datetime_strings.append(np.nan)
            except (ValueError, KeyError, IndexError) as e:
                if debug:
                    print(f"  Error with row {i}: {str(e)}")
                datetime_strings.append(np.nan)

        data["datetime"] = pd.to_datetime(datetime_strings, errors="coerce")

        # If all values are NaT, try alternative approach
        if data["datetime"].isna().all():
            if debug:
                print("  First datetime conversion approach failed, trying alternative...")

            # Try to convert Date to datetime first
            data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

            # If Start Time is already a time object, convert to string
            if pd.api.types.is_datetime64_dtype(data["Start Time"]):
                data["Start Time"] = data["Start Time"].dt.strftime("%H:%M:%S")

            # For each row with a valid Date, combine with Start Time
            data["datetime"] = pd.NaT
            valid_dates = ~data["Date"].isna()

            if valid_dates.any():
                date_strings = data.loc[valid_dates, "Date"].dt.strftime("%Y-%m-%d")
                data.loc[valid_dates, "datetime"] = pd.to_datetime(
                    date_strings + " " + data.loc[valid_dates, "Start Time"],
                    errors="coerce",
                )
    except (ValueError, TypeError, KeyError) as e:
        print(f"  Error in datetime conversion: {str(e)}")
        if debug:
            traceback.print_exc()

    # Print stats on the new datetime column
    valid_dt_count = (~data["datetime"].isna()).sum()
    total_count = len(data)
    print(f"  Created {valid_dt_count}/{total_count} valid datetime values")

    return data


def seconds_to_hours(seconds):
    """
    Convert seconds to hours.

    Parameters:
    -----------
    seconds : float or pd.Timedelta
        Time in seconds or as a Timedelta object

    Returns:
    --------
    float
        Time in hours

    Examples:
    ---------
    >>> seconds_to_hours(3600)
    1.0

    >>> seconds_to_hours(7200)
    2.0
    """
    if isinstance(seconds, pd.Timedelta):
        return seconds.total_seconds() / 3600
    return seconds / 3600


def minutes_to_hours(minutes):
    """
    Convert minutes to hours.

    Parameters:
    -----------
    minutes : float
        Time in minutes

    Returns:
    --------
    float
        Time in hours

    Examples:
    ---------
    >>> minutes_to_hours(60)
    1.0

    >>> minutes_to_hours(90)
    1.5
    """
    return minutes / 60


def hours_to_minutes(hours):
    """
    Convert hours to minutes.

    Parameters:
    -----------
    hours : float
        Time in hours

    Returns:
    --------
    float
        Time in minutes

    Examples:
    ---------
    >>> hours_to_minutes(1.0)
    60.0

    >>> hours_to_minutes(1.5)
    90.0
    """
    return hours * 60
