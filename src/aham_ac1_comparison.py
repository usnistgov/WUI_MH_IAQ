#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AHAM AC-1 Test Standard Comparison
===================================

Compare AHAM AC-1 test smoke concentrations (24,000-35,000 #/cm³, 0.1-1.0 µm)
with WUI fire QuantAQ measurements to determine equivalent PM1 mass concentrations.

Calculates conversion factors between particle number concentration (#/cm³) and
PM1 mass (µg/m³) at peak concentrations for burns 4-10, then translates AC-1
standard range to mass units for both bedroom2 and morning room locations.

Note: QuantAQ measures 0.35-1.0 µm (bins 0-2) vs AC-1 spec of 0.1-1.0 µm,
providing conservative estimates (excludes 0.1-0.35 µm particles).

Outputs:
    Terminal: Results table with burn ID, instrument, max PM1, particle count,
              and AC-1 equivalent ranges. Summary statistics (mean, median, std dev)
              for all metrics and per-instrument comparisons.

Author: Nathan Lima, NIST
Date: 2025
"""

# ============================================================================
# IMPORT MODULES
# ============================================================================
# Standard library
import warnings
import sys
from pathlib import Path

# Third-party
import pandas as pd

# Add repository root to path for portable data access
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

# Local imports
from data_paths import get_common_file, get_instrument_path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# File paths
BURN_LOG_PATH = get_common_file('burn_log')
QUANTAQ_BEDROOM_PATH = get_instrument_path('quantaq_bedroom') / 'MOD-PM-00194-b0fc215029fa4852b926bc50b28fda5a.csv'
QUANTAQ_MORNING_ROOM_PATH = get_instrument_path('quantaq_kitchen') / 'MOD-PM-00197-a6dd467a147a4d95a7b98a8a10ab4ea3.csv'

# AHAM AC-1 test standard concentration range (#/cm³) for particles 0.1-1.0 µm
AC1_LOWER_LIMIT = 24000  # particles/cm³
AC1_UPPER_LIMIT = 35000  # particles/cm³

# QuantAQ instrument time shift corrections (minutes)
# Applied to synchronize instruments to common time reference
TIME_SHIFTS = {"Bedroom2": -2.97, "Morning Room": 0.0}  # MOD-PM-00194  # MOD-PM-00197

# Burn IDs to analyze (burns 4-10)
BURN_IDS = ["burn4", "burn5", "burn6", "burn7", "burn8", "burn9", "burn10"]

# QuantAQ bin definitions (particle size ranges in µm)
# bin0: 0.35-0.46 µm, bin1: 0.46-0.66 µm, bin2: 0.66-1.0 µm
PARTICLE_BINS = ["bin0", "bin1", "bin2"]
# PARTICLE_BINS = ["neph_bin0", "neph_bin1", "neph_bin2"] #added for testing with neph data


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
def load_burn_log(filepath):
    """Load burn log Excel file and return Sheet2 dataframe

    Parameters:
        filepath (str): Path to burn_log.xlsx

    Returns:
        pd.DataFrame: Burn log data with burn dates and metadata
    """
    try:
        burn_log = pd.read_excel(filepath, sheet_name="Sheet2")
        print(f"Loaded burn log: {len(burn_log)} entries")
        return burn_log
    except (FileNotFoundError, ValueError, OSError) as e:
        print(f"ERROR loading burn log: {str(e)[:100]}")
        return None


def load_quantaq_data(filepath, instrument_name):
    """Load QuantAQ CSV data and prepare for analysis

    Parameters:
        filepath (str): Path to QuantAQ CSV file
        instrument_name (str): Instrument identifier for reporting

    Returns:
        pd.DataFrame: QuantAQ data with processed timestamp column
    """
    try:
        # Read CSV
        data = pd.read_csv(filepath)

        # Convert timestamp from ISO format to datetime
        # Original format: "2024-01-15T12:30:00Z"
        data["timestamp_local"] = pd.to_datetime(
            data["timestamp_local"].str.replace("T", " ").str.replace("Z", ""),
            errors="coerce",
        ).dt.tz_localize(None)

        # Apply time shift correction for synchronization
        time_shift_minutes = TIME_SHIFTS.get(instrument_name, 0.0)
        data["timestamp_local"] += pd.Timedelta(minutes=time_shift_minutes)

        # Create date column for filtering
        data["Date"] = data["timestamp_local"].dt.date  # type: ignore[attr-defined]

        print(f"  Loaded {instrument_name}: {len(data)} records")
        return data

    except (FileNotFoundError, ValueError, KeyError, OSError) as e:
        print(f"  ERROR loading {instrument_name}: {str(e)[:100]}")
        return None


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================
def calculate_peak_metrics(data, burn_date, burn_id, instrument_name):
    """Calculate peak PM1 and corresponding particle count for a single burn

    Parameters:
        data (pd.DataFrame): QuantAQ data with pm1 and particle bin columns
        burn_date (datetime.date): Date of the burn to analyze
        burn_id (str): Burn identifier (e.g., "burn4")
        instrument_name (str): Instrument identifier for reporting

    Returns:
        dict: Dictionary with keys: burn_id, instrument, max_pm1, particle_count,
              conversion_factor, ac1_lower, ac1_upper, or None if calculation fails
    """
    try:
        # Filter data for burn date
        burn_data = data[data["Date"] == burn_date].copy()

        if burn_data.empty:
            print(f"    WARNING: No data for {burn_id} on {burn_date}")
            return None

        # Find maximum PM1 value
        max_pm1 = burn_data["pm1"].max()

        if pd.isna(max_pm1):
            print(f"    WARNING: No valid PM1 data for {burn_id}")
            return None

        # Get timestamp of maximum PM1
        max_pm1_timestamp = burn_data.loc[burn_data["pm1"].idxmax(), "timestamp_local"]

        # Get particle counts at the peak PM1 timestamp
        # Sum bins 0-2 covering 0.35-1.0 µm range
        peak_row = burn_data[burn_data["timestamp_local"] == max_pm1_timestamp]

        if peak_row.empty:
            print(f"    WARNING: Cannot find peak timestamp for {burn_id}")
            return None

        # Sum particle bins
        particle_count = 0
        for bin_col in PARTICLE_BINS:
            if bin_col in peak_row.columns:
                bin_value = peak_row[bin_col].values[0]
                if not pd.isna(bin_value):
                    particle_count += bin_value

        if particle_count == 0:
            print(f"    WARNING: Zero particle count for {burn_id}")
            return None

        # Calculate conversion factor (µg/m³ per particle/cm³)
        conversion_factor = max_pm1 / particle_count

        # Calculate equivalent AC-1 concentration range in PM1 units
        ac1_lower_equiv = AC1_LOWER_LIMIT * conversion_factor
        ac1_upper_equiv = AC1_UPPER_LIMIT * conversion_factor

        return {
            "burn_id": burn_id,
            "instrument": instrument_name,
            "max_pm1": max_pm1,
            "particle_count": particle_count,
            "conversion_factor": conversion_factor,
            "ac1_lower": ac1_lower_equiv,
            "ac1_upper": ac1_upper_equiv,
        }

    except (KeyError, ValueError, IndexError) as e:
        print(f"    ERROR processing {burn_id} for {instrument_name}: {str(e)[:100]}")
        return None


def process_all_burns(burn_log, quantaq_data_dict):
    """Process all burns and compile results

    Parameters:
        burn_log (pd.DataFrame): Burn log with dates
        quantaq_data_dict (dict): Dictionary mapping instrument names to data DataFrames

    Returns:
        pd.DataFrame: Compiled results for all burns and instruments
    """
    results_list = []

    print(f"\n{'='*80}")
    print("PROCESSING BURNS 4-10")
    print(f"{'='*80}")

    for burn_id in BURN_IDS:
        # Get burn date from burn log
        burn_row = burn_log[burn_log["Burn ID"] == burn_id]

        if burn_row.empty:
            print(f"\nWARNING: {burn_id} not found in burn log")
            continue

        burn_date = pd.to_datetime(burn_row["Date"].iloc[0]).date()
        print(f"\n{burn_id.upper()} ({burn_date}):")

        # Process each instrument
        for instrument_name, instrument_data in quantaq_data_dict.items():
            if instrument_data is None:
                continue

            result = calculate_peak_metrics(
                instrument_data, burn_date, burn_id, instrument_name
            )

            if result is not None:
                results_list.append(result)
                print(
                    f"  {instrument_name}: "
                    f"PM1={result['max_pm1']:.1f} µg/m³, "
                    f"Count={result['particle_count']:.1f} #/cm³"
                )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)

    return results_df


# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================
def print_results_table(results_df):
    """Print formatted table of results to console

    Parameters:
        results_df (pd.DataFrame): Results dataframe with all metrics
    """
    print(f"\n{'='*80}")
    print("AHAM AC-1 CONCENTRATION RANGES")
    print(f"{'='*80}")
    print(f"AC-1 Standard: {AC1_LOWER_LIMIT:,}–{AC1_UPPER_LIMIT:,} #/cm³ (0.1–1.0 µm)")
    print("QuantAQ Measurement Range: 0.35–1.0 µm (excludes 0.1–0.35 µm particles)")
    print(f"{'='*80}\n")

    # Print header
    header = (
        f"{'Burn':<8} {'Instrument':<15} {'Max PM1':>12} {'Particle Count':>16} "
        f"{'AC-1 Range':>28}"
    )
    print(header)
    print(f"{'':<8} {'':<15} {'(µg/m³)':>12} {'(#/cm³)':>16} {'(µg/m³)':>28}")
    print("-" * len(header))

    # Print each result row
    for _, row in results_df.iterrows():
        print(
            f"{row['burn_id']:<8} "
            f"{row['instrument']:<15} "
            f"{row['max_pm1']:>12.1f} "
            f"{row['particle_count']:>16.1f} "
            f"{row['ac1_lower']:>12.1f} – {row['ac1_upper']:>12.1f}"
        )

    print("-" * len(header))


def print_summary_statistics(results_df):
    """Calculate and print summary statistics

    Parameters:
        results_df (pd.DataFrame): Results dataframe with all metrics
    """
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")

    # Overall statistics
    print("Overall Statistics (All Burns, Both Instruments):")
    print("  Max PM1:")
    print(f"    Mean:   {results_df['max_pm1'].mean():>10.1f} µg/m³")
    print(f"    Median: {results_df['max_pm1'].median():>10.1f} µg/m³")
    print(f"    Std Dev:{results_df['max_pm1'].std():>10.1f} µg/m³")

    print("\n  Particle Count (0.35–1.0 µm):")
    print(f"    Mean:   {results_df['particle_count'].mean():>10.1f} #/cm³")
    print(f"    Median: {results_df['particle_count'].median():>10.1f} #/cm³")
    print(f"    Std Dev:{results_df['particle_count'].std():>10.1f} #/cm³")

    print("\n  Conversion Factor (µg/m³ per #/cm³):")
    print(f"    Mean:   {results_df['conversion_factor'].mean():>10.6f}")
    print(f"    Median: {results_df['conversion_factor'].median():>10.6f}")
    print(f"    Std Dev:{results_df['conversion_factor'].std():>10.6f}")

    print("\n  AC-1 Range (Lower Bound):")
    print(f"    Mean:   {results_df['ac1_lower'].mean():>10.1f} µg/m³")
    print(f"    Median: {results_df['ac1_lower'].median():>10.1f} µg/m³")
    print(f"    Std Dev:{results_df['ac1_lower'].std():>10.1f} µg/m³")

    print("\n  AC-1 Range (Upper Bound):")
    print(f"    Mean:   {results_df['ac1_upper'].mean():>10.1f} µg/m³")
    print(f"    Median: {results_df['ac1_upper'].median():>10.1f} µg/m³")
    print(f"    Std Dev:{results_df['ac1_upper'].std():>10.1f} µg/m³")

    # Instrument-specific statistics
    print(f"\n{'-'*80}")
    print("Instrument Comparison:")
    print("-" * 80 + "\n")

    for instrument in results_df["instrument"].unique():
        instrument_data = results_df[results_df["instrument"] == instrument]
        print(f"{instrument}:")
        print(f"  Number of burns: {len(instrument_data)}")
        print(f"  Mean Max PM1: {instrument_data['max_pm1'].mean():>10.1f} µg/m³")
        print(
            f"  Mean Particle Count: {instrument_data['particle_count'].mean():>10.1f} #/cm³"
        )
        print(
            f"  Mean AC-1 Range: "
            f"{instrument_data['ac1_lower'].mean():.1f}–"
            f"{instrument_data['ac1_upper'].mean():.1f} µg/m³\n"
        )


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Execute complete AHAM AC-1 comparison analysis"""

    print(f"\n{'='*80}")
    print("AHAM AC-1 TEST STANDARD COMPARISON ANALYSIS")
    print(f"{'='*80}\n")

    # Load burn log
    print("Loading burn log...")
    burn_log = load_burn_log(BURN_LOG_PATH)

    if burn_log is None:
        print("FATAL ERROR: Cannot load burn log. Exiting.")
        return None

    # Load QuantAQ data from both instruments
    print("\nLoading QuantAQ sensor data...")
    quantaq_data_dict = {
        "Bedroom2": load_quantaq_data(QUANTAQ_BEDROOM_PATH, "Bedroom2"),
        "Morning Room": load_quantaq_data(QUANTAQ_MORNING_ROOM_PATH, "Morning Room"),
    }

    # Check if both instruments loaded successfully
    if all(data is None for data in quantaq_data_dict.values()):
        print("FATAL ERROR: Cannot load any QuantAQ data. Exiting.")
        return None

    # Process all burns
    results_df = process_all_burns(burn_log, quantaq_data_dict)

    if results_df.empty:
        print("\nWARNING: No results calculated. Check data files and date ranges.")
        return None

    # Print results
    print_results_table(results_df)
    print_summary_statistics(results_df)

    # Additional context note
    print(f"\n{'='*80}")
    print("IMPORTANT NOTES")
    print(f"{'='*80}")
    print("1. QuantAQ particle bins measure 0.35–1.0 µm, while AC-1 standard")
    print("   specifies 0.1–1.0 µm. This means particles between 0.1–0.35 µm")
    print("   are not captured in these measurements.")
    print("\n2. The calculated AC-1 equivalent ranges represent conservative")
    print("   estimates. True AC-1 equivalent concentrations would be higher")
    print("   if particles in the 0.1–0.35 µm range were included.")
    print(f"{'='*80}\n")

    return results_df


if __name__ == "__main__":
    results = main()

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
