"""
SMPS Total Concentration Export Utility

This script reads raw SMPS (Scanning Mobility Particle Sizer) data files and
exports a simplified CSV containing only datetime and total concentration values.
The output file is intended for sharing with collaborators.

The script handles the transposed SMPS data format where:
- First column contains all data labels
- Each subsequent column represents a time point
- Data includes Date, Start Time, and Total Concentration rows

Usage:
    1. Set CONCENTRATION_TYPE to either 'MassConc' or 'NumConc'
    2. Set OUTPUT_PATH to desired output directory
    3. Run: conda run -n wui python scripts/export_smps_total_concentration.py

Author: Nathan Lima
Date: 2025-01-08
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ====== CONFIGURATION VARIABLES ======
# Set this to 'MassConc' or 'NumConc'
CONCENTRATION_TYPE = "MassConc"  # Options: 'MassConc' or 'NumConc'

# Set output directory path
OUTPUT_PATH = "C:/Users/nml/Downloads/exported_data"

# ======================================


def load_config():
    """Load data_config.json to get SMPS data path"""
    repo_root = Path(__file__).parent.parent
    config_path = repo_root / "data_config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def read_transposed_smps_file(file_path, conc_type="MassConc"):
    """
    Read a raw SMPS file in transposed format and extract datetime and total concentration.

    Parameters:
    -----------
    file_path : Path or str
        Path to the SMPS Excel file
    conc_type : str
        'MassConc' or 'NumConc' - determines which Total Concentration row to read

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: datetime, Total Concentration, units
    """
    # Read the Excel file without headers
    df = pd.read_excel(file_path, header=None)

    # Find the row indices for Date, Start Time, and Total Concentration
    # First column contains all labels
    labels = df.iloc[:, 0].astype(str)

    # Find key rows
    date_row_idx = None
    time_row_idx = None
    total_conc_row_idx = None

    for idx, label in enumerate(labels):
        if label == "Date":
            date_row_idx = idx
        elif label == "Start Time":
            time_row_idx = idx
        elif "Total Concentration" in label:
            total_conc_row_idx = idx
            # Extract units from label (e.g., "Total Concentration(µg/m³)")
            if "(" in label and ")" in label:
                units = label[label.find("(") + 1 : label.find(")")]
            else:
                units = "unknown"

    if date_row_idx is None or time_row_idx is None or total_conc_row_idx is None:
        raise ValueError(f"Could not find required rows in {file_path}")

    # Extract data from columns 1 onwards (column 0 is labels)
    dates = df.iloc[date_row_idx, 1:].values
    times = df.iloc[time_row_idx, 1:].values
    concentrations = df.iloc[total_conc_row_idx, 1:].values

    # Combine date and time into datetime
    datetimes = []
    for date, time in zip(dates, times):
        try:
            if pd.notna(date) and pd.notna(time):
                dt_str = f"{date} {time}"
                dt = pd.to_datetime(dt_str)
                datetimes.append(dt)
            else:
                datetimes.append(pd.NaT)
        except:
            datetimes.append(pd.NaT)

    # Create DataFrame
    result_df = pd.DataFrame(
        {"datetime": datetimes, "Total Concentration": concentrations, "units": units}
    )

    # Remove rows with NaT datetime
    result_df = result_df.dropna(subset=["datetime"])

    return result_df


def process_all_smps_files(conc_type="MassConc", output_dir=None):
    """
    Process all SMPS files of the specified concentration type and export to CSV.

    Parameters:
    -----------
    conc_type : str
        'MassConc' or 'NumConc'
    output_dir : str or Path
        Directory to save the output CSV file
    """
    print(f"\n{'=' * 60}")
    print("SMPS Total Concentration Export")
    print(f"{'=' * 60}")
    print(f"Concentration Type: {conc_type}")

    # Load configuration
    config = load_config()
    smps_path = Path(config["instruments"]["smps"]["path"])

    print(f"SMPS Data Directory: {smps_path}")

    # Find all files matching the concentration type
    pattern = f"MH_apollo_bed_*_{conc_type}.xlsx"
    smps_files = sorted(smps_path.glob(pattern))

    print(f"\nFound {len(smps_files)} {conc_type} files:")
    for f in smps_files:
        print(f"  - {f.name}")

    if len(smps_files) == 0:
        print(f"\nERROR: No {conc_type} files found matching pattern: {pattern}")
        print(f"       in directory: {smps_path}")
        return

    # Process each file
    all_data = []
    units = None

    for file_path in smps_files:
        print(f"\nProcessing: {file_path.name}")
        try:
            df = read_transposed_smps_file(file_path, conc_type)
            all_data.append(df)

            # Get units from first file
            if units is None and len(df) > 0:
                units = df["units"].iloc[0]

            print(f"  ✓ Extracted {len(df)} data points")
            print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")

        except Exception as e:
            print(f"  ✗ Error processing {file_path.name}: {str(e)}")
            continue

    if len(all_data) == 0:
        print("\nERROR: No data was successfully processed")
        return

    # Combine all data
    print(f"\n{'=' * 60}")
    print("Combining data from all files...")
    combined_df = pd.concat(all_data, ignore_index=True)

    # Sort by datetime
    combined_df = combined_df.sort_values("datetime").reset_index(drop=True)

    # Remove duplicate timestamps (keep first occurrence)
    combined_df = combined_df.drop_duplicates(subset="datetime", keep="first")

    # Drop the units column (not needed in output)
    combined_df = combined_df.drop(columns=["units"])

    # Add units to column name
    if units:
        combined_df.columns = ["datetime", f"Total Concentration ({units})"]

    print(f"Total data points: {len(combined_df)}")
    print(
        f"Date range: {combined_df['datetime'].min()} to {combined_df['datetime'].max()}"
    )

    # Export to CSV
    if output_dir is None:
        output_dir = Path.cwd() / "output"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"SMPS_Total_{conc_type}.csv"
    output_path = output_dir / output_filename

    combined_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 60}")
    print("Export complete!")
    print(f"Output file: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.2f} KB")
    print(f"{'=' * 60}\n")

    # Display first few rows
    print("Preview of exported data:")
    print(combined_df.head(10).to_string(index=False))
    print("...")
    print(combined_df.tail(5).to_string(index=False))


if __name__ == "__main__":
    # Validate configuration
    if CONCENTRATION_TYPE not in ["MassConc", "NumConc"]:
        print(
            f"ERROR: CONCENTRATION_TYPE must be 'MassConc' or 'NumConc', got '{CONCENTRATION_TYPE}'"
        )
        sys.exit(1)

    # Process files
    process_all_smps_files(conc_type=CONCENTRATION_TYPE, output_dir=OUTPUT_PATH)
