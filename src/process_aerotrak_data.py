"""
AeroTrak Particle Counter Data Processing Script.

This module processes raw data exported from TSI AeroTrak particle counters
(e.g., Model 9306-V2). It reads Excel (.xlsx) or CSV export files, extracts
instrument metadata and cut point sizes from the header, then calculates
particle mass concentrations (µg/m³) and number concentrations (#/m³) for
each size bin.

The script also calculates standard PM metrics (PM1.0, PM2.5, PM10) when the
instrument's bin boundaries permit accurate aggregation. A PM metric is only
calculated if a bin upper boundary exists at exactly that cutoff size.

Mass concentration calculations assume spherical particles with unit density
(1 g/cm³), which is a common assumption for optical particle counters when
actual particle density is unknown.

Author: Nathan Lima
Date: 12/11/2025
"""

import os
from typing import TypedDict, cast

import numpy as np
import pandas as pd


# Type definitions for structured data
class BinInfo(TypedDict):
    """Type definition for particle size bin information."""

    lower: float
    upper: float
    channel: int


class ColumnInfo(TypedDict):
    """Type definition for concentration column metadata."""

    name: str
    lower: float
    upper: float


class AeroTrakMetadata(TypedDict):
    """Type definition for AeroTrak file metadata."""

    model: str | None
    serial: str | None
    calibration_date: str | None
    calibration_due: str | None
    flow_rate: float | None
    cut_points: list[float] | None
    header_row: int | None


def _extract_cut_points(row: pd.Series) -> list[float]:
    """
    Extract cut point sizes from a row of the header.

    Args:
        row: A pandas Series containing the cut point row data.

    Returns:
        List of cut point sizes in µm.
    """
    cut_points = []
    for val in row.iloc[1:]:
        if pd.notna(val):
            try:
                cut_points.append(float(val))
            except (ValueError, TypeError):
                break
        else:
            break
    return cut_points


def _parse_metadata_row(
    first_cell: str, row: pd.Series, metadata: AeroTrakMetadata
) -> bool:
    """
    Parse a single row from the header and update metadata dictionary.

    Args:
        first_cell: The content of the first cell (label column).
        row: The full row as a pandas Series.
        metadata: Dictionary to update with parsed values.

    Returns:
        True if this row is the column header row (Record #), False otherwise.
    """
    # Map of label prefixes to metadata keys
    label_map = {
        "Model": "model",
        "Serial": "serial",
        "Last Calibrated": "calibration_date",
        "Calibration Due": "calibration_due",
        "Flow": "flow_rate",
    }

    for prefix, key in label_map.items():
        if first_cell.startswith(prefix):
            metadata[key] = row.iloc[1]
            return False

    if "Cut Point" in first_cell:
        metadata["cut_points"] = _extract_cut_points(row)
        return False

    if first_cell == "Record #":
        return True

    return False


def parse_aerotrak_header(file_path: str) -> AeroTrakMetadata:
    """
    Parse the header section of an AeroTrak export file (Excel or CSV).

    AeroTrak files contain instrument metadata in the first ~10 rows before
    the actual measurement data begins. This function extracts key information
    including instrument model, serial number, calibration dates, flow rate,
    and critically, the cut point sizes that define particle size bins.

    Args:
        file_path: Path to the AeroTrak file (.xlsx or .csv).

    Returns:
        A dictionary containing:
            - model: Instrument model number
            - serial: Instrument serial number
            - calibration_date: Last calibration date
            - calibration_due: Next calibration due date
            - flow_rate: Sample flow rate in L/min
            - cut_points: List of cut point sizes in µm
            - header_row: Row index where column headers are located

    Raises:
        ValueError: If cut point sizes cannot be found in the file header.
    """
    # Determine file type and read the first 15 rows without header processing
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".csv":
        header_df = pd.read_csv(file_path, header=None, nrows=15)
    elif file_extension in [".xlsx", ".xls"]:
        header_df = pd.read_excel(file_path, sheet_name=0, header=None, nrows=15)
    else:
        raise ValueError(
            f"Unsupported file format: {file_extension}. "
            "Please provide a .xlsx or .csv file."
        )

    metadata: AeroTrakMetadata = {
        "model": None,
        "serial": None,
        "calibration_date": None,
        "calibration_due": None,
        "flow_rate": None,
        "cut_points": None,
        "header_row": None,
    }

    # Iterate through rows to find metadata fields
    for idx, row in header_df.iterrows():
        first_cell = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""

        is_header_row = _parse_metadata_row(first_cell, row, metadata)
        if is_header_row:
            metadata["header_row"] = cast(int, idx)
            break

    if metadata["cut_points"] is None:
        raise ValueError(
            "Could not find 'Cut Point Sizes' in file header. "
            "Ensure the file is a valid AeroTrak export."
        )

    return metadata


def calculate_geometric_mean(lower: float, upper: float) -> float:
    """
    Calculate the geometric mean of two values.

    For particle size distributions, the geometric mean provides a
    representative diameter for particles within a size bin. This is
    preferred over arithmetic mean because particle size distributions
    are typically log-normal.

    Args:
        lower: Lower bound of the size range (µm).
        upper: Upper bound of the size range (µm).

    Returns:
        Geometric mean of the two values (µm).
    """
    return np.exp((np.log(lower) + np.log(upper)) / 2)


def calculate_particle_mass(diameter_um: float, density_g_cm3: float = 1.0) -> float:
    """
    Calculate the mass of a spherical particle.

    Assumes particles are spherical and uses the provided density to
    calculate mass from diameter.

    Args:
        diameter_um: Particle diameter in micrometers (µm).
        density_g_cm3: Particle density in g/cm³. Default is 1.0 (water/standard).

    Returns:
        Particle mass in micrograms (µg).
    """
    # Convert diameter from µm to cm
    diameter_cm = diameter_um * 1e-4
    radius_cm = diameter_cm / 2

    # Calculate volume of sphere in cm³
    volume_cm3 = (4 / 3) * np.pi * (radius_cm**3)

    # Calculate mass in grams, then convert to micrograms
    mass_g = volume_cm3 * density_g_cm3
    mass_ug = mass_g * 1e6

    return mass_ug


def _build_bin_definitions(
    cut_points: list[float], final_bin_upper: float
) -> list[BinInfo]:
    """
    Build bin definitions from cut points.

    Args:
        cut_points: List of cut point sizes in µm.
        final_bin_upper: Upper bound for the largest size bin in µm.

    Returns:
        List of bin information dictionaries.
    """
    bins: list[BinInfo] = []
    for i, lower in enumerate(cut_points):
        if i < len(cut_points) - 1:
            upper = cut_points[i + 1]
        else:
            upper = final_bin_upper
        bins.append({"lower": lower, "upper": upper, "channel": i + 1})
    return bins


def _process_single_bin(
    bin_info: BinInfo,
    data_df: pd.DataFrame,
    volume_m3: pd.Series,
    particle_density: float,
    results_df: pd.DataFrame,
) -> tuple[ColumnInfo | None, ColumnInfo | None]:
    """
    Process a single size bin and add columns to results DataFrame.

    Args:
        bin_info: Dictionary with bin lower/upper bounds and channel number.
        data_df: Source DataFrame with raw particle counts.
        volume_m3: Sample volume in cubic meters.
        particle_density: Assumed particle density in g/cm³.
        results_df: DataFrame to add calculated columns to (modified in place).

    Returns:
        Tuple of (mass_col_info, number_col_info) or (None, None) if column missing.
    """
    lower = bin_info["lower"]
    upper = bin_info["upper"]
    channel = bin_info["channel"]

    # Find the differential count column for this channel
    diff_col = f"Ch{channel} Diff (#)"

    if diff_col not in data_df.columns:
        print(f"Warning: Column '{diff_col}' not found, skipping channel {channel}")
        return None, None

    # Get particle counts for this bin
    particle_counts = data_df[diff_col]

    # Calculate representative particle diameter using geometric mean
    rep_diameter = calculate_geometric_mean(lower, upper)

    # Calculate mass per particle (µg)
    mass_per_particle = calculate_particle_mass(rep_diameter, particle_density)

    # Calculate number concentration (#/m³)
    number_col_name = f"PM{lower}-{upper} (#/m³)"
    results_df[number_col_name] = particle_counts / volume_m3

    # Calculate mass concentration (µg/m³)
    mass_col_name = f"PM{lower}-{upper} (µg/m³)"
    results_df[mass_col_name] = (particle_counts / volume_m3) * mass_per_particle

    mass_info: ColumnInfo = {"name": mass_col_name, "lower": lower, "upper": upper}
    number_info: ColumnInfo = {"name": number_col_name, "lower": lower, "upper": upper}

    return mass_info, number_info


def _get_bin_upper_boundaries(mass_cols: list[ColumnInfo]) -> set[float]:
    """
    Get the set of all bin upper boundaries.

    Args:
        mass_cols: List of mass concentration column metadata.

    Returns:
        Set of upper boundary values in µm.
    """
    return {col_info["upper"] for col_info in mass_cols}


def _calculate_pm_metrics(
    results_df: pd.DataFrame,
    mass_cols: list[ColumnInfo],
    number_cols: list[ColumnInfo],
) -> None:
    """
    Calculate aggregate PM metrics (PM1.0, PM2.5, PM10).

    Adds columns for both mass and number concentrations for each PM metric
    ONLY when the bin boundaries permit accurate calculation. A PM metric
    requires a bin upper boundary at exactly that cutoff size.

    For example:
        - PM1.0 requires a bin ending at 1.0 µm
        - PM2.5 requires a bin ending at 2.5 µm
        - PM10 requires a bin ending at 10.0 µm

    If no such bin boundary exists, the PM metric is skipped entirely.

    Args:
        results_df: DataFrame to add PM metric columns to (modified in place).
        mass_cols: List of mass concentration column metadata.
        number_cols: List of number concentration column metadata.
    """
    # Get all bin upper boundaries to check if PM metrics can be calculated
    bin_boundaries = _get_bin_upper_boundaries(mass_cols)

    pm_cutoffs = [
        {"name": "PM1.0", "cutoff": 1.0},
        {"name": "PM2.5", "cutoff": 2.5},
        {"name": "PM10", "cutoff": 10.0},
    ]

    print("\nCalculating aggregate PM metrics:")

    for pm in pm_cutoffs:
        cutoff = pm["cutoff"]
        pm_name = pm["name"]

        # Check if a bin boundary exists at this cutoff
        # A PM metric can only be accurately calculated if there's a bin
        # whose upper bound equals the cutoff
        if cutoff not in bin_boundaries:
            print(
                f"  {pm_name}: SKIPPED - no bin boundary at {cutoff} µm. "
                f"Available boundaries: {sorted(bin_boundaries)}"
            )
            continue

        # Find all bins whose upper bound is at or below the cutoff
        contributing_mass_cols = [
            col_info["name"] for col_info in mass_cols if col_info["upper"] <= cutoff
        ]

        # Calculate mass concentration for this PM metric
        if contributing_mass_cols:
            results_df[f"{pm_name} (µg/m³)"] = results_df[contributing_mass_cols].sum(
                axis=1
            )
            print(f"  {pm_name}: Calculated from bins {contributing_mass_cols}")

        # Calculate number concentration for this PM metric
        contributing_num_cols = [
            col_info["name"] for col_info in number_cols if col_info["upper"] <= cutoff
        ]

        if contributing_num_cols:
            results_df[f"{pm_name} (#/m³)"] = results_df[contributing_num_cols].sum(
                axis=1
            )


def process_aerotrak_data(
    input_file_path: str,
    output_file_path: str,
    particle_density: float = 1.0,
    final_bin_upper: float = 25.0,
) -> pd.DataFrame:
    """
    Process AeroTrak particle counter data and calculate mass/number concentrations.

    This function reads raw AeroTrak export files (Excel or CSV), extracts the
    cut point sizes from the header, and calculates both mass concentration (µg/m³)
    and number concentration (#/m³) for each size bin. It also calculates aggregate
    PM metrics (PM1.0, PM2.5, PM10) when bin boundaries allow.

    The mass calculation assumes spherical particles with the specified density.
    This is a standard assumption for OPC (Optical Particle Counter) data when
    actual particle composition is unknown.

    Args:
        input_file_path: Path to the input AeroTrak file (.xlsx or .csv).
        output_file_path: Path for the output file with calculated concentrations.
        particle_density: Assumed particle density in g/cm³. Default is 1.0.
        final_bin_upper: Upper bound for the largest size bin in µm. Default is 25.0.

    Returns:
        DataFrame containing the processed data with all calculated columns.

    Raises:
        ValueError: If required columns (Volume, Diff counts) are not found in the data.
    """
    # Parse header to get metadata and cut point sizes
    metadata = parse_aerotrak_header(input_file_path)
    cut_points = metadata["cut_points"]
    if cut_points is None:
        raise ValueError("cut_points should not be None after parsing header")

    # Print instrument information
    print(f"Instrument Model: {metadata['model']}")
    print(f"Serial Number: {metadata['serial']}")
    print(f"Flow Rate: {metadata['flow_rate']} L/min")
    print(f"Cut Point Sizes (µm): {cut_points}")

    # Determine file type and read the data section using the identified header row
    file_extension = os.path.splitext(input_file_path)[1].lower()

    if file_extension == ".csv":
        data_df = pd.read_csv(input_file_path, header=metadata["header_row"])
    elif file_extension in [".xlsx", ".xls"]:
        data_df = pd.read_excel(
            input_file_path, sheet_name=0, header=metadata["header_row"]
        )
    else:
        raise ValueError(
            f"Unsupported file format: {file_extension}. "
            "Please provide a .xlsx or .csv file."
        )

    data_df.columns = data_df.columns.str.strip()

    # Verify required volume column exists and convert to m³
    volume_col = "Volume (L)"
    if volume_col not in data_df.columns:
        raise ValueError(f"Required column '{volume_col}' not found in data.")
    volume_m3 = data_df[volume_col] * 1e-3  # 1 L = 1e-3 m³

    # Build bin definitions and print them
    bins = _build_bin_definitions(cut_points, final_bin_upper)
    print("\nSize bins defined:")
    for bin_def in bins:
        print(
            f"  Channel {bin_def['channel']}: {bin_def['lower']}-{bin_def['upper']} µm"
        )

    # Initialize results DataFrame with timestamp if available
    results_df = pd.DataFrame()
    timestamp_col = "Date and Time"
    if timestamp_col in data_df.columns:
        results_df[timestamp_col] = data_df[timestamp_col]

    # Process each size bin
    mass_cols: list[ColumnInfo] = []
    number_cols: list[ColumnInfo] = []

    for bin_info in bins:
        mass_info, number_info = _process_single_bin(
            bin_info, data_df, volume_m3, particle_density, results_df
        )
        if mass_info is not None:
            mass_cols.append(mass_info)
        if number_info is not None:
            number_cols.append(number_info)

    # Calculate aggregate PM metrics (only where bin boundaries allow)
    _calculate_pm_metrics(results_df, mass_cols, number_cols)

    # Save results based on output file extension
    output_extension = os.path.splitext(output_file_path)[1].lower()

    if output_extension == ".csv":
        results_df.to_csv(output_file_path, index=False)
    elif output_extension in [".xlsx", ".xls"]:
        results_df.to_excel(output_file_path, index=False)
    else:
        # Default to Excel if extension is unclear
        results_df.to_excel(output_file_path, index=False)

    print(f"\nResults saved to: {output_file_path}")

    return results_df


def main() -> None:
    """
    Main entry point for the AeroTrak data processing script.

    Provides a graphical file selection interface using tkinter dialogs.
    Prompts the user to select an input AeroTrak data file (Excel or CSV)
    and specify an output location for the processed data.
    """
    # pylint: disable=import-outside-toplevel
    # Tkinter imported here to allow module use in non-GUI environments
    import tkinter as tk
    from tkinter import filedialog

    # Print welcome message and instructions
    print("=" * 70)
    print("AeroTrak Particle Counter Data Processing Script")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Read raw AeroTrak particle counter data (Excel or CSV format)")
    print("  2. Extract instrument metadata and particle size bin information")
    print("  3. Calculate mass concentration (µg/m³) for each size bin")
    print("  4. Calculate number concentration (#/m³) for each size bin")
    print("  5. Calculate PM1.0, PM2.5, and PM10 metrics (when applicable)")
    print("  6. Save the processed data to a new file")
    print("\nNOTE: Mass calculations assume spherical particles with unit density")
    print("      (1.0 g/cm³), which is standard for optical particle counters.")
    print("\n" + "=" * 70)
    print("Please select your input and output files in the dialogs that follow.")
    print("=" * 70 + "\n")

    # Initialize tkinter root window (hidden)
    root = tk.Tk()
    root.withdraw()

    # Prompt user to select input file
    print("Opening file browser - Please select your AeroTrak data file...")
    input_file_path = filedialog.askopenfilename(
        title="Select AeroTrak Data File (Excel or CSV)",
        filetypes=[
            ("Data Files", "*.xlsx;*.xls;*.csv"),
            ("Excel Files", "*.xlsx;*.xls"),
            ("CSV Files", "*.csv"),
            ("All Files", "*.*"),
        ],
    )

    if not input_file_path:
        print("\nNo input file selected. Exiting.")
        return

    print(f"\nInput file selected: {input_file_path}")

    # Prompt user to select output location
    print("\nOpening file browser - Please select where to save the processed data...")
    output_file_path = filedialog.asksaveasfilename(
        title="Save Processed Data As",
        defaultextension=".xlsx",
        filetypes=[
            ("Excel Files", "*.xlsx"),
            ("CSV Files", "*.csv"),
            ("All Files", "*.*"),
        ],
    )

    if not output_file_path:
        print("\nNo output location selected. Exiting.")
        return

    print(f"Output file will be saved as: {output_file_path}")
    print("\n" + "=" * 70)
    print("Processing data...")
    print("=" * 70 + "\n")

    # Process the data
    try:
        process_aerotrak_data(input_file_path, output_file_path)
        print("\n" + "=" * 70)
        print("Processing complete!")
        print("=" * 70)
        print(f"\nYour processed data has been saved to:\n  {output_file_path}")
        print("\nThe output file contains:")
        print("  - Mass concentrations (µg/m³) for each particle size bin")
        print("  - Number concentrations (#/m³) for each particle size bin")
        print("  - PM metrics (PM1.0, PM2.5, PM10) where bin boundaries allow")
        print("\n" + "=" * 70 + "\n")
    except Exception as err:
        print("\n" + "=" * 70)
        print("ERROR: Processing failed!")
        print("=" * 70)
        print(f"\nError message: {err}")
        print("\nPlease check that:")
        print("  - The input file is a valid AeroTrak export (Excel or CSV)")
        print("  - The file contains the expected header structure")
        print("  - The file includes 'Cut Point Sizes' information")
        print("  - The file has the required data columns (Volume, Ch1 Diff, etc.)")
        print("\n" + "=" * 70 + "\n")
        raise


if __name__ == "__main__":
    main()
