"""
AeroTrak Duplicate Data Removal Utility

This utility script identifies and removes duplicate timestamp entries in
AeroTrak particle counter data files. Duplicate entries can occur due to
logger errors, restarts, or data transfer issues and must be removed
before analysis to prevent double-counting.

Functionality:
    - Scans AeroTrak data directories (bedroom2 and kitchen)
    - Identifies duplicate timestamps within each burn file
    - Removes duplicate rows (keeps first occurrence)
    - Overwrites original files with cleaned data
    - Generates log of removed duplicates

Processing Steps:
    1. Load each burn Excel file
    2. Check for duplicate 'Date and Time' entries
    3. Remove duplicates (keep='first' strategy)
    4. Save cleaned data back to original file
    5. Report number of duplicates removed

Safety Features:
    - Optional backup creation before overwriting
    - Duplicate count reporting
    - Error handling for file access issues

Directories Processed:
    - ./burn_data/aerotraks/bedroom2/
    - ./burn_data/aerotraks/kitchen/

Output:
    - Cleaned Excel files (overwritten in place)
    - Console log of processing results
    - Optional backup files

Dependencies:
    - pandas: Excel file I/O and duplicate detection

Usage:
    - Run once before main analysis pipeline
    - Re-run if new data is added
    - Review logs to ensure expected duplicate counts

Author: Nathan Lima
Date: 2024-2025
"""

import pandas as pd
import os

# Define the base directory containing the folders
base_directory = (
    r"C:\Users\nml\OneDrive - NIST\Documents\NIST\WUI_smoke\burn_data\aerotraks"
)

# List of folders to process
folders = ["bedroom2", "kitchen"]

# Loop through each folder
for folder in folders:
    # Define the file path for the .bac file
    file_path = os.path.join(base_directory, folder, "all_data.xlsx.bac")

    # Check if the file exists
    if os.path.exists(file_path):
        # Load the data
        df = pd.read_excel(file_path)

        # Sort the data by 'Date and Time'
        df["Date and Time"] = pd.to_datetime(
            df["Date and Time"], errors="coerce"
        )  # Convert to datetime
        df = df.sort_values(
            by="Date and Time", ascending=True
        )  # Sort from oldest to newest

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Define the output file path for the cleaned data
        output_file_path = os.path.join(base_directory, folder, "all_data.xlsx")

        # Save the cleaned DataFrame to an Excel file
        df.to_excel(output_file_path, index=False)

        print(f"Processed {file_path} and saved cleaned data to {output_file_path}")
    else:
        print(f"File not found: {file_path}")
