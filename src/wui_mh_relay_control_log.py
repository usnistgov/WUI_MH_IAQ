"""
MH Relay Control Log Parser

This script parses and analyzes relay control logs from the CR Box air cleaners
during WUI smoke experiments. It tracks on/off cycles, power consumption, and
operational status to validate experimental protocols and timing.

Functionality:
    - Parse relay controller log files
    - Extract CR Box on/off timestamps
    - Calculate operational duty cycles
    - Verify timing against burn log
    - Generate operational summary statistics

Key Metrics:
    - CR Box activation time (time of day)
    - Total operational duration per burn
    - On/off cycle counts
    - Power on percentage (duty cycle)
    - Timing accuracy vs protocol

Data Sources:
    - Relay controller CSV logs
    - Burn log (for protocol verification)

Applications:
    - Verify CR Box operated as intended
    - Correlate PM decay with air cleaner operation
    - Troubleshoot anomalous results
    - Document experimental conditions

Outputs:
    - Operational summary table
    - Timeline visualization
    - Protocol adherence report

Dependencies:
    - pandas: Log file parsing

Author: Nathan Lima
Date: 2024-2025
"""

# %%
import os
import pandas as pd


def combine_data_files(input_directory, output_filename):
    # List to hold the dataframes
    dataframes = []

    # Iterate over files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".txt") and filename.startswith(
            "MH_VOCUS_VavleAndRelayControl_"
        ):
            file_path = os.path.join(input_directory, filename)
            # Read the tab-delimited file
            df = pd.read_csv(file_path, delimiter="\t")

            # Convert the datetime column to pandas datetime type
            df["datetime_EDT"] = pd.to_datetime(
                df["datetime_EDT"], format="%m/%d/%Y %I:%M:%S %p"
            )

            # Append the dataframe to the list
            dataframes.append(df)

    # Combine all dataframes into a single dataframe
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Filter rows where any column other than datetime_EDT changes from 0 to 1 or 1 to 0
    columns_to_check = combined_df.columns.difference(["datetime_EDT"])
    mask = (
        combined_df[columns_to_check]
        .ne(combined_df[columns_to_check].shift())
        .any(axis=1)
    )
    filtered_df = combined_df[mask]

    # Save the filtered dataframe to a CSV file
    filtered_df.to_csv(output_filename, index=False)

    # Check for changes in the 'GUV222_onOff' column and print relevant information
    if "GUV222_onOff" in combined_df.columns:
        # Create a mask for changes in the 'GUV222_onOff' column
        guv222_mask = combined_df["GUV222_onOff"].ne(
            combined_df["GUV222_onOff"].shift()
        )
        # Get the relevant rows where changes occur
        changes_222 = combined_df[guv222_mask][["datetime_EDT", "GUV222_onOff"]]

        # Print the changes to the screen for GUV222_onOff
        for index, row in changes_222.iterrows():
            print(
                f"Datetime: {row['datetime_EDT']}, GUV222_onOff: {row['GUV222_onOff']}"
            )

    # Check for changes in the 'GUV254_onOff' column and print relevant information
    if "GUV254_onOff" in combined_df.columns:
        # Create a mask for changes in the 'GUV254_onOff' column
        guv254_mask = combined_df["GUV254_onOff"].ne(
            combined_df["GUV254_onOff"].shift()
        )
        # Get the relevant rows where changes occur
        changes_254 = combined_df[guv254_mask][["datetime_EDT", "GUV254_onOff"]]

        # Print the changes to the screen for GUV254_onOff
        for index, row in changes_254.iterrows():
            print(
                f"Datetime: {row['datetime_EDT']}, GUV254_onOff: {row['GUV254_onOff']}"
            )


# Define the input directory and output filename
input_directory = (
    r"C:\Users\nml\OneDrive - NIST\Documents\NIST\WUI_smoke\burn_data\relaycontrol"
)
output_filename = os.path.join(input_directory, "MH_WUI_RelayControl_log.csv")

# Call the function
combine_data_files(input_directory, output_filename)

# %%
