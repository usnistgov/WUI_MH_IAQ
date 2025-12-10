import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np


def process_aerotrak_data(input_file_path, output_file_path):
    # Load the AeroTrak data from the Excel file
    aerotrak_data = pd.read_excel(input_file_path)
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
    else:
        raise ValueError("Volume column not found in the data")

    def g_mean(x):
        a = np.log(x)
        return np.exp(a.mean())

    # Initialize new columns for mass concentration and calculate values
    pm_columns = []
    if "volume_cm" in locals():
        for i, channel in enumerate(size_channels):
            if channel in size_values:
                next_channel = (
                    size_channels[i + 1] if i < len(size_channels) - 1 else None
                )
                next_size_value = size_values.get(next_channel, 25)

                particle_size = g_mean([size_values[channel], next_size_value])
                particle_size_m = particle_size * 1e-6

                diff_col = f"{channel} Diff (#)"
                if diff_col in aerotrak_data.columns:
                    particle_counts = aerotrak_data[diff_col]

                    radius_m = particle_size_m / 2
                    volume_per_particle = (4 / 3) * np.pi * (radius_m**3)
                    particle_mass = (
                        volume_per_particle * 1e6 * 1e6
                    )  # density assumed as 1 g/cm³

                    new_diff_col_µg_m3 = (
                        f"PM{size_values[channel]}-{next_size_value} Diff (µg/m³)"
                    )
                    aerotrak_data[new_diff_col_µg_m3] = (
                        particle_counts / (volume_cm * 1e-6)
                    ) * particle_mass
                    pm_columns.append(new_diff_col_µg_m3)

                    # Calculate #/m³
                    new_diff_col_m3 = (
                        f"PM{size_values[channel]}-{next_size_value} Diff (#/m³)"
                    )
                    aerotrak_data[new_diff_col_m3] = particle_counts / (
                        volume_cm * 1e-6
                    )
                    pm_columns.append(new_diff_col_m3)

        # Save the results to a new Excel file
        pm05_to_pm25_columns = [
            col
            for col in pm_columns
            if "PM0.5" in col
            or "PM1-" in col
            or "PM1.0-" in col
            or "PM2.5-" in col
            or "PM3-" in col
            or "PM5-" in col
            or "PM10-" in col
        ]
        results_df = aerotrak_data[pm05_to_pm25_columns]
        results_df.to_excel(output_file_path, index=False)
    else:
        print("Error: 'volume_cm' is not defined.")


def main():
    # Prompt the user to select an Aerotrak file
    root = tk.Tk()
    root.withdraw()
    input_file_path = filedialog.askopenfilename(
        title="Select Aerotrak File", filetypes=[("Excel Files", "*.xlsx")]
    )

    if input_file_path:
        # Prompt the user to select a save location
        output_file_path = filedialog.asksaveasfilename(
            title="Save Processed Data",
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx")],
        )
        if output_file_path:
            process_aerotrak_data(input_file_path, output_file_path)
        else:
            print("No save location selected.")
    else:
        print("No file selected.")


if __name__ == "__main__":
    main()
