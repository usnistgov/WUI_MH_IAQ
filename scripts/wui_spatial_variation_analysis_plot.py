import pandas as pd
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource
import numpy as np
from scipy.interpolate import interp1d, make_interp_spline
import os
import sys

# Import utils
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(os.path.join(grandparent_dir, "general_utils", "scripts"))
from metadata_utils import get_script_metadata

# Read Excel file
excel_file = r"C:\Users\nml\OneDrive - NIST\Documents\NIST\WUI_smoke\burn_data\spatial_variation_analysis.xlsx"
aerotrak_df = pd.read_excel(excel_file, sheet_name="AeroTrak")
quantaq_df = pd.read_excel(excel_file, sheet_name="QuantAQ")

# Filter data for required Burn IDs
burn_ids = ["burn2", "burn4", "burn9"]
aerotrak_df = aerotrak_df[aerotrak_df["Burn_ID"].isin(burn_ids)]
quantaq_df = quantaq_df[quantaq_df["Burn_ID"].isin(burn_ids)]

# Define PM sizes
pm_sizes = ["PM1 (µg/m³)", "PM2.5 (µg/m³)", "PM10 (µg/m³)"]
aerotrak_pm_size = "PM3 (µg/m³)"

# Create figures directory if it doesn't exist
output_dir = r"C:\Users\nml\OneDrive - NIST\Documents\NIST\WUI_smoke\Paper_figures"
os.makedirs(output_dir, exist_ok=True)

# Create figures
for pm_size in pm_sizes:
    p = figure(
        title=pm_size,
        x_axis_label="Number of CR Boxes",
        y_axis_label="Ratio",
        y_range=(0, 1.5),
    )

    # Filter data for current PM size
    if pm_size == "PM2.5 (µg/m³)":
        aerotrak_data = aerotrak_df[aerotrak_df["PM_Size"] == aerotrak_pm_size]
    else:
        aerotrak_data = aerotrak_df[aerotrak_df["PM_Size"] == pm_size]
    quantaq_data = quantaq_df[quantaq_df["PM_Size"] == pm_size]

    # Plot data
    for device_data, device_name in [
        (aerotrak_data, "AeroTrak"),
        (quantaq_data, "QuantAQ"),
    ]:
        for ratio in ["Peak_Ratio_Index", "CRBox_Activation_Ratio", "Average_Ratio"]:
            x = device_data["Burn_ID"].map({"burn2": 4, "burn4": 1, "burn9": 2})
            y = device_data[ratio]
            p.scatter(x, y, legend_label=f"{device_name} {ratio}")

            # Add fit
            x_fit = np.linspace(x.min(), x.max(), 100)
            try:
                # Try spline fit
                f = make_interp_spline(x, y, k=3)
                y_fit = f(x_fit)
            except ValueError:
                # Fallback to quadratic fit if spline fails
                f = interp1d(x, y, kind="quadratic")
                y_fit = f(x_fit)
            p.line(x_fit, y_fit, legend_label=f"{device_name} {ratio} fit")

    # Customize x-axis
    p.xaxis.ticker = [1, 2, 3, 4]
    p.xaxis.major_label_overrides = {1: "1", 2: "2", 3: "", 4: "4"}

    # Save figure
    output_file(os.path.join(output_dir, f"{pm_size}.html"))
    save(p)

    # Add metadata to figure
    metadata = get_script_metadata(__file__)
    with open(os.path.join(output_dir, f"{pm_size}.html"), "a") as f:
        f.write(f"<!-- {metadata} -->")
