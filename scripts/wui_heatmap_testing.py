# -*- coding: utf-8 -*-
"""
Heatmap Visualization Testing and Development

This script serves as a development environment for creating and refining
heatmap visualizations of particle size distributions and concentration fields.
It prototypes different heatmap approaches before implementation in production
analysis scripts.

Testing Focus Areas:
    - Color scale selection (linear, log, custom)
    - Interpolation methods for sparse data
    - Axis formatting and labeling
    - Colorbar customization
    - Performance with large datasets

Heatmap Types Prototyped:
    1. SMPS size-time heatmaps
    2. Spatial concentration heat maps
    3. Multi-burn comparison heatmaps
    4. Correlation matrix visualizations

Visualization Libraries Tested:
    - Bokeh: Interactive web-based heatmaps
    - Matplotlib: Publication-quality static heatmaps
    - Plotly: Alternative interactive approach
    - Seaborn: Statistical heatmaps

Development Process:
    1. Load sample or test data
    2. Experiment with different visualization parameters
    3. Assess readability and clarity
    4. Optimize performance
    5. Export examples for review
    6. Implement successful approaches in production

Dependencies:
    - pandas: Data preparation
    - numpy: Array operations
    - bokeh: Interactive heatmaps
    - matplotlib: Static heatmaps

Note:
    - Experimental code not intended for production
    - May contain uncommented or temporary code
    - Successfully tested approaches moved to production scripts

Author: Nathan Lima
Date: 2024-2025
"""

# %% RUN
# import needed modules
print("Importing needed modules")
import os
import numpy as np
import pandas as pd
from datetime import datetime
from bokeh.plotting import figure, output_file, show, output_notebook
from bokeh.models.tools import HoverTool, CrosshairTool, Span
from bokeh.layouts import gridplot
from bokeh.models import CrosshairTool, Span

# %% RUN User defines directory path for datset, dataset used, and dataset final location
# User set absolute_path
absolute_path = "C:/Users/nml/OneDrive - NIST/Documents/NIST/WUI_smoke/burn_data/"  # USER ENTERED PROJECT PATH
os.chdir(absolute_path)

# use only one dataset at a time
dataset = "smps"
print("dataset selected: " + dataset)

# Set up Datasets that are used
print(dataset + " dataset selected and final file path defined")
if dataset == "smps":
    df = pd.read_excel(
        "./smps/MH_apollo_bed_04252024_MassConc.xlsx", sheet_name="all_data"
    )  # USER ENTERED FINAL PATH FOR RECS

elif dataset == "ahs21":
    df = pd.read_csv("./ahs_data/2021/household.csv")  # USER ENTERED FINAL PATH FOR AHS

# %% Example 1
from math import pi

import pandas as pd

from bokeh.models import BasicTicker, PrintfTickFormatter
from bokeh.plotting import figure, show
from bokeh.sampledata.unemployment1948 import data
from bokeh.transform import linear_cmap

data["Year"] = data["Year"].astype(str)
data = data.set_index("Year")
data.drop("Annual", axis=1, inplace=True)
data.columns.name = "Month"

years = list(data.index)
months = list(reversed(data.columns))

# reshape to 1D array or rates with a month and year for each row.
df = pd.DataFrame(data.stack(), columns=["rate"]).reset_index()

# this is the colormap from the original NYTimes plot
colors = [
    "#75968f",
    "#a5bab7",
    "#c9d9d3",
    "#e2e2e2",
    "#dfccce",
    "#ddb7b1",
    "#cc7878",
    "#933b41",
    "#550b1d",
]

TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

p = figure(
    title=f"US Unemployment ({years[0]} - {years[-1]})",
    x_range=years,
    y_range=months,
    x_axis_location="above",
    width=900,
    height=400,
    tools=TOOLS,
    toolbar_location="below",
    tooltips=[("date", "@Month @Year"), ("rate", "@rate%")],
)

p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_text_font_size = "7px"
p.axis.major_label_standoff = 0
p.xaxis.major_label_orientation = pi / 3

r = p.rect(
    x="Year",
    y="Month",
    width=1,
    height=1,
    source=df,
    fill_color=linear_cmap("rate", colors, low=df.rate.min(), high=df.rate.max()),
    line_color=None,
)

p.add_layout(
    r.construct_color_bar(
        major_label_text_font_size="7px",
        ticker=BasicTicker(desired_num_ticks=len(colors)),
        formatter=PrintfTickFormatter(format="%d%%"),
        label_standoff=6,
        border_line_color=None,
        padding=5,
    ),
    "right",
)

show(p)


# %% Example 2
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, ColorBar
from bokeh.transform import linear_cmap
from bokeh.io import output_notebook
import pandas as pd
import numpy as np
from bokeh.palettes import Viridis256

# Example data (replace this with your actual data)
# Creating a sample DataFrame with random data
np.random.seed(0)
time_points = pd.date_range("2024-01-01", periods=10)
size_bins = ["Size1", "Size2", "Size3", "Size4"]
data = (
    np.random.rand(len(time_points), len(size_bins)) * 100
)  # Random data for illustration
df = pd.DataFrame(data, columns=size_bins, index=time_points)

# Prepare data for Bokeh
df_stack = df.stack().reset_index()
df_stack.columns = ["Time", "Size", "Value"]

# Bokeh plot setup
output_notebook()  # Output to notebook (use output_file() for standalone HTML)
p = figure(
    x_axis_label="Size Bins",
    y_axis_label="Time",
    x_range=size_bins,
    y_range=list(df.index.astype(str)),
)

# Create a color mapper
mapper = linear_cmap(
    field_name="Value",
    palette=Viridis256,
    low=df_stack["Value"].min(),
    high=df_stack["Value"].max(),
)

# Plotting the rectangles (heatmap)
p.rect(
    x="Size",
    y="Time",
    width=1,
    height=1,
    source=ColumnDataSource(df_stack),
    fill_color=mapper,
    line_color=None,
)

# Add color bar
color_bar = ColorBar(color_mapper=mapper["transform"], width=8, location=(0, 0))
p.add_layout(color_bar, "right")

# Configure plot aesthetics
p.title.text = "Size-Resolved Time Series Heatmap"
p.title.align = "center"
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_text_font_size = "10px"
p.axis.major_label_standoff = 0
p.xaxis.major_label_orientation = np.pi / 3

show(p)

# %%
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import matplotlib.dates as mdates  # Import mdates for time-related locators

# %% RUN User defines directory path for dataset, dataset used, and dataset final location
# User set absolute_path
absolute_path = "C:/Users/nml/OneDrive - NIST/Documents/NIST/WUI_smoke/"  # USER ENTERED PROJECT PATH
os.chdir(absolute_path)

# Read the Excel file, specifying the sheet name
data = pd.read_excel(
    "./burn_data/smps/MH_apollo_bed_05312024_MassConc.xlsx",
    sheet_name="all_data",  # specify the sheet name
    parse_dates=[0],
    index_col=[0],
)

# Inspect column names
print("Column names:", data.columns)


# Try to convert column names to floats
def try_parse_float(value):
    try:
        return float(value)
    except ValueError:
        return np.nan  # or handle the non-numeric case as needed


wavelength = [try_parse_float(v) for v in data.columns]

# Check for any NaNs or non-numeric values in the wavelength list
print("Wavelength values:", wavelength)
if any(np.isnan(wavelength)):
    print("Warning: Some wavelengths could not be converted to float.")

# Filter data for the specific day (2024-05-31)
data_filtered = data.loc["2024-05-31"]

# Drop columns with all NaN values
data_filtered = data_filtered.dropna(axis=1, how="all")

# Ensure that the data is only for the day 2024-05-31
time = data_filtered.index

# Handle cases where the wavelength list might be empty or contain NaNs
wavelength = [w for w in wavelength if not np.isnan(w)]

# Create the DataArray
da = xr.DataArray(
    data=data_filtered.values,  # use .values to get the underlying numpy array
    dims=["time", "wavelength"],
    coords={"time": time, "wavelength": wavelength},
)

# Convert time to hours since start of the day
time_hours = (time - time.normalize()).total_seconds() / 3600

# Create a custom colormap that goes from purple to green to red
cmap = mcolors.LinearSegmentedColormap.from_list(
    "purple_green_red", ["purple", "green", "red"]
)

# %%
# Plot using matplotlib
plt.figure(figsize=(12, 6))

# Ensure valid data for the log scale
data_for_plot = np.log10(da.T.where(da.T > 0))  # Apply log scale safely

# Plot with the custom colormap
pc = plt.pcolormesh(time_hours, wavelength, data_for_plot, shading="auto", cmap=cmap)

# Add a color bar with custom label
cbar = plt.colorbar(pc, label="dNdlogDP (cm$^{-3}$)")

# Format the color bar ticks in scientific notation
formatter = ticker.ScalarFormatter()
formatter.set_powerlimits(
    (-3, 4)
)  # Use scientific notation for values outside this range
cbar.formatter = formatter
cbar.update_ticks()

# Set x-axis ticks every 6 hours
plt.xticks(np.arange(0, time_hours.max() + 1, 6))

# Add minor ticks at a more reasonable interval, e.g., every 1 hour
plt.gca().xaxis.set_minor_locator(mdates.MinuteLocator(interval=1440))
plt.gca().xaxis.set_minor_formatter(ticker.NullFormatter())  # Hide minor tick labels

# Set y-axis to log scale but show 10 and 100 as primary ticks
plt.yscale("log")
plt.ylim(9, 420)  # Set y-axis limits

# Major ticks (primary) and labels
plt.yticks([10, 100], ["10", "100"])  # Custom y-axis ticks and labels

# Minor ticks (secondary) and labels
plt.gca().yaxis.set_minor_locator(ticker.FixedLocator([20, 40, 60, 80, 200, 400]))
plt.gca().yaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

# Customize tick parameters for major and minor ticks
plt.gca().tick_params(
    axis="y", which="major", labelsize=12, width=2
)  # Major ticks: larger text, thicker lines
plt.gca().tick_params(
    axis="y", which="minor", labelsize=8, width=1
)  # Minor ticks: smaller text, thinner lines

plt.gca().tick_params(
    axis="x", which="major", labelsize=12, width=2
)  # Major ticks: larger text, thicker lines
plt.gca().tick_params(
    axis="x", which="minor", labelsize=8, width=1
)  # Minor ticks: smaller text, thinner lines

# Update axis labels
plt.xlabel("Hours of the day")
plt.ylabel("Diameter (nm)")
plt.title("SMPS Mass Concentration of Particles for Burn 10")

# Show the plot
plt.show()

# %%
import pandas as pd
import xarray as xr
import numpy as np
import os
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColorBar, LinearColorMapper
from bokeh.palettes import Viridis256

# Initialize Bokeh output to display in notebook
output_notebook()

# %% RUN User defines directory path for dataset, dataset used, and dataset final location
# User set absolute_path
absolute_path = "C:/Users/nml/OneDrive - NIST/Documents/NIST/WUI_smoke/"  # USER ENTERED PROJECT PATH
os.chdir(absolute_path)

# Read the Excel file, specifying the sheet name
data = pd.read_excel(
    "./burn_data/smps/MH_apollo_bed_05312024_MassConc.xlsx",
    sheet_name="all_data",  # specify the sheet name
    parse_dates=[0],
    index_col=[0],
)


# Try to convert column names to floats
def try_parse_float(value):
    try:
        return float(value)
    except ValueError:
        return np.nan  # or handle the non-numeric case as needed


wavelength = [try_parse_float(v) for v in data.columns]

# Filter data for the specific day (2024-05-31)
data_filtered = data.loc["2024-05-31"]

# Drop columns with all NaN values
data_filtered = data_filtered.dropna(axis=1, how="all")

# Ensure that the data is only for the day 2024-05-31
time = data_filtered.index

# Handle cases where the wavelength list might be empty or contain NaNs
wavelength = [w for w in wavelength if not np.isnan(w)]

# Create the DataArray
da = xr.DataArray(
    data=data_filtered.values,  # use .values to get the underlying numpy array
    dims=["time", "wavelength"],
    coords={"time": time, "wavelength": wavelength},
)

# Convert time to hours since start of the day
time_hours = (time - time.normalize()).total_seconds() / 3600

# Prepare data for Bokeh
data_for_plot = np.log10(da.T.where(da.T > 0))  # Apply log scale safely

# Compute min and max values for the color mapper
data_min = float(data_for_plot.min())
data_max = float(data_for_plot.max())

# Create a Bokeh figure
p = figure(
    width=800,
    height=500,
    title="Mass Concentration of Particles for Burn 10",
    x_axis_label="Hours of the day",
    y_axis_label="Diameter (nm)",
    x_range=(0, max(time_hours)),
    y_range=(min(wavelength), max(wavelength)),
    x_axis_type="linear",
    y_axis_type="log",
)

# Add image
color_mapper = LinearColorMapper(palette=Viridis256, low=data_min, high=data_max)
p.image(
    image=[data_for_plot],
    x=0,
    y=min(wavelength),
    dw=max(time_hours),
    dh=max(wavelength),
    color_mapper=color_mapper,
)

# Add color bar
color_bar = ColorBar(color_mapper=color_mapper, width=8, location=(0, 0))
p.add_layout(color_bar, "right")

# Set major ticks on x-axis every 2 hours
p.xaxis.ticker = [i for i in range(0, int(max(time_hours)) + 1, 1)]

# Set major ticks on y-axis with more intervals
# Choose tick positions based on the log scale range
y_ticks = [10, 20, 40, 60, 80, 100, 200, 400]
p.yaxis.ticker = y_ticks

# Customize tick parameters for major ticks
p.yaxis.major_label_text_font_size = "12pt"
p.yaxis.major_tick_line_width = 2
p.xaxis.major_label_text_font_size = "12pt"
p.xaxis.major_tick_line_width = 2

# Update axis labels
p.xaxis.axis_label = "Hours of the day"
p.yaxis.axis_label = "Diameter (nm)"
p.title.text = "Mass Concentration of Particles for Burn 10"

# Show the plot
show(p)

# %%
