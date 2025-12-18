# Enhanced WUI Clean Air Delivery Rate vs Total Surface Area Analysis
# This script analyzes the relationship between filter surface area and CADR performance
# for Wildland-Urban Interface (WUI) smoke filtration research

# Import necessary libraries
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from bokeh.plotting import figure, show, save, output_file
from bokeh.models import ColumnDataSource, Range1d
from bokeh.models.formatters import NumeralTickFormatter
from bokeh.layouts import column
import matplotlib.pyplot as plt
import os

# Add repository root to path for portable data access
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

from src.data_paths import get_data_root, get_common_file

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Define constants for MERV filter surface areas (m²)
MERV_13_SURFACE_AREA_PER_CRBOX = 3.87483096  # m² per CRBox
MERV_12A_SURFACE_AREA_PER_CRBOX = 15.99609704  # m² per CRBox

# Output directory for saving figures - using portable path
data_root = get_data_root()
OUTPUT_DIR = str(get_common_file('output_figures'))

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# BURN CHARACTERISTICS CONFIGURATION
# =============================================================================

# Define burn characteristics (using default MERV 13 unless specified)
burn_characteristics = {
    'burn2': {'crboxes': 4, 'merv_type': 'MERV_13', 'filter_condition': 'New'},
    'burn3': {'crboxes': 1, 'merv_type': 'MERV_13', 'filter_condition': 'Used'},
    'burn4': {'crboxes': 1, 'merv_type': 'MERV_13', 'filter_condition': 'New'},
    'burn6': {'crboxes': 1, 'merv_type': 'MERV_13', 'filter_condition': 'New'},
    'burn7': {'crboxes': 2, 'merv_type': 'MERV_12A', 'filter_condition': 'New'},
    'burn8': {'crboxes': 2, 'merv_type': 'MERV_12A', 'filter_condition': 'Used'},
    'burn9': {'crboxes': 2, 'merv_type': 'MERV_13', 'filter_condition': 'New'},
    'burn10': {'crboxes': 2, 'merv_type': 'MERV_13', 'filter_condition': 'Used'}
}

# Define instruments and their corresponding pollutant types
instruments = ['AeroTrakB', 'AeroTrakK', 'DustTrak', 'PurpleAirK', 'QuantAQB', 'QuantAQK', 'SMPS']
pollutant_types_pm25 = {
    'AeroTrakB': 'PM3 (µg/m³)',
    'AeroTrakK': 'PM3 (µg/m³)',
    'DustTrak': 'PM2.5 (µg/m³)',
    'PurpleAirK': 'PM2.5 (µg/m³)',
    'QuantAQB': 'PM2.5 (µg/m³)',
    'QuantAQK': 'PM2.5 (µg/m³)',
    'SMPS': 'Total Concentration (µg/m³)'
}

# =============================================================================
# DATA READING FUNCTIONS
# =============================================================================

def read_instrument_cadr(instrument, pollutant_type):
    """
    Read Clean Air Delivery Rate (CADR) data from Excel files for a specific instrument.
    
    Parameters:
    -----------
    instrument : str
        Name of the instrument (e.g., 'AeroTrakB', 'DustTrak', etc.)
    pollutant_type : str
        Type of pollutant measured (e.g., 'PM2.5 (µg/m³)')
    
    Returns:
    --------
    dict
        Dictionary with burn names as keys and CADR data (value and uncertainty) as values
    """
    file_path = str(data_root / "burn_data" / "burn_calcs" / f"{instrument}_decay_and_CADR.xlsx")
    try:
        # Read Excel file and filter for relevant data
        df = pd.read_excel(file_path)
        df_filtered = df[(df['pollutant'] == pollutant_type) & (df['CADR'].notna())]
        
        # Create result dictionary
        result = {}
        for _, row in df_filtered.iterrows():
            burn = row['burn']
            cadr = row['CADR']
            uncertainty = row['CADR_uncertainty']
            result[burn] = {'CADR': cadr, 'uncertainty': uncertainty}
        
        return result
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}

def is_in_bedroom(instrument, burn_number):
    """
    Determine if an instrument is located in the bedroom for a given burn.
    
    Parameters:
    -----------
    instrument : str
        Name of the instrument
    burn_number : int
        Burn number
    
    Returns:
    --------
    bool
        True if instrument is in bedroom, False otherwise
    """
    # Bedroom instruments for all burns
    if instrument in ['AeroTrakB', 'QuantAQB', 'SMPS']:
        return True
    # DustTrak was in bedroom for burns 1-6
    if instrument == 'DustTrak' and 1 <= burn_number <= 6:
        return True
    return False

def calculate_mean_cadr(data_by_instrument, burn, location='All'):
    """
    Calculate mean CADR and uncertainty across instruments for a specific burn.
    
    Parameters:
    -----------
    data_by_instrument : dict
        Dictionary containing CADR data for all instruments
    burn : str
        Burn identifier (e.g., 'burn2')
    location : str
        Location filter ('All' or 'Bedroom')
    
    Returns:
    --------
    tuple
        (mean_cadr, mean_uncertainty) or (None, None) if no data available
    """
    cadr_values = []
    uncertainties = []
    
    # Iterate through instruments (excluding SMPS and Mean for this calculation)
    for instrument in data_by_instrument:
        if instrument in ['SMPS', 'Mean']:
            continue
        
        if burn in data_by_instrument[instrument]:
            burn_number = int(burn.replace('burn', ''))
            
            # Apply location filter if specified
            if location == 'Bedroom' and not is_in_bedroom(instrument, burn_number):
                continue
            
            cadr_values.append(data_by_instrument[instrument][burn]['CADR'])
            uncertainties.append(data_by_instrument[instrument][burn]['uncertainty'])
    
    if cadr_values:
        # Calculate mean and uncertainties
        mean_cadr = np.mean(cadr_values)
        # Standard error of the mean
        sem = np.std(cadr_values) / np.sqrt(len(cadr_values))
        # Combined instrument uncertainty
        instrument_uncertainty = np.sqrt(np.sum(np.array(uncertainties)**2)) / len(cadr_values)
        # Total uncertainty (quadrature sum)
        mean_uncertainty = np.sqrt(sem**2 + instrument_uncertainty**2)
        
        return mean_cadr, mean_uncertainty
    
    return None, None

# =============================================================================
# DATA PROCESSING
# =============================================================================

# Read CADR data for all instruments
print("Reading CADR data from instrument files...")
data_by_instrument = {}
for instrument in instruments:
    data_by_instrument[instrument] = read_instrument_cadr(instrument, pollutant_types_pm25[instrument])
    print(f"Loaded data for {instrument}: {len(data_by_instrument[instrument])} burns")

# Prepare data for plotting
burns_to_plot = ['burn2', 'burn3', 'burn4', 'burn6', 'burn7', 'burn8', 'burn9', 'burn10']

# Initialize data arrays for both absolute and normalized plots
surface_areas = []
mean_cadr = []
mean_uncertainty = []
smps_cadr = []
smps_uncertainty = []
filter_conditions = []

# For normalized plot (CADR per CRBox and Surface Area per CRBox)
normalized_surface_areas = []
normalized_mean_cadr = []
normalized_mean_uncertainty = []
normalized_smps_cadr = []
normalized_smps_uncertainty = []

print("\nProcessing burn data...")
for burn in burns_to_plot:
    # Get burn characteristics
    crboxes = burn_characteristics[burn]['crboxes']
    merv_type = burn_characteristics[burn]['merv_type']
    filter_condition = burn_characteristics[burn]['filter_condition']
    
    # Calculate total surface area
    surface_area_per_crbox = (MERV_13_SURFACE_AREA_PER_CRBOX if merv_type == 'MERV_13' 
                             else MERV_12A_SURFACE_AREA_PER_CRBOX)
    total_surface_area = crboxes * surface_area_per_crbox
    
    # Store surface area and filter condition
    surface_areas.append(total_surface_area)
    filter_conditions.append(filter_condition)
    
    # Calculate normalized surface area (per CRBox)
    normalized_surface_areas.append(total_surface_area / crboxes)
    
    # Calculate mean CADR (bedroom location for burn6, all locations for others)
    location = 'Bedroom' if burn == 'burn6' else 'All'
    mean_val, uncertainty_val = calculate_mean_cadr(data_by_instrument, burn, location)
    mean_cadr.append(mean_val)
    mean_uncertainty.append(uncertainty_val)
    
    # Calculate normalized values (per CRBox)
    if mean_val is not None:
        normalized_mean_cadr.append(mean_val / crboxes)
        normalized_mean_uncertainty.append(uncertainty_val / crboxes)
    else:
        normalized_mean_cadr.append(None)
        normalized_mean_uncertainty.append(None)
    
    print(f"{burn}: Surface Area = {total_surface_area:.2f} m² ({total_surface_area/crboxes:.2f} m²/CRBox), "
          f"CADR = {mean_val:.2f} ± {uncertainty_val:.2f} ({mean_val/crboxes:.2f} ± {uncertainty_val/crboxes:.2f} per CRBox)")
    
    # Get SMPS data if available
    if 'SMPS' in data_by_instrument and burn in data_by_instrument['SMPS']:
        smps_val = data_by_instrument['SMPS'][burn]['CADR']
        smps_unc = data_by_instrument['SMPS'][burn]['uncertainty']
        smps_cadr.append(smps_val)
        smps_uncertainty.append(smps_unc)
        # Normalized SMPS values
        normalized_smps_cadr.append(smps_val / crboxes)
        normalized_smps_uncertainty.append(smps_unc / crboxes)
    else:
        smps_cadr.append(None)
        smps_uncertainty.append(None)
        normalized_smps_cadr.append(None)
        normalized_smps_uncertainty.append(None)

# =============================================================================
# BOKEH DATA SOURCE PREPARATION
# =============================================================================

def create_data_sources(surface_areas, mean_cadr_vals, mean_uncertainty_vals, 
                       smps_cadr_vals, smps_uncertainty_vals, filter_conditions, 
                       include_used=True):
    """
    Create Bokeh ColumnDataSource objects for plotting.
    
    Parameters:
    -----------
    surface_areas : list
        Surface area values
    mean_cadr_vals : list
        Mean CADR values
    mean_uncertainty_vals : list
        Mean CADR uncertainties
    smps_cadr_vals : list
        SMPS CADR values
    smps_uncertainty_vals : list
        SMPS CADR uncertainties
    filter_conditions : list
        Filter conditions ('New' or 'Used')
    include_used : bool
        Whether to include 'Used' filter data
    
    Returns:
    --------
    tuple
        (mean_source_new, mean_source_used, smps_source_new, smps_source_used)
    """
    # Set random seed for reproducible jittering
    np.random.seed(0)
    
    # Helper function to create jittered positions
    def create_jittered_positions(data, condition, target_condition):
        return [x + np.random.uniform(-0.2, 0.2) 
                for x, c in zip(data, condition) if c == target_condition]
    
    # Helper function to filter data by condition
    def filter_by_condition(data, condition, target_condition):
        return [y for y, c in zip(data, condition) if c == target_condition]
    
    # Create jittered surface areas for 'New' condition
    mean_surface_area_new_jittered = create_jittered_positions(surface_areas, filter_conditions, 'New')
    smps_surface_area_new_jittered = create_jittered_positions(
        [x for x, y in zip(surface_areas, smps_cadr_vals) if y is not None], 
        [c for c, y in zip(filter_conditions, smps_cadr_vals) if y is not None], 
        'New'
    )
    
    # Create data sources for 'New' filters
    mean_source_new = ColumnDataSource(data={
        'surface_area_jittered': mean_surface_area_new_jittered,
        'mean_cadr': filter_by_condition(mean_cadr_vals, filter_conditions, 'New'),
        'mean_lower': [cadr_val - unc_val for cadr_val, unc_val, condition in 
                      zip(mean_cadr_vals, mean_uncertainty_vals, filter_conditions) if condition == 'New'],
        'mean_upper': [cadr_val + unc_val for cadr_val, unc_val, condition in 
                      zip(mean_cadr_vals, mean_uncertainty_vals, filter_conditions) if condition == 'New']
    })
    
    smps_source_new = ColumnDataSource(data={
        'surface_area_jittered': smps_surface_area_new_jittered,
        'smps_cadr': [y for y, c in zip(smps_cadr_vals, filter_conditions) if y is not None and c == 'New'],
        'smps_lower': [cadr_val - unc_val for cadr_val, unc_val, condition in 
                      zip(smps_cadr_vals, smps_uncertainty_vals, filter_conditions) 
                      if cadr_val is not None and condition == 'New'],
        'smps_upper': [cadr_val + unc_val for cadr_val, unc_val, condition in 
                      zip(smps_cadr_vals, smps_uncertainty_vals, filter_conditions) 
                      if cadr_val is not None and condition == 'New']
    })
    
    # Create data sources for 'Used' filters (only if include_used is True)
    mean_source_used = None
    smps_source_used = None
    
    if include_used:
        mean_surface_area_used_jittered = create_jittered_positions(surface_areas, filter_conditions, 'Used')
        smps_surface_area_used_jittered = create_jittered_positions(
            [x for x, y in zip(surface_areas, smps_cadr_vals) if y is not None], 
            [c for c, y in zip(filter_conditions, smps_cadr_vals) if y is not None], 
            'Used'
        )
        
        mean_source_used = ColumnDataSource(data={
            'surface_area_jittered': mean_surface_area_used_jittered,
            'mean_cadr': filter_by_condition(mean_cadr_vals, filter_conditions, 'Used'),
            'mean_lower': [cadr_val - unc_val for cadr_val, unc_val, condition in 
                          zip(mean_cadr_vals, mean_uncertainty_vals, filter_conditions) if condition == 'Used'],
            'mean_upper': [cadr_val + unc_val for cadr_val, unc_val, condition in 
                          zip(mean_cadr_vals, mean_uncertainty_vals, filter_conditions) if condition == 'Used']
        })
        
        smps_source_used = ColumnDataSource(data={
            'surface_area_jittered': smps_surface_area_used_jittered,
            'smps_cadr': [y for y, c in zip(smps_cadr_vals, filter_conditions) if y is not None and c == 'Used'],
            'smps_lower': [cadr_val - unc_val for cadr_val, unc_val, condition in 
                          zip(smps_cadr_vals, smps_uncertainty_vals, filter_conditions) 
                          if cadr_val is not None and condition == 'Used'],
            'smps_upper': [cadr_val + unc_val for cadr_val, unc_val, condition in 
                          zip(smps_cadr_vals, smps_uncertainty_vals, filter_conditions) 
                          if cadr_val is not None and condition == 'Used']
        })
    
    return mean_source_new, mean_source_used, smps_source_new, smps_source_used

# =============================================================================
# REFERENCE DATA PROCESSING
# =============================================================================

# Read reference data from Excel file
print("Loading reference data...")
reference_df = pd.read_excel(str(data_root / "burn_data" / "reference_surfacearea_and_cadr.xlsx"))

# Create a dictionary to map paper names to colors
paper_colors = {}
for i, paper in enumerate(reference_df['Paper'].unique()):
    # Convert matplotlib color to hex format for Bokeh
    paper_colors[paper] = "#{:02x}{:02x}{:02x}".format(*[int(x*255) for x in plt.cm.tab20(i % 20)])

print(f"Loaded reference data from {len(reference_df['Paper'].unique())} papers")

# =============================================================================
# PLOT CREATION FUNCTIONS
# =============================================================================

def create_cadr_plot(surface_areas, mean_cadr_vals, mean_uncertainty_vals, 
                    smps_cadr_vals, smps_uncertainty_vals, filter_conditions,
                    title, y_axis_label, include_used=True, normalized=False):
    """
    Create a CADR vs Surface Area plot.
    
    Parameters:
    -----------
    surface_areas : list
        Surface area values
    mean_cadr_vals : list
        Mean CADR values
    mean_uncertainty_vals : list
        Mean CADR uncertainties
    smps_cadr_vals : list
        SMPS CADR values
    smps_uncertainty_vals : list
        SMPS CADR uncertainties
    filter_conditions : list
        Filter conditions
    title : str
        Plot title
    y_axis_label : str
        Y-axis label
    include_used : bool
        Whether to include used filter data
    normalized : bool
        Whether this is a normalized plot (affects axis ranges and labels)
    
    Returns:
    --------
    bokeh.plotting.figure
        Bokeh figure object
    """
    # Create data sources
    mean_source_new, mean_source_used, smps_source_new, smps_source_used = create_data_sources(
        surface_areas, mean_cadr_vals, mean_uncertainty_vals, 
        smps_cadr_vals, smps_uncertainty_vals, filter_conditions, include_used
    )
    
    # Create the figure
    p = figure(
        height=900,
        width=900,
        title=title,
        toolbar_location="right",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        background_fill_color="white",
        border_fill_color="white"
    )
    
    # Add mean CADR circles with error bars for 'New' filters
    p.scatter(
        x='surface_area_jittered',
        y='mean_cadr',
        size=10,
        line_color='#2ca02c',  # Green
        fill_color='#2ca02c',
        source=mean_source_new,
        legend_label='Mean PM2.5 (New)'
    )
    
    # Add error bars for mean CADR 'New' filters
    p.segment(
        x0='surface_area_jittered',
        y0='mean_lower',
        x1='surface_area_jittered',
        y1='mean_upper',
        line_color='#2ca02c',
        line_width=2,
        source=mean_source_new
    )
    
    # Add SMPS CADR circles with error bars for 'New' filters
    p.scatter(
        x='surface_area_jittered',
        y='smps_cadr',
        size=10,
        line_color='#d45087',  # Pink-red
        fill_color='#d45087',
        source=smps_source_new,
        legend_label='PM0.4 (SMPS) (New)'
    )
    
    # Add error bars for SMPS CADR 'New' filters
    p.segment(
        x0='surface_area_jittered',
        y0='smps_lower',
        x1='surface_area_jittered',
        y1='smps_upper',
        line_color='#d45087',
        line_width=2,
        source=smps_source_new
    )
    
    # Add 'Used' filter data if requested (commented out by default)
    if include_used and mean_source_used is not None:
        # Mean CADR for 'Used' filters (hatched pattern)
        p.scatter(
            x='surface_area_jittered',
            y='mean_cadr',
            size=10,
            line_color='#2ca02c',
            fill_color=None,
            hatch_pattern='right_diagonal_line',
            hatch_color='#2ca02c',
            source=mean_source_used,
            legend_label='Mean PM2.5 (Used)'
        )
        
        # Error bars for mean CADR 'Used' filters
        p.segment(
            x0='surface_area_jittered',
            y0='mean_lower',
            x1='surface_area_jittered',
            y1='mean_upper',
            line_color='#2ca02c',
            line_width=2,
            source=mean_source_used
        )
    
    if include_used and smps_source_used is not None:
        # SMPS CADR for 'Used' filters (hatched pattern)
        p.scatter(
            x='surface_area_jittered',
            y='smps_cadr',
            size=10,
            line_color='#d45087',
            fill_color=None,
            hatch_pattern='right_diagonal_line',
            hatch_color='#d45087',
            source=smps_source_used,
            legend_label='PM0.4 (SMPS) (Used)'
        )
        
        # Error bars for SMPS CADR 'Used' filters
        p.segment(
            x0='surface_area_jittered',
            y0='smps_lower',
            x1='surface_area_jittered',
            y1='smps_upper',
            line_color='#d45087',
            line_width=2,
            source=smps_source_used
        )
    
    # Add reference values to the plot
    for paper in reference_df['Paper'].unique():
        paper_data = reference_df[reference_df['Paper'] == paper]
        paper_source = ColumnDataSource(data={
            'surface_area': paper_data['surface_area'].tolist(),
            'cadr': paper_data['CADR'].tolist(),
            'lower': [cadr_val - unc_val for cadr_val, unc_val in 
                     zip(paper_data['CADR'], paper_data['CADR_uncertainty'])],
            'upper': [cadr_val + unc_val for cadr_val, unc_val in 
                     zip(paper_data['CADR'], paper_data['CADR_uncertainty'])]
        })
        
        # Add reference data points
        p.scatter(
            x='surface_area',
            y='cadr',
            size=10,
            marker='x',
            line_color=paper_colors[paper],
            line_width=2,
            source=paper_source,
            legend_label=paper
        )
        
        # Add error bars for reference data
        p.segment(
            x0='surface_area',
            y0='lower',
            x1='surface_area',
            y1='upper',
            line_color=paper_colors[paper],
            line_width=2,
            source=paper_source
        )
    
    # Customize the plot based on whether it's normalized or not
    if normalized:
        # For normalized plots (per CRBox)
        p.x_range = Range1d(0, 20)  # Surface area per CRBox is smaller
        p.y_range = Range1d(0, 1500)  # CADR per CRBox is smaller
        p.xaxis.axis_label = "Surface Area per CRBox (m²)"
    else:
        # For absolute plots
        p.x_range = Range1d(0, 35)
        p.y_range = Range1d(0, 2500)
        p.xaxis.axis_label = "Total Surface Area (m²)"
    
    p.yaxis.axis_label = y_axis_label
    p.xaxis.formatter = NumeralTickFormatter(format="0.0")
    p.yaxis.formatter = NumeralTickFormatter(format="0")
    p.legend.location = "top_right"
    
    return p

# =============================================================================
# CREATE PLOTS
# =============================================================================

print("\nCreating plots...")

# Create first plot: Absolute CADR vs Total Surface Area (without used data)
plot1 = create_cadr_plot(
    surface_areas, mean_cadr, mean_uncertainty, 
    smps_cadr, smps_uncertainty, filter_conditions,
    title="Total Surface Area vs. CADR",
    y_axis_label="CADR",
    include_used=False,  # Comment out used data as requested
    normalized=False
)

# Create second plot: Normalized CADR (per CRBox) vs Normalized Surface Area (per CRBox)
plot2 = create_cadr_plot(
    normalized_surface_areas, normalized_mean_cadr, normalized_mean_uncertainty,
    normalized_smps_cadr, normalized_smps_uncertainty, filter_conditions,
    title="Normalized Total Surface Area vs. CADR",
    y_axis_label="CADR per CRBox",
    include_used=False,  # Comment out used data as requested
    normalized=True
)

# Adjust y-axis range for normalized plot
plot2.y_range = Range1d(0, 1700)

# =============================================================================
# SAVE AND DISPLAY PLOTS
# =============================================================================

# Create layout with both plots
layout = column(plot1, plot2)

# Set up HTML output
output_file(os.path.join(OUTPUT_DIR, "cadr_vs_surface_area_analysis.html"))

# Save the plots
print(f"Saving plots to {OUTPUT_DIR}/cadr_vs_surface_area_analysis.html")
save(layout)

# Display the plots
print("Displaying plots...")
show(layout)

print("\nAnalysis complete!")
print(f"Plots saved to: {os.path.join(OUTPUT_DIR, 'cadr_vs_surface_area_analysis.html')}")

# =============================================================================
# OPTIONAL: UNCOMMENT BELOW TO INCLUDE USED FILTER DATA
# =============================================================================

# To include used filter data in the plots, uncomment the following section:

"""
# Create plots with used filter data included
plot1_with_used = create_cadr_plot(
    surface_areas, mean_cadr, mean_uncertainty, 
    smps_cadr, smps_uncertainty, filter_conditions,
    title="Total Surface Area vs. CADR (Including Used Filters)",
    y_axis_label="CADR",
    include_used=True,
    normalized=False
)

plot2_with_used = create_cadr_plot(
    normalized_surface_areas, normalized_mean_cadr, normalized_mean_uncertainty,
    normalized_smps_cadr, normalized_smps_uncertainty, filter_conditions,
    title="Surface Area per CRBox vs. CADR per CRBox (Including Used Filters)",
    y_axis_label="CADR per CRBox",
    include_used=True,
    normalized=True
)

# Create layout and save
layout_with_used = column(plot1_with_used, plot2_with_used)
output_file(os.path.join(OUTPUT_DIR, "cadr_vs_surface_area_analysis_with_used.html"))
save(layout_with_used)
show(layout_with_used)

print(f"Plots with used filter data saved to: {os.path.join(OUTPUT_DIR, 'cadr_vs_surface_area_analysis_with_used.html')}")
"""