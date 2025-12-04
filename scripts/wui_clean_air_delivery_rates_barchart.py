"""
WUI CADR Bar Chart Visualization Module

This script generates comprehensive bar chart visualizations of Clean Air Delivery Rates
(CADR) from wildland-urban interface smoke experiments. It creates publication-quality
figures comparing CADR performance across different experimental configurations,
filter types, and particle size fractions.

Key Features:
    - Multi-instrument CADR comparison bar charts
    - PM size-dependent CADR analysis (PM0.4, PM1, PM2.5, PM10)
    - Filter configuration comparisons (1 vs 2 vs 4 air cleaners)
    - New vs used filter performance analysis
    - MERV-13 vs MERV-12A filter comparison
    - Statistical analysis integration
    - Automated figure generation with metadata

Chart Types Generated:
    1. Main CADR comparison (all configurations)
    2. Filter count analysis (sealed room vs whole house)
    3. Filter condition comparison (new vs used filters)
    4. Filter type comparison (MERV-13 vs MERV-12A)

Data Sources:
    - Decay rate CSV files from multiple instruments
    - Burn log with experimental configuration details
    - Baseline and uncertainty data from CADR calculations

Visualization Features:
    - Color-coded bars by burn configuration
    - Error bars representing 95% confidence intervals
    - Consistent formatting across all charts
    - Interactive Bokeh HTML outputs
    - Customizable axis ranges and labels

Methodology:
    - CADR derived from exponential decay rates
    - Statistical comparisons using z-tests
    - Uncertainty propagation from decay fitting
    - Baseline correction applied consistently

Outputs:
    - HTML interactive figures in Paper_figures directory
    - PNG exports for publication (optional)
    - Metadata annotations on each figure

Dependencies:
    - pandas: Data loading and manipulation
    - numpy: Numerical calculations
    - bokeh: Interactive visualization
    - metadata_utils: Script metadata generation

Configuration Constants:
    - BURN_LABELS: Human-readable burn identifiers
    - COLOR_PALETTE: Consistent color scheme across figures
    - Y_AXIS_RANGES: Standardized axis limits

Author: Nathan Lima
Date: 2024-2025
"""

import os
import pandas as pd
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, Label, Range1d, Div, Span
from bokeh.plotting import figure
from bokeh.models.formatters import NumeralTickFormatter
from bokeh.layouts import column

# Import utils
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(os.path.join(grandparent_dir, "general_utils", "scripts"))

# utils
from metadata_utils import get_script_metadata  # type: ignore[import-untyped]  # pylint: disable=import-error,wrong-import-position  # noqa: E402

# Configuration constants
ABSOLUTE_PATH = (
    "C:/Users/nml/OneDrive - NIST/Documents/NIST/WUI_smoke/burn_data/burn_calcs"
)
OUTPUT_PATH = "C:/Users/nml/OneDrive - NIST/Documents/NIST/WUI_smoke/Paper_figures"

# Define burn labels for consistent naming across charts
BURN_LABELS = {
    "burn1": "01-House",
    "burn2": "4 Air Cleaners",
    "burn3": "03-House-1-U",
    "burn4": "1 Air Cleaner",
    "burn5": "05-Room",
    "burn6": "1 Air Cleaner (Sealed Room)",
    "burn7": "07-House-2A-N",
    "burn8": "08-House-2A-U",
    "burn9": "2 Air Cleaners",
    "burn10": "10-House-2-U",
}

# Color scheme for different data types
COLORS = {
    "SMPS": "#d45087",  # Pink-red for standard SMPS data
    "SMPS_scaled": "#1900ff",  # Blue for scaled SMPS data
}

# Reference line parameters
REFERENCE_CADR = 390  # m³/h
REFERENCE_ERROR = 30  # ±30 m³/h


def read_smps_data():
    """
    Read CADR data from SMPS Excel file.

    Returns:
        dict: Dictionary with burn IDs as keys and CADR data as values
              Format: {burn_id: {'CADR': value, 'uncertainty': value}}
    """
    file_path = os.path.join(ABSOLUTE_PATH, "SMPS_decay_and_CADR.xlsx")

    try:
        # Read Excel file
        df = pd.read_excel(file_path)

        # Filter for Total Concentration data only
        df_filtered = df[df["pollutant"] == "Total Concentration (µg/m³)"].copy()

        # Remove rows without CADR values
        df_filtered = df_filtered[df_filtered["CADR_per_CRbox"].notna()]

        # Extract burn data into dictionary
        result = {}
        for _, row in df_filtered.iterrows():
            burn = row["burn"]
            cadr = row["CADR_per_CRbox"]
            uncertainty = row["CADR_per_CRbox_uncertainty"]

            result[burn] = {"CADR": cadr, "uncertainty": uncertainty}

        return result

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}


def add_reference_line_with_error(plot):
    """
    Add reference line at 390 m³/h with error band.

    Args:
        plot: Bokeh figure object to add reference line to
    """
    # Error band as translucent filled area - extend to full plot width
    num_categories = len(plot.x_range.factors)
    plot.quad(
        top=REFERENCE_CADR + REFERENCE_ERROR,
        bottom=REFERENCE_CADR - REFERENCE_ERROR,
        left=0,  # Extend beyond the leftmost bar
        right=num_categories,  # Extend beyond the rightmost bar
        fill_color="gray",
        fill_alpha=0.2,
        line_color=None,
    )

    # Main reference line
    hline = Span(
        location=REFERENCE_CADR,
        dimension="width",
        line_color="black",
        line_dash="dashed",
        line_width=2,
    )
    plot.add_layout(hline)

    # Label for reference line
    label = Label(
        x=0,
        y=REFERENCE_CADR,
        text=f"ASTM WK81750 Rate: {REFERENCE_CADR} ± {REFERENCE_ERROR} m³/h",
        text_color="black",
        text_font_style="bold",
        x_offset=10,
        y_offset=10,
    )
    plot.add_layout(label)


def create_base_figure(x_labels, chart_title=""):
    """
    Create base Bokeh figure with standard formatting.

    Args:
        x_labels (list): List of x-axis labels
        chart_title (str): Title for the chart

    Returns:
        figure: Configured Bokeh figure object
    """
    p = figure(
        x_range=x_labels,
        height=500,
        width=900,
        title=chart_title,
        toolbar_location="right",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        background_fill_color="white",
        border_fill_color="white",
    )

    # Customize plot appearance
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = "lightgray"
    p.ygrid.grid_line_alpha = 0.6
    p.y_range = Range1d(0, 600)
    p.yaxis.axis_label = "CADR per CR box (m³/h)"
    p.yaxis.formatter = NumeralTickFormatter(format="0")
    p.xaxis.major_label_orientation = "horizontal"

    return p


def configure_legend(plot):
    """
    Configure legend properties after legend items have been added.

    Args:
        plot: Bokeh figure object with legend items
    """
    if plot.legend:
        plot.legend.location = "top_left"
        plot.legend.background_fill_alpha = 0.6


def add_bars_and_error_bars(plot, source_data):
    """
    Add bars and error bars to a Bokeh plot.

    Args:
        plot: Bokeh figure object
        source_data (dict): Data dictionary for ColumnDataSource
    """
    source = ColumnDataSource(data=source_data)

    # Add bars - check if we have individual colors or single color
    if "color" in source_data:
        plot.vbar(
            x="burn",
            top="CADR",
            width=0.5,
            source=source,
            color="color",
            legend_field="legend_label",
        )
    else:
        plot.vbar(
            x="burn",
            top="CADR",
            width=0.5,
            source=source,
            color=COLORS["SMPS"],
            legend_label="PM0.4",
        )

    # Add error bars
    plot.segment(
        x0="burn",
        y0="lower",
        x1="burn",
        y1="upper",
        source=source,
        line_color="black",
        line_width=1.5,
    )


def create_filter_count_chart(smps_data, script_metadata):
    """
    Create bar chart comparing CADR by number of air cleaners (1, 2, 4).

    Args:
        smps_data (dict): SMPS data dictionary
        script_metadata (str): Script execution metadata

    Returns:
        column: Bokeh layout with plot and metadata
    """
    filter_burns = ["burn4", "burn9", "burn2"]  # 1, 2, and 4 CR boxes

    # Prepare data for plotting
    source_data = {"burn": [], "CADR": [], "upper": [], "lower": []}

    for burn in filter_burns:
        if burn not in smps_data:
            continue

        burn_label = BURN_LABELS.get(burn, burn)
        data = smps_data[burn]
        cadr = data["CADR"]
        uncertainty = data["uncertainty"]

        source_data["burn"].append(burn_label)
        source_data["CADR"].append(cadr)
        source_data["upper"].append(cadr + uncertainty)
        source_data["lower"].append(max(0, cadr - uncertainty))

    # Create figure and add elements
    p = create_base_figure(source_data["burn"])
    add_bars_and_error_bars(p, source_data)
    add_reference_line_with_error(p)
    configure_legend(p)  # Configure legend after bars are added

    # Create summary metadata
    summary = []
    for burn in filter_burns:
        if burn in smps_data:
            cadr = smps_data[burn]["CADR"]
            uncertainty = smps_data[burn]["uncertainty"]
            burn_label = BURN_LABELS.get(burn, burn)
            summary.append(f"{burn_label}: {cadr:.1f} ± {uncertainty:.1f} m³/h")

    summary_text = "<br>".join(summary)
    div_text = f"<small>SMPS Total Concentration (µg/m³) CADR:<br>{summary_text}<br>{script_metadata}</small>"
    text_div = Div(text=div_text, width=900)

    return column(p, text_div)


def create_scaled_comparison_chart(smps_data, script_metadata):
    """
    Create bar chart with scaled burn4, original burn4, and burn6.

    Args:
        smps_data (dict): SMPS data dictionary
        script_metadata (str): Script execution metadata

    Returns:
        column: Bokeh layout with plot and metadata
    """
    scaling_factor = 33 / 324  # Room volume scaling factor

    # Prepare data for plotting
    source_data = {
        "burn": [],
        "CADR": [],
        "upper": [],
        "lower": [],
        "color": [],
        "legend_label": [],
    }

    # Add scaled burn4 data
    if "burn4" in smps_data:
        burn4_data = smps_data["burn4"]
        burn4_cadr = burn4_data["CADR"]
        burn4_uncertainty = burn4_data["uncertainty"]

        # Scaled version
        scaled_cadr = burn4_cadr * scaling_factor
        scaled_uncertainty = burn4_uncertainty * scaling_factor

        source_data["burn"].append("1 Air Cleaner (Room Scaled)")
        source_data["CADR"].append(scaled_cadr)
        source_data["upper"].append(scaled_cadr + scaled_uncertainty)
        source_data["lower"].append(max(0, scaled_cadr - scaled_uncertainty))
        source_data["color"].append(COLORS["SMPS_scaled"])
        source_data["legend_label"].append("PM0.4 (Scaled)")

        # Original version
        source_data["burn"].append("1 Air Cleaner (House)")
        source_data["CADR"].append(burn4_cadr)
        source_data["upper"].append(burn4_cadr + burn4_uncertainty)
        source_data["lower"].append(max(0, burn4_cadr - burn4_uncertainty))
        source_data["color"].append(COLORS["SMPS"])
        source_data["legend_label"].append("PM0.4")

    # Add burn6 data
    if "burn6" in smps_data:
        burn6_data = smps_data["burn6"]
        burn6_cadr = burn6_data["CADR"]
        burn6_uncertainty = burn6_data["uncertainty"]

        source_data["burn"].append(BURN_LABELS["burn6"])
        source_data["CADR"].append(burn6_cadr)
        source_data["upper"].append(burn6_cadr + burn6_uncertainty)
        source_data["lower"].append(max(0, burn6_cadr - burn6_uncertainty))
        source_data["color"].append(COLORS["SMPS"])
        source_data["legend_label"].append("PM0.4")

    # Create figure and add elements
    p = create_base_figure(source_data["burn"])
    add_bars_and_error_bars(p, source_data)
    add_reference_line_with_error(p)
    configure_legend(p)  # Configure legend after bars are added

    # Create summary metadata
    summary = []
    if "burn4" in smps_data:
        burn4_cadr = smps_data["burn4"]["CADR"]
        burn4_uncertainty = smps_data["burn4"]["uncertainty"]
        scaled_cadr = burn4_cadr * scaling_factor
        scaled_uncertainty = burn4_uncertainty * scaling_factor

        summary.append(
            f"04-House-1-N (Scaled): {scaled_cadr:.1f} ± {scaled_uncertainty:.1f} m³/h"
        )
        summary.append(f"04-House-1-N: {burn4_cadr:.1f} ± {burn4_uncertainty:.1f} m³/h")

    if "burn6" in smps_data:
        burn6_cadr = smps_data["burn6"]["CADR"]
        burn6_uncertainty = smps_data["burn6"]["uncertainty"]
        summary.append(
            f"{BURN_LABELS['burn6']}: {burn6_cadr:.1f} ± {burn6_uncertainty:.1f} m³/h"
        )

    summary_text = "<br>".join(summary)
    div_text = f"<small>SMPS Total Concentration (µg/m³) CADR:<br>{summary_text}<br>Scaling factor: {scaling_factor:.5f}<br>{script_metadata}</small>"
    text_div = Div(text=div_text, width=900)

    return column(p, text_div)


def create_combined_comparison_chart(smps_data, script_metadata):
    """
    Create bar chart with 1 Air Cleaner (House), 1 Air Cleaner (Room), and 4 Air Cleaners (House).

    Args:
        smps_data (dict): SMPS data dictionary
        script_metadata (str): Script execution metadata

    Returns:
        column: Bokeh layout with plot and metadata
    """
    # Prepare data for plotting
    source_data = {"burn": [], "CADR": [], "upper": [], "lower": []}

    # Add burn4 data (1 Air Cleaner House)
    if "burn4" in smps_data:
        burn4_data = smps_data["burn4"]
        burn4_cadr = burn4_data["CADR"]
        burn4_uncertainty = burn4_data["uncertainty"]

        source_data["burn"].append("1 Air Cleaner (House)")
        source_data["CADR"].append(burn4_cadr)
        source_data["upper"].append(burn4_cadr + burn4_uncertainty)
        source_data["lower"].append(max(0, burn4_cadr - burn4_uncertainty))

    # Add burn6 data (1 Air Cleaner Room)
    if "burn6" in smps_data:
        burn6_data = smps_data["burn6"]
        burn6_cadr = burn6_data["CADR"]
        burn6_uncertainty = burn6_data["uncertainty"]

        source_data["burn"].append("1 Air Cleaner (Sealed Room)")
        source_data["CADR"].append(burn6_cadr)
        source_data["upper"].append(burn6_cadr + burn6_uncertainty)
        source_data["lower"].append(max(0, burn6_cadr - burn6_uncertainty))

    # Add burn2 data (4 Air Cleaners House)
    if "burn2" in smps_data:
        burn2_data = smps_data["burn2"]
        burn2_cadr = burn2_data["CADR"]
        burn2_uncertainty = burn2_data["uncertainty"]

        source_data["burn"].append("4 Air Cleaners (House)")
        source_data["CADR"].append(burn2_cadr)
        source_data["upper"].append(burn2_cadr + burn2_uncertainty)
        source_data["lower"].append(max(0, burn2_cadr - burn2_uncertainty))

    # Create figure and add elements
    p = create_base_figure(source_data["burn"])
    add_bars_and_error_bars(p, source_data)
    add_reference_line_with_error(p)
    configure_legend(p)  # Configure legend after bars are added

    # Create summary metadata
    summary = []
    if "burn4" in smps_data:
        cadr = smps_data["burn4"]["CADR"]
        uncertainty = smps_data["burn4"]["uncertainty"]
        summary.append(f"1 Air Cleaner (House): {cadr:.1f} ± {uncertainty:.1f} m³/h")

    if "burn6" in smps_data:
        cadr = smps_data["burn6"]["CADR"]
        uncertainty = smps_data["burn6"]["uncertainty"]
        summary.append(
            f"1 Air Cleaner (Sealed Room): {cadr:.1f} ± {uncertainty:.1f} m³/h"
        )

    if "burn2" in smps_data:
        cadr = smps_data["burn2"]["CADR"]
        uncertainty = smps_data["burn2"]["uncertainty"]
        summary.append(f"4 Air Cleaners (House): {cadr:.1f} ± {uncertainty:.1f} m³/h")

    summary_text = "<br>".join(summary)
    div_text = f"<small>SMPS Total Concentration (µg/m³) CADR:<br>{summary_text}<br>{script_metadata}</small>"
    text_div = Div(text=div_text, width=900)

    return column(p, text_div)


def main():
    """
    Main function to create and save all three CADR comparison charts.
    """
    # Read SMPS data
    smps_data = read_smps_data()
    if not smps_data:
        print("No SMPS data found. Exiting.")
        return

    # Get script metadata
    metadata = get_script_metadata()

    # Create and save Chart 1: Filter count comparison (1, 2, 4 air cleaners)
    print("Creating filter count comparison chart...")
    chart1 = create_filter_count_chart(smps_data, metadata)
    output_file(os.path.join(OUTPUT_PATH, "smps_cadr_by_filter_count_original.html"))
    show(chart1)

    # Create and save Chart 2: Scaled comparison
    print("Creating scaled comparison chart...")
    chart2 = create_scaled_comparison_chart(smps_data, metadata)
    output_file(os.path.join(OUTPUT_PATH, "smps_cadr_scaled_comparison.html"))
    show(chart2)

    # Create and save Chart 3: Combined comparison (NEW)
    print("Creating combined comparison chart...")
    chart3 = create_combined_comparison_chart(smps_data, metadata)
    output_file(os.path.join(OUTPUT_PATH, "smps_cadr_combined_comparison.html"))
    show(chart3)

    print("All charts created successfully!")


if __name__ == "__main__":
    main()
