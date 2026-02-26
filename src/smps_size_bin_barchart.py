"""
SMPS Size-Bin Baseline-Corrected Decay Rate Bar Charts
=======================================================

Generates bar charts showing baseline-corrected, per-air-cleaner decay rates
for the four SMPS ultrafine particle size bins used in the WUI Mobile Home IAQ
study. The x-axis shows the four size bins (9-100 nm, 100-200 nm, 200-300 nm,
300-437 nm) and conditions (1, 2, or 4 air cleaners; new vs. used filters;
MERV 12A vs. MERV 13) are shown as grouped bars.

Data source: SMPS_decay_and_CADR.xlsx, produced by
    clean_air_delivery_rates_pmsizes.py (run with dataset = "SMPS").

Prerequisite
------------
Run src/clean_air_delivery_rates_pmsizes.py with ``dataset = "SMPS"`` first to
generate the SMPS_decay_and_CADR.xlsx file in the burn_calcs directory.

Chart configurations
--------------------
1. Filter count comparison   : 1, 2, and 4 portable air cleaners (burns 4, 9, 2)
2. New vs. used filter        : new and used filters at 1 and 2 PAC counts
                                (burns 4, 3, 9, 10)
3. MERV filter comparison     : MERV 12A vs. MERV 13 (burns 7, 8, 9, 10)

Methodology
-----------
- Baseline: burn1 decay rate for each size bin (no air cleaners, whole-house)
- Correction: decay_corrected = (decay_observed - decay_baseline) / n_CR_boxes
- Only burns with CR boxes (n_CR_boxes > 0) are plotted

Output
------
HTML interactive figures saved to the ``output_figures`` directory:
  - SMPS_size_bin_bar_chart_filter_count.html
  - SMPS_size_bin_bar_chart_new_vs_used.html
  - SMPS_size_bin_bar_chart_merv_comparison.html

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2024-2026
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.io import output_file, save
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Div, Range1d
from bokeh.models.formatters import NumeralTickFormatter
from bokeh.plotting import figure
from bokeh.transform import dodge

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

from scripts import get_script_metadata  # pylint: disable=wrong-import-position
from src.data_paths import (  # pylint: disable=wrong-import-position
    get_common_file,
    get_data_root,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

data_root = get_data_root()

# burn_calcs uses the same path construction as decay_rate_barchart.py
BURN_CALCS_PATH = str(data_root / "burn_data" / "burn_calcs")
OUTPUT_PATH = str(get_common_file("output_figures"))

# SMPS size-bin column names (must match exactly what process_smps_data produces)
SMPS_SIZE_BINS = [
    "Ʃ9-100nm (µg/m³)",
    "Ʃ100-200nm (µg/m³)",
    "Ʃ200-300nm (µg/m³)",
    "Ʃ300-437nm (µg/m³)",
]

# Display labels for x-axis tick marks
BIN_DISPLAY_LABELS = {
    "Ʃ9-100nm (µg/m³)": "9-100 nm",
    "Ʃ100-200nm (µg/m³)": "100-200 nm",
    "Ʃ200-300nm (µg/m³)": "200-300 nm",
    "Ʃ300-437nm (µg/m³)": "300-437 nm",
}

# Short x-axis category labels used internally by Bokeh
BIN_CATEGORIES = [BIN_DISPLAY_LABELS[b] for b in SMPS_SIZE_BINS]

# Burn-to-number-of-CR-boxes mapping
CRBOX_MAPPING = {
    "burn1": 0,
    "burn2": 4,
    "burn3": 1,
    "burn4": 1,
    "burn5": 0,
    "burn6": 1,
    "burn7": 2,
    "burn8": 2,
    "burn9": 2,
    "burn10": 2,
}

# Text formatting — matches the style used in decay_rate_barchart.py
TEXT_CONFIG = {
    "font_size": "12pt",
    "title_font_size": "12pt",
    "axis_label_font_size": "12pt",
    "axis_tick_font_size": "12pt",
    "legend_font_size": "12pt",
    "label_font_size": "12pt",
    "font_style": "normal",
    "plot_font_style": "bold",
    "html_font_weight": "normal",
}

# Colors for bar conditions — cycles through these for multi-condition charts.
# Taken from the same palette used in decay_rate_barchart.py.
CONDITION_COLORS = [
    "#003f5c",  # deep blue
    "#665191",  # purple
    "#d45087",  # pink
    "#ff7c43",  # orange
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_smps_decay_data():
    """Load SMPS decay parameters from SMPS_decay_and_CADR.xlsx.

    Returns
    -------
    dict
        Nested dict: {size_bin_label: {burn_id: {"decay": float,
                                                   "uncertainty": float,
                                                   "CRboxes": int}}}
    """
    file_path = os.path.join(BURN_CALCS_PATH, "SMPS_decay_and_CADR.xlsx")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"SMPS decay file not found: {file_path}\n"
            "Run clean_air_delivery_rates_pmsizes.py with dataset='SMPS' first."
        )

    df = pd.read_excel(file_path)
    df = df[df["decay"].notna()]

    data = {bin_col: {} for bin_col in SMPS_SIZE_BINS}

    for _, row_data in df.iterrows():
        pollutant = row_data["pollutant"]
        if pollutant not in SMPS_SIZE_BINS:
            continue

        burn = row_data["burn"]
        decay = float(row_data["decay"])
        uncertainty = (
            float(row_data["decay_uncertainty"])
            if pd.notna(row_data["decay_uncertainty"])
            else 0.0
        )
        crboxes = CRBOX_MAPPING.get(burn, 1)

        data[pollutant][burn] = {
            "decay": decay,
            "uncertainty": uncertainty,
            "CRboxes": crboxes,
        }

    return data


# ---------------------------------------------------------------------------
# Baseline correction
# ---------------------------------------------------------------------------


def calculate_baselines(smps_data):
    """Calculate baseline decay rates from burn1 for each size bin.

    Parameters
    ----------
    smps_data : dict
        Output of load_smps_decay_data().

    Returns
    -------
    dict
        {size_bin_label: baseline_decay_rate (float)}
    """
    baselines = {}
    for bin_col in SMPS_SIZE_BINS:
        if "burn1" in smps_data[bin_col]:
            baselines[bin_col] = smps_data[bin_col]["burn1"]["decay"]
        else:
            print(f"  Warning: burn1 data missing for {bin_col} — baseline set to 0.")
            baselines[bin_col] = 0.0
    return baselines


def apply_baseline_correction(smps_data, baselines, burns, normalize_by_crboxes=True):
    """Apply baseline correction (and optional PAC normalization) for a set of burns.

    Parameters
    ----------
    smps_data : dict
        Output of load_smps_decay_data().
    baselines : dict
        Output of calculate_baselines().
    burns : list of str
        Burn IDs to process.
    normalize_by_crboxes : bool
        If True, divide baseline-corrected decay by number of CR boxes.

    Returns
    -------
    dict
        {size_bin_label: {burn_id: {"decay": float, "uncertainty": float}}}
    """
    corrected = {bin_col: {} for bin_col in SMPS_SIZE_BINS}

    for bin_col in SMPS_SIZE_BINS:
        baseline = baselines[bin_col]
        for burn in burns:
            if burn not in smps_data[bin_col]:
                continue
            entry = smps_data[bin_col][burn]
            crboxes = entry["CRboxes"]
            if crboxes == 0:
                continue  # skip no-CR-box burns (baselines)

            if normalize_by_crboxes:
                decay_corr = (entry["decay"] - baseline) / crboxes
                unc_corr = entry["uncertainty"] / crboxes
            else:
                decay_corr = entry["decay"] - baseline
                unc_corr = entry["uncertainty"]

            corrected[bin_col][burn] = {
                "decay": decay_corr,
                "uncertainty": unc_corr,
            }

    return corrected


# ---------------------------------------------------------------------------
# Chart creation
# ---------------------------------------------------------------------------


def create_smps_bin_barchart(corrected_data, config, script_metadata):
    """Create a grouped bar chart for SMPS size bins.

    Parameters
    ----------
    corrected_data : dict
        Output of apply_baseline_correction().
    config : dict
        Chart configuration dict with keys:
            - "burns": list of burn IDs to plot
            - "labels": {burn_id: display_label}
            - "filename": output HTML filename
            - "title": figure title string
    script_metadata : str
        Metadata string from get_script_metadata().

    Returns
    -------
    bokeh layout
        Bokeh column layout containing the figure and metadata div.
    """
    burns = config["burns"]
    labels = config["labels"]

    # Build source data table
    source_data = {
        "bin_label": [],
        "condition": [],
        "decay": [],
        "upper": [],
        "lower": [],
        "is_used": [],
    }
    detailed_info = []

    for bin_col in SMPS_SIZE_BINS:
        display = BIN_DISPLAY_LABELS[bin_col]
        for burn in burns:
            if burn not in corrected_data[bin_col]:
                continue
            entry = corrected_data[bin_col][burn]
            decay = entry["decay"]
            unc = entry["uncertainty"]
            cond_label = labels[burn]
            is_used = burn in ["burn3", "burn8", "burn10"]

            source_data["bin_label"].append(display)
            source_data["condition"].append(cond_label)
            source_data["decay"].append(decay)
            source_data["upper"].append(decay + unc)
            source_data["lower"].append(decay - unc)
            source_data["is_used"].append(is_used)

            detailed_info.append(
                f"{cond_label} ({display}): {decay:.3f} ± {unc:.3f} h\u207b\u00b9"
            )

    # Unique conditions (preserve insertion order)
    unique_conditions = list(dict.fromkeys(source_data["condition"]))

    # Create figure
    p = figure(
        x_range=BIN_CATEGORIES,
        height=700,
        width=1200,
        toolbar_location="right",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        background_fill_color="white",
        border_fill_color="white",
    )

    n_cond = len(unique_conditions)
    bar_width = 0.8 / n_cond
    offset_start = -0.4 + bar_width / 2

    # Determine if labels contain New/Used (MERV chart)
    conditions_include_new_used = all(
        ("New" in c or "Used" in c) for c in unique_conditions
    )

    # If MERV-type chart: assign one colour per filter type, hatch for used
    filter_type_colors = {}
    if conditions_include_new_used:
        filter_types = []
        for cond in unique_conditions:
            ft = cond.replace(" New", "").replace(" Used", "")
            if ft not in filter_types:
                filter_types.append(ft)
        for i, ft in enumerate(filter_types):
            filter_type_colors[ft] = CONDITION_COLORS[i % len(CONDITION_COLORS)]

    for i, condition in enumerate(unique_conditions):
        # Collect rows for this condition
        cond_data = {
            "bin_label": [],
            "decay": [],
            "upper": [],
            "lower": [],
            "is_used": [],
        }
        for j in range(len(source_data["condition"])):
            if source_data["condition"][j] == condition:
                for key in cond_data:
                    cond_data[key].append(source_data[key][j])

        if not cond_data["bin_label"]:
            continue

        pos = offset_start + i * bar_width

        if conditions_include_new_used:
            # Color by filter type; hatch for used
            ft = condition.replace(" New", "").replace(" Used", "")
            color = filter_type_colors[ft]
            is_used_cond = "Used" in condition

            src = ColumnDataSource(data=cond_data)
            vbar_kwargs = {
                "x": dodge("bin_label", pos, range=p.x_range),
                "top": "decay",
                "width": bar_width * 0.85,
                "source": src,
                "color": color,
                "legend_label": condition,
            }
            if is_used_cond:
                vbar_kwargs["hatch_pattern"] = "right_diagonal_line"
                vbar_kwargs["hatch_color"] = "black"
            p.vbar(**vbar_kwargs)
            p.segment(
                x0=dodge("bin_label", pos, range=p.x_range),
                y0="lower",
                x1=dodge("bin_label", pos, range=p.x_range),
                y1="upper",
                source=src,
                line_color="black",
                line_width=1.5,
            )

        else:
            # Non-MERV chart: colour by condition, hatch for used burns
            color = CONDITION_COLORS[i % len(CONDITION_COLORS)]

            new_rows = {"bin_label": [], "decay": [], "upper": [], "lower": []}
            used_rows = {"bin_label": [], "decay": [], "upper": [], "lower": []}
            for k in range(len(cond_data["bin_label"])):
                target = used_rows if cond_data["is_used"][k] else new_rows
                for key in ("bin_label", "decay", "upper", "lower"):
                    target[key].append(cond_data[key][k])

            if new_rows["bin_label"]:
                src_new = ColumnDataSource(data=new_rows)
                p.vbar(
                    x=dodge("bin_label", pos, range=p.x_range),
                    top="decay",
                    width=bar_width * 0.85,
                    source=src_new,
                    color=color,
                    legend_label=condition,
                )
                p.segment(
                    x0=dodge("bin_label", pos, range=p.x_range),
                    y0="lower",
                    x1=dodge("bin_label", pos, range=p.x_range),
                    y1="upper",
                    source=src_new,
                    line_color="black",
                    line_width=1.5,
                )

            if used_rows["bin_label"]:
                src_used = ColumnDataSource(data=used_rows)
                vbar_kwargs = {
                    "x": dodge("bin_label", pos, range=p.x_range),
                    "top": "decay",
                    "width": bar_width * 0.85,
                    "source": src_used,
                    "color": color,
                    "hatch_pattern": "right_diagonal_line",
                    "hatch_color": "black",
                }
                if not new_rows["bin_label"]:
                    vbar_kwargs["legend_label"] = condition
                p.vbar(**vbar_kwargs)
                p.segment(
                    x0=dodge("bin_label", pos, range=p.x_range),
                    y0="lower",
                    x1=dodge("bin_label", pos, range=p.x_range),
                    y1="upper",
                    source=src_used,
                    line_color="black",
                    line_width=1.5,
                )

    # Axes and formatting
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = "lightgray"
    p.ygrid.grid_line_alpha = 0.6
    p.y_range = Range1d(0, 2)

    p.yaxis.axis_label = (
        "Baseline-Corrected Decay Rate per Portable Air Cleaner (h\u207b\u00b9)"
    )
    p.yaxis.formatter = NumeralTickFormatter(format="0.0")
    p.xaxis.axis_label = "SMPS Particle Size Bin"
    p.xaxis.major_label_orientation = "horizontal"

    # Text formatting
    for axis in (p.xaxis, p.yaxis):
        axis.axis_label_text_font_size = TEXT_CONFIG["axis_label_font_size"]
        axis.axis_label_text_font_style = TEXT_CONFIG["plot_font_style"]
        axis.major_label_text_font_size = TEXT_CONFIG["axis_tick_font_size"]
        axis.major_label_text_font_style = TEXT_CONFIG["font_style"]

    p.title.text_font_size = TEXT_CONFIG["title_font_size"]
    p.title.text_font_style = TEXT_CONFIG["plot_font_style"]

    p.legend.location = "top_right"
    p.legend.orientation = "vertical"
    p.legend.background_fill_alpha = 0.6
    p.legend.label_text_font_size = TEXT_CONFIG["legend_font_size"]
    p.legend.label_text_font_style = TEXT_CONFIG["font_style"]
    p.legend.spacing = 1
    p.legend.padding = 5
    p.legend.margin = 5

    # Metadata div
    detailed_text = "<br>".join(detailed_info)
    div_html = (
        f'<div style="font-size: {TEXT_CONFIG["font_size"]}; '
        f"font-weight: {TEXT_CONFIG['html_font_weight']}; "
        f'font-style: {TEXT_CONFIG["font_style"]};">'
        f"<b>Detailed Values:</b><br>{detailed_text}<br><br>"
        f"Baseline-corrected decay rates (burn1 subtracted from all measurements)<br>"
        f"Solid bars: New filters, Hatched bars: Used filters<br><br>"
        f"{script_metadata}"
        f"</div>"
    )
    text_div = Div(text=div_html, width=900)

    return column(p, text_div)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("SMPS Size-Bin Barchart — loading data...")

    smps_data = load_smps_decay_data()
    baselines = calculate_baselines(smps_data)

    print("Baselines (burn1 decay rates):")
    for bin_col, val in baselines.items():
        print(f"  {BIN_DISPLAY_LABELS[bin_col]}: {val:.4f} h\u207b\u00b9")

    metadata = get_script_metadata()
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    chart_configs = [
        {
            "burns": ["burn4", "burn9", "burn2"],
            "labels": {
                "burn4": "1 Air Cleaner",
                "burn9": "2 Air Cleaners",
                "burn2": "4 Air Cleaners",
            },
            "title": "SMPS: Baseline-Corrected Decay Rate by Number of Portable Air Cleaners",
            "filename": "SMPS_size_bin_bar_chart_filter_count.html",
        },
        {
            "burns": ["burn4", "burn3", "burn9", "burn10"],
            "labels": {
                "burn4": "1 New Air Cleaner",
                "burn3": "1 Used Air Cleaner",
                "burn9": "2 New Air Cleaners",
                "burn10": "2 Used Air Cleaners",
            },
            "title": "SMPS: Baseline-Corrected Decay Rate — New vs. Used Filters",
            "filename": "SMPS_size_bin_bar_chart_new_vs_used.html",
        },
        {
            "burns": ["burn7", "burn8", "burn9", "burn10"],
            "labels": {
                "burn7": "MERV 12A New",
                "burn8": "MERV 12A Used",
                "burn9": "MERV 13 New",
                "burn10": "MERV 13 Used",
            },
            "title": "SMPS: Baseline-Corrected Decay Rate — MERV 12A vs. MERV 13",
            "filename": "SMPS_size_bin_bar_chart_merv_comparison.html",
        },
    ]

    for config in chart_configs:
        print(f"\nGenerating {config['filename']}...")
        corrected = apply_baseline_correction(smps_data, baselines, config["burns"])
        chart = create_smps_bin_barchart(corrected, config, metadata)

        out_path = os.path.join(OUTPUT_PATH, config["filename"])
        output_file(out_path)
        save(chart)
        print(f"  Saved to: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
