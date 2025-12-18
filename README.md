# Manufactured Housing Wildland-Urban Interface (WUI) Fire Smoke Study

## Project Overview

This repository contains Python analysis scripts for evaluating indoor air quality during simulated wildfire smoke infiltration events in a manufactured home test structure. The research investigates the effectiveness of various mitigation strategies, including portable air cleaners, HVAC filtration systems, and compartmentalization approaches for protecting indoor environments from wildfire smoke.

**Research Focus:**
- Particulate matter (PM) concentration dynamics during smoke infiltration events
- Clean Air Delivery Rate (CADR) calculations for air cleaning devices
- Spatial variation in PM concentrations across different rooms
- Particle size distribution analysis (0.3 μm to 25 μm)
- Decay rate characterization for different mitigation strategies
- Filter performance evaluation (MERV 12A and MERV 13)
- Compartmentalization strategy comparison

## Experimental Design

The study conducted multiple controlled burn experiments (Burn 1-10) in a manufactured home test structure with the following instrumentation:

**Monitoring Instruments:**
- **AeroTrak 9510** - Optical particle counters in bedroom and kitchen/morning room
- **QuantAQ MODULAIR-PM** - Low-cost PM sensors in multiple locations
- **TSI DustTrak** - Real-time aerosol monitor
- **TSI SMPS (Scanning Mobility Particle Sizer)** - Ultrafine particle size distribution (9-437 nm)
- **PurpleAir** - Community-grade PM sensors
- Temperature and relative humidity sensors

**Measured Pollutants:**
- PM₀.₅, PM₁, PM₂.₅, PM₃, PM₄, PM₅, PM₁₀, PM₁₅, PM₂₅ (μg/m³)
- Particle number concentrations (#/cm³)
- Particle size distributions

**Mitigation Strategies Tested:**
- Portable air cleaners with MERV filtration (various CADR ratings)
- Central HVAC systems with MERV 12A and MERV 13 filters
- Room compartmentalization (closed bedroom with/without filtration)
- Multiple filter configurations (1 and 2 filters analyzed; 4 filter configuration excluded due to data quality issues)

## Repository Structure

All analysis scripts are located in the `src/` directory and organized by analysis type:

### Clean Air Delivery Rate (CADR) Analysis
- **`clean_air_delivery_rates_update.py`** - Primary CADR calculation script with exponential decay fitting for all instruments
- **`clean_air_delivery_rates_barchart.py`** - Visualization of CADR values across different burn experiments
- **`clean_air_delivery_rates_pmsizes_SIUniformaty.py`** - CADR analysis with SI unit formatting and uniformity checks
- **`clean_air_delivery_rates_vs_total_surface_area.py`** - Correlation analysis between CADR and particle surface area
- **`cadr_comparison_statistical_analysis.py`** - Statistical comparison of CADR values across experimental conditions

### Compartmentalization and Mitigation Strategy Analysis
- **`compartmentalization_strategy_comparison.py`** - Comprehensive comparison of compartmentalization approaches (open house vs. closed bedroom vs. closed bedroom with filtration)
- **`decay_rate_barchart.py`** - Visualization and comparison of decay rates for different mitigation strategies

### Concentration Dynamics
- **`conc_increase_to_decrease.py`** - Analysis of concentration increase phase versus decay phase
- **`peak_concentration_script.py`** - Identification and characterization of peak PM concentrations during burns

### Spatial Variation Analysis
- **`spatial_variation_analysis.py`** - Quantification of spatial variability in PM concentrations between rooms (Peak Ratio Index, Average Ratio, RSD calculations)
- **`spatial_variation_analysis_plot.py`** - Interactive Bokeh visualizations of spatial variation metrics comparing bedroom vs morning room under different CR Box configurations

### Instrument Comparison and Validation
- **`aerotrak_vs_smps.py`** - Comparison between AeroTrak optical particle counter and SMPS measurements
- **`dusttrak-rh_comparison.py`** - Analysis of DustTrak performance and relative humidity effects
- **`purpleair_comparison.py`** - Validation of low-cost PurpleAir sensors against reference instruments
- **`quantaq_pm2_5_burn8.py`** - Detailed QuantAQ sensor analysis for specific burn experiment
- **`general_particle_count_comparison.py`** - Cross-instrument particle count comparison
- **`aham_ac1_comparison.py`** - Comparison of AHAM AC-1 test environment smoke concentration standards with actual WUI fire measurements

### SMPS (Scanning Mobility Particle Sizer) Analysis
- **`smps_filterperformance.py`** - Filter performance evaluation using ultrafine particle measurements
- **`smps_finepm_comparison.py`** - Comparison of fine PM behavior across instruments
- **`smps_heatmap.py`** - Heatmap visualization of particle size distribution evolution
- **`smps_mass_vs_conc.py`** - Mass concentration versus number concentration analysis

### Environmental Parameters
- **`temp-rh_comparison.py`** - Temperature and relative humidity monitoring and correlation with PM measurements

### Data Processing Utilities
- **`remove_aerotrak_dup_data.py`** - Data cleaning script for removing duplicate AeroTrak timestamps
- **`mh_relay_control_log.py`** - Processing of relay control system logs for HVAC and filtration operation
- **`process_aerotrak_data.py`** - Core processing module for TSI AeroTrak particle counter data files (calculates mass and number concentrations, PM metrics)

### Publication Figures
- **`toc_figure_script.py`** - Generates single burn PM2.5 concentration figure for journal Table of Contents graphic (WUI implications paper)

## Dependencies

### Required Python Packages
```python
# Core data processing
pandas >= 1.3.0
numpy >= 1.20.0

# Statistical analysis
scipy >= 1.7.0

# Visualization
bokeh >= 2.4.0
matplotlib >= 3.3.0

# Excel file support
openpyxl >= 3.0.0

# Built-in packages (no installation required)
datetime
pathlib
json
os
sys
typing
```

### Installation
```bash
pip install pandas numpy scipy bokeh matplotlib openpyxl
```

## Getting Started

### Initial Setup

The repository uses a configuration-based system to locate data files. This allows scripts to run on different machines without code modifications.

1. **Install dependencies**
   ```bash
   pip install pandas numpy scipy bokeh matplotlib openpyxl
   ```

2. **Configure data paths**
   - Copy the configuration template:
     ```bash
     cp data_config.template.json data_config.json
     ```
   - Edit `data_config.json` with your local data paths:
     ```json
     {
       "machine_name": "YourMachineName",
       "data_root": "C:/path/to/your/WUI_smoke",
       "instruments": {
         "aerotrak_bedroom": {
           "path": "C:/path/to/your/WUI_smoke/burn_data/aerotraks/bedroom2"
         },
         ...
       }
     }
     ```
   - See `data_config.template.json` for the complete structure with all instruments

3. **Verify configuration**
   ```bash
   python -c "from src.data_paths import resolver; resolver.list_instruments()"
   ```
   This will display all configured instruments and verify that data files are accessible.

**Note:** The `data_config.json` file is not tracked by git (it's in `.gitignore`) to keep your local file paths private. Each user maintains their own configuration file.

### Data Structure

Ensure your data files are organized as follows:
```
WUI_smoke/
├── burn_log.xlsx                                    # Master burn experiment log
├── burn_data/
│   ├── aerotraks/
│   │   ├── bedroom2/                                # AeroTrak bedroom location
│   │   │   └── all_data.xlsx
│   │   └── kitchen/                                 # AeroTrak kitchen location
│   │       └── all_data.xlsx
│   ├── quantaq/
│   │   ├── MOD-PM-00194-*.csv                      # QuantAQ bedroom
│   │   └── MOD-PM-00197-*.csv                      # QuantAQ kitchen
│   ├── dusttrak/
│   │   └── *.xlsx
│   ├── smps/
│   │   └── *.txt
│   ├── purpleair/
│   │   └── *.csv
│   ├── miniams/
│   │   └── *.csv
│   ├── vaisala_th/                                  # Temperature/RH sensors
│   │   └── *.xlsx
│   └── relaycontrol/                                # HVAC relay logs
│       └── *.txt
├── burn_dates_decay_aerotraks_bedroom.xlsx
├── burn_dates_decay_aerotraks_kitchen.xlsx
├── burn_dates_decay_smps.xlsx
├── peak_concentrations_all_instruments.xlsx
├── spatial_variation_analysis.xlsx
└── Paper_figures/                                   # Output directory for plots
```

## Usage

### Running Analysis Scripts

All scripts are located in the `src/` directory. Scripts can be run from the command line or interactively in Jupyter/VS Code:

```bash
# Example: Run CADR analysis
cd NIST_wui_mh_iaq
python src/clean_air_delivery_rates_update.py

# Example: Run spatial variation analysis
python src/spatial_variation_analysis.py
python src/spatial_variation_analysis_plot.py
```

### Output

Most scripts generate interactive Bokeh HTML plots that are saved to your configured output directory (typically `Paper_figures/`). These HTML files can be opened in any web browser for interactive data exploration.

### Example: CADR Analysis

```python
# Select dataset to analyze
dataset = 'AeroTrakB'  # Options: 'AeroTrakB', 'AeroTrakK', 'QuantAQb', 'QuantAQk', 'DustTrakB', 'SMPS'

# The script will:
# 1. Load and synchronize timestamp data
# 2. Identify peak concentrations for each burn
# 3. Fit exponential decay curves to post-peak data
# 4. Calculate CADR values from decay rates
# 5. Generate interactive plots with uncertainty bands
# 6. Export results to Excel and HTML
```

### Key Parameters

Most analysis scripts share common configuration parameters:

- **`time_shift`**: Temporal synchronization offset between instruments (minutes)
- **`baseline_values`**: Background PM concentrations before burns
- **`fit_ranges`**: Time windows for exponential decay fitting
- **`process_pollutants`**: List of PM size fractions to analyze

## Data Processing Methods

### CADR Calculation
Clean Air Delivery Rate is calculated using first-order exponential decay analysis:

```
C(t) = C₀ × exp(-kt)

CADR = k × V

where:
- C(t) = PM concentration at time t
- C₀ = Initial (peak) concentration
- k = Decay rate constant (min⁻¹)
- V = Room volume (m³)
```

### Spatial Variation Metrics
- **Peak Ratio Index (PRI)**: Ratio of peak concentrations between locations
- **Average Ratio**: Time-averaged concentration ratio during decay period
- **Relative Standard Deviation (RSD)**: Coefficient of variation across locations

### Statistical Analysis
- Exponential curve fitting with `scipy.optimize.curve_fit`
- Uncertainty propagation for CADR calculations
- Multi-instrument data synchronization
- Baseline subtraction

## Visualization

All visualization is performed using **Bokeh**, producing interactive HTML plots with:
- Hover tooltips for data inspection
- Crosshair tools for precise value reading
- Zoomable and pannable plots
- Legend toggling for multi-series comparison
- Consistent color schemes across PM size fractions
- Uncertainty bands for fitted parameters

## Data Quality and Validation

The analysis pipeline includes:
- Duplicate data removal for AeroTrak instruments
- Temporal alignment across instruments (typical offsets: 0-5 minutes)
- Baseline concentration correction
- Special case handling for specific burns (e.g., burn6 custom decay time)
- Cross-instrument validation (AeroTrak vs. SMPS, QuantAQ vs. reference instruments)
- Relative humidity effects on optical measurements

## Citation

If you use this code or data in your research, please cite:

[Publication details to be added upon journal acceptance]

**Related Publications:**
- TBD

## Contributing

This repository is maintained for research reproducibility. For questions, issues, or collaboration inquiries:

**Principal Investigator:** Poppendieck, Dustin G. (Fed) <dustin.poppendieck@nist.gov>
**Data Analyst:** Nathan Lima  
**Institution:** National Institute of Standards and Technology (NIST)  
**Contact:** nathan.lima@nist.gov

## Acknowledgments

This research was conducted at the National Institute of Standards and Technology (NIST) as part of the Indoor Air Quality and Ventilation Group and Wildland-Urban Interface fire Group.

## Version History

- **v1.0** (2025) - Initial release with complete analysis pipeline for burns 1-10
- Data collection period: April - June 2024
- Analysis completed: 2025

---

**Last Updated:** December 2025
**Repository Maintainer:** Nathan Lima (NIST)
