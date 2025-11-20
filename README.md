# Manufactured Housing Wildland-Urban Interface (WUI) Fire Smoke Study

## Project Overview

This repository contains Python analysis scripts for evaluating indoor air quality during simulated wildfire smoke infiltration events in a manufactured home test structure. The research investigates the effectiveness of various mitigation strategies, including portable air cleaners, HVAC filtration systems, and compartmentalization approaches for protecting indoor environments from wildfire smoke.

**Research Focus:**
- Particulate matter (PM) concentration dynamics during smoke infiltration events
- Clean Air Delivery Rate (CADR) calculations for air cleaning devices
- Spatial variation in PM concentrations across different rooms
- Particle size distribution analysis (0.3 μm to 25 μm)
- Decay rate characterization for different mitigation strategies
- Filter performance evaluation (MERV 12 and MERV 13)
- Compartmentalization strategy comparison

## Experimental Design

The study conducted multiple controlled burn experiments (Burn 2-10) in a manufactured home test structure with the following instrumentation:

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
- Portable air cleaners with HEPA filtration (various CADR ratings)
- Central HVAC systems with MERV 12 and MERV 13 filters
- Room compartmentalization (closed bedroom with/without filtration)
- Multiple filter configurations (1, 2, and 4 filters)

## Repository Structure

### Clean Air Delivery Rate (CADR) Analysis
- **`wui_clean_air_delivery_rates_update.py`** - Primary CADR calculation script with exponential decay fitting for all instruments
- **`wui_clean_air_delivery_rates_barchart.py`** - Visualization of CADR values across different burn experiments
- **`wui_clean_air_delivery_rates_pmsizes_SIUniformaty.py`** - CADR analysis with SI unit formatting and uniformity checks
- **`wui_clean_air_delivery_rates_vs_total_surface_area.py`** - Correlation analysis between CADR and particle surface area
- **`cadr_comparison_statistical_analysis.py`** - Statistical comparison of CADR values across experimental conditions

### Compartmentalization and Mitigation Strategy Analysis
- **`wui_compartmentalization_strategy_comparison.py`** - Comprehensive comparison of compartmentalization approaches (open house vs. closed bedroom vs. closed bedroom with filtration)
- **`wui_decay_rate_barchart.py`** - Visualization and comparison of decay rates for different mitigation strategies

### Concentration Dynamics
- **`wui_conc_increase_to_decrease.py`** - Analysis of concentration increase phase versus decay phase
- **`peak_concentration_script.py`** - Identification and characterization of peak PM concentrations during burns

### Spatial Variation Analysis
- **`wui_spatial_variation_analysis.py`** - Quantification of spatial variability in PM concentrations between rooms (Peak Ratio Index, Average Ratio, RSD calculations)

### Instrument Comparison and Validation
- **`wui_aerotrak_vs_smps.py`** - Comparison between AeroTrak optical particle counter and SMPS measurements
- **`wui_dusttrak-rh_comparison.py`** - Analysis of DustTrak performance and relative humidity effects
- **`wui_purpleair_comparison.py`** - Validation of low-cost PurpleAir sensors against reference instruments
- **`wui_quantaq_pm2_5_burn8.py`** - Detailed QuantAQ sensor analysis for specific burn experiment
- **`wui_general_particle_count_comparison.py`** - Cross-instrument particle count comparison

### SMPS (Scanning Mobility Particle Sizer) Analysis
- **`wui_smps_filterperformance.py`** - Filter performance evaluation using ultrafine particle measurements
- **`wui_smps_finepm_comparison.py`** - Comparison of fine PM behavior across instruments
- **`wui_smps_heatmap.py`** - Heatmap visualization of particle size distribution evolution
- **`wui_smps_mass_vs_conc.py`** - Mass concentration versus number concentration analysis

### Environmental Parameters
- **`wui_temp-rh_comparison.py`** - Temperature and relative humidity monitoring and correlation with PM measurements

### Data Processing Utilities
- **`wui_remove_aerotrak_dup_data.py`** - Data cleaning script for removing duplicate AeroTrak timestamps
- **`wui_mh_relay_control_log.py`** - Processing of relay control system logs for HVAC and filtration operation
- **`wui_figure_testing.py`** - Script for testing and refining figure formatting and layout
- **`wui_heatmap_testing.py`** - Development script for heatmap visualization approaches

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

# Date/time handling
datetime
```

## Usage

### General Workflow

1. **Set Working Directory**: Update the `absolute_path` variable in each script to point to your data directory:
```python
absolute_path = 'C:/path/to/your/WUI_smoke/'
```

2. **Data Structure**: Ensure your data files are organized as follows:
```
WUI_smoke/
├── burn_log.xlsx                    # Master burn experiment log
├── burn_data/
│   ├── aerotraks/
│   │   ├── bedroom2/all_data.xlsx
│   │   └── kitchen/all_data.xlsx
│   ├── quantaq/
│   │   ├── MOD-PM-00194-[id].csv  # Bedroom sensor
│   │   └── MOD-PM-00197-[id].csv  # Kitchen sensor
│   ├── dusttrak/
│   ├── smps/
│   └── purpleair/
├── burn_dates_decay_aerotraks_bedroom.xlsx
├── burn_dates_decay_aerotraks_kitchen.xlsx
└── burn_dates_decay_smps.xlsx
```

3. **Run Analysis Scripts**: Scripts can be run interactively in Jupyter/VS Code or as standalone Python scripts
```bash
python wui_clean_air_delivery_rates_update.py
```

4. **Output**: Most scripts generate interactive Bokeh HTML plots saved to `./Paper_figures/` directory

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
- Baseline subtraction and drift correction

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

## Key Findings Applications

These analysis scripts support research findings related to:
- Effectiveness of portable air cleaners in reducing wildfire smoke PM
- Impact of HVAC filtration on whole-house air quality
- Benefits of room compartmentalization during smoke events
- Particle size-dependent filtration efficiency
- Temporal dynamics of PM infiltration and removal
- Low-cost sensor performance for community monitoring

## Citation

If you use this code or data in your research, please cite:

[Publication details to be added upon journal acceptance]

**Related Publications:**
- NIST Technical Note series on manufactured housing and wildfire smoke
- Indoor Air journal submissions (in preparation)

## Contributing

This repository is maintained for research reproducibility. For questions, issues, or collaboration inquiries:

**Principal Investigator:** [Name]  
**Data Analyst:** Nathan Lima  
**Institution:** National Institute of Standards and Technology (NIST)  
**Contact:** [email]

## License

[Specify license - typically public domain for NIST work or appropriate open-source license]

## Acknowledgments

This research was conducted at the National Institute of Standards and Technology (NIST) as part of the Wildland-Urban Interface fire research program. The authors acknowledge support from [funding sources] and thank [collaborators].

## Version History

- **v1.0** (2025) - Initial release with complete analysis pipeline for burns 2-10
- Data collection period: [dates]
- Analysis completed: 2025

## Related Resources

- [NIST WUI Fire Research Program](https://www.nist.gov/el/fire-research-division-73300/wildland-urban-interface-wui-fire-research)
- [EPA Air Sensor Toolbox](https://www.epa.gov/air-sensor-toolbox)
- [ASHRAE Standards for Indoor Air Quality](https://www.ashrae.org/)

## Technical Notes

### Known Issues
- AeroTrak time synchronization requires manual adjustment per instrument
- SMPS data requires external processing before import
- Some burn experiments have incomplete data due to instrument malfunctions
- Bokeh plot rendering may be slow for datasets with >100,000 points

### Troubleshooting
- **Import errors**: Ensure all dependencies are installed in the active Python environment
- **File path errors**: Check that `absolute_path` is set correctly for your system
- **Missing data**: Verify that all required Excel/CSV files are in the correct directories
- **Plot display issues**: Try using `output_notebook()` for Jupyter or `output_file()` for standalone HTML

---

**Last Updated:** November 2025  
**Repository Maintainer:** Nathan Lima (NIST)
