# WUI Research Data Documentation

## Overview

This directory structure mirrors your local data organization by instrument type. The actual data files are stored on your local machine and configured via `data_config.json` in the repository root.

**IMPORTANT:** This repository does NOT contain actual data files. Data stays local on each researcher's machine.

## Instrument Data Organization

### AeroTrak 9510 Optical Particle Counter

**Bedroom Location (`aerotrak_bedroom`):**
- **Location:** Configured in `data_config.json` → `instruments.aerotrak_bedroom.path`
- **File Format:** Excel (.xlsx)
- **File Name:** `all_data.xlsx`
- **Columns:** Date and Time, PM0.5, PM1, PM3, PM5, PM10, PM25 (µg/m³)
- **Sampling Rate:** 1-minute intervals
- **Source:** TSI AeroTrak 9510 (bedroom2 location)
- **Time Shift:** +2.16 minutes (for synchronization)
- **Special Cases:** Burn 3 uses 5-minute rolling average
- **Baseline Values:** Pre-configured for each PM size

**Kitchen Location (`aerotrak_kitchen`):**
- **Location:** Configured in `data_config.json` → `instruments.aerotrak_kitchen.path`
- **File Format:** Excel (.xlsx)
- **File Name:** `all_data.xlsx`
- **Columns:** Date and Time, PM0.5, PM1, PM3, PM5, PM10, PM25 (µg/m³)
- **Sampling Rate:** 1-minute intervals
- **Source:** TSI AeroTrak 9510 (kitchen/morning room location)
- **Time Shift:** +5 minutes (for synchronization)
- **Baseline Calculation:** Weighted average from Burn 5 and Burn 6

### SMPS (Scanning Mobility Particle Sizer)

- **Location:** Configured in `data_config.json` → `instruments.smps.path`
- **File Format:** Excel (.xlsx)
- **File Name Pattern:** `MH_apollo_bed_MMDDYYYY_MassConc.xlsx`
- **Size Range:** 9.47-414.2 nm (ultrafine particles)
- **Columns:** Date, Start Time, size bins, Total Concentration (µg/m³)
- **Sampling Rate:** 5-minute scans
- **Source:** TSI SMPS 3938 (bedroom location)
- **Time Shift:** 0 minutes
- **Special Cases:** Burn 6 has custom decay time window
- **Size Bins:** 9-100nm, 100-200nm, 200-300nm, 300-414nm

### DustTrak DRX Aerosol Monitor

- **Location:** Configured in `data_config.json` → `instruments.dusttrak.path`
- **File Format:** Excel (.xlsx)
- **File Name:** `all_data.xlsx`
- **Parameters:** PM1, PM2.5, PM4, PM10, PM15 (µg/m³)
- **Sampling Rate:** 1-second logging
- **Source:** TSI DustTrak DRX 8533
- **Location:** Bedroom (Burns 1-6), Kitchen (Burns 7-10)
- **Time Shift:** +7 minutes
- **Notes:** PM15 = TOTAL mass concentration
- **Special Cases:** Burn 6 has custom decay time window

### PurpleAir Sensors

- **Location:** Configured in `data_config.json` → `instruments.purpleair.path`
- **File Format:** Excel (.xlsx)
- **File Name:** `garage-kitchen.xlsx`
- **Sheet Name:** `(P2)kitchen`
- **Parameters:** PM2.5 (µg/m³), Temperature, Humidity
- **Sampling Rate:** 2-minute averages
- **Source:** PurpleAir PA-II-SD (kitchen/garage area)
- **Burns:** 6-10 only
- **Time Shift:** 0 minutes
- **Notes:** Dual laser sensors (A & B channels), "Average" column used

### QuantAQ MODULAIR-PM Air Quality Monitor

**Bedroom Location (`quantaq_bedroom`):**
- **Location:** Configured in `data_config.json` → `instruments.quantaq_bedroom.path`
- **File Format:** CSV
- **File Name:** `MOD-PM-00194-b0fc215029fa4852b926bc50b28fda5a.csv`
- **Parameters:** PM1, PM2.5, PM10 (µg/m³)
- **Sampling Rate:** 1-minute intervals
- **Source:** QuantAQ MODULAIR-PM (unit 00194, bedroom2)
- **Burns:** 4-10 only
- **Time Shift:** -2.97 minutes
- **Special Cases:** Burn 6 has custom decay time window

**Kitchen Location (`quantaq_kitchen`):**
- **Location:** Configured in `data_config.json` → `instruments.quantaq_kitchen.path`
- **File Format:** CSV
- **File Name:** `MOD-PM-00197-a6dd467a147a4d95a7b98a8a10ab4ea3.csv`
- **Parameters:** PM1, PM2.5, PM10 (µg/m³)
- **Sampling Rate:** 1-minute intervals
- **Source:** QuantAQ MODULAIR-PM (unit 00197, kitchen)
- **Burns:** 4-10 only
- **Time Shift:** 0 minutes

### Mini-AMS (Aerosol Mass Spectrometer)

- **Location:** Configured in `data_config.json` → `instruments.miniams.path`
- **File Format:** Excel (.xlsx)
- **File Name:** `WUI_AMS_Species.xlsx`
- **Parameters:** Organic, Nitrate, Sulfate, Ammonium, Chloride (µg/m³)
- **Sampling Rate:** Variable
- **Source:** Mini-AMS (bedroom location)
- **Burns:** 1-3 only
- **Time Shift:** 0 minutes
- **Notes:** Chemical speciation of PM

### Vaisala Temperature and Relative Humidity Sensors

- **Location:** Configured in `data_config.json` → `instruments.vaisala_th.path`
- **File Format:** Excel (.xlsx)
- **File Names:**
  - Burn 1: `20240421-MH_Task_Logger_Data.xlsx` (sheet: "T_RH")
  - Burns 2-9: `20240429-MH_Data_Processed.xlsx` (sheet: "Sheet1")
  - Burn 10: `20240531-MH_Data_Processed.xlsx` (sheet: "Sheet1")
- **Parameters:** Temperature (°C), Relative Humidity (%)
- **Locations:** Bedroom2, Kitchen, Morning Room, Living Room
- **Sampling Rate:** Variable
- **Source:** Vaisala HMT series sensors

## Common Files and Folders

### Burn Log
- **Location:** `common_folders.burn_log` in `data_config.json`
- **File Name:** `burn_log.xlsx`
- **Sheet:** "Sheet2"
- **Contains:** Burn ID, Date, garage closed time, CR Box on time, experimental notes

### Burn Dates and Decay Windows
- **AeroTrak Bedroom:** `burn_dates_decay_aerotraks_bedroom.xlsx`
- **AeroTrak Kitchen:** `burn_dates_decay_aerotraks_kitchen.xlsx`
- **SMPS:** `burn_dates_decay_smps.xlsx`
- **Purpose:** Pre-defined time windows for exponential decay fitting in CADR calculations

### Processed Data Files
- **Peak Concentrations:** `burn_data/peak_concentrations_all_instruments.xlsx`
  - Contains maximum PM concentrations for each burn and instrument
- **Spatial Variation:** `burn_data/spatial_variation_analysis.xlsx`
  - Contains spatial ratios between bedroom and kitchen locations

### Output Figures
- **Location:** `common_folders.output_figures` in `data_config.json`
- **Directory:** `Paper_figures/`
- **Contents:** Interactive Bokeh HTML plots, publication-ready figures

## Data Collection Campaigns

### Burn Experiments
- **Burn 1-10:** April - June 2024
- **Test Structure:** Manufactured home at NIST facility
- **Smoke Source:** Controlled wildfire fuel combustion in attached garage
- **Experimental Variables:**
  - Number of CR Box portable air cleaners (1, 2, or 4 units)
  - Compartmentalization strategies (open house vs. closed bedroom)
  - HVAC operation modes
  - Filter types (MERV 12A, MERV 13)

### Experimental Timeline
Each burn follows a standard protocol:
1. **Baseline Period:** Pre-burn background measurements
2. **Garage Closed:** Smoke injection begins (t=0 reference point)
3. **Peak Concentration:** Maximum PM reached
4. **CR Box Activation:** Portable air cleaners turned on
5. **Decay Period:** Exponential decay of PM concentrations (2-4 hours)

## File Naming Conventions

### AeroTrak Files
```
all_data.xlsx  (consolidated file for all burns)
```

### SMPS Files
```
MH_apollo_bed_MMDDYYYY_MassConc.xlsx
Example: MH_apollo_bed_04212024_MassConc.xlsx
```

### QuantAQ Files
```
MOD-PM-XXXXX-[device_id].csv
Example: MOD-PM-00194-b0fc215029fa4852b926bc50b28fda5a.csv
```

## Quality Control

### Data Flags and Status
- **AeroTrak:** Flow Status and Laser Status must be "OK" for valid data
- **DustTrak:** RH > 90% may cause measurement interference
- **PurpleAir:** A/B channel discrepancies flagged
- **SMPS:** Infinite and NaN values removed

### Known Issues
- **AeroTrak Duplicate Timestamps:** Use `wui_remove_aerotrak_dup_data.py` to clean
- **Burn 3 AeroTrak Noise:** 5-minute rolling average applied
- **Burn 6 Decay Window:** Custom decay end time due to early termination
- **Burns with 4 CR Boxes:** Excluded from spatial variation analysis due to data quality

### Time Synchronization
All instruments are time-shifted relative to the "garage closed" event:
- AeroTrak Bedroom: +2.16 minutes
- AeroTrak Kitchen: +5.0 minutes
- DustTrak: +7.0 minutes
- QuantAQ Bedroom: -2.97 minutes
- QuantAQ Kitchen: 0 minutes
- SMPS: 0 minutes
- Mini-AMS: 0 minutes
- PurpleAir: 0 minutes

## Data Access

**This repository does NOT contain data files.**

For data access:
1. **Lab members:** Configure your `data_config.json` with local data paths
2. **External researchers:** Contact Dustin Poppendieck <dustin.poppendieck@nist.gov> for data sharing agreements

## Metadata and Calibration

Additional supporting documentation:
- **Instrument specifications:** See main README.md
- **Calibration records:** Stored with raw data files
- **Experimental protocols:** Available upon request
- **QA/QC procedures:** Documented in analysis scripts

## References

1. [Publication TBD - WUI smoke infiltration methodology]
2. [Publication TBD - CADR measurements]
3. [Dataset DOI - if published to public repository]

---

**Data Collection:** April - June 2024
**Analysis Period:** 2024-2025
**Principal Investigator:** Dustin Poppendieck (NIST)
**Data Analyst:** Nathan Lima (NIST)
