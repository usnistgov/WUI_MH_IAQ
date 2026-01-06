#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic Script for Spatial Variation Analysis

This script helps diagnose why there are n/a values in the spatial variation analysis.
It checks data availability, time alignment, and potential issues.

Author: Nathan Lima
Date: 2025-12-23
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add repository root to path
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

from src.data_paths import get_data_root, get_common_file, get_instrument_path
from scripts.data_loaders import load_burn_log

# Load data
data_root = get_data_root()
burn_log = load_burn_log(get_common_file('burn_log'))

print("=" * 80)
print("SPATIAL VARIATION ANALYSIS DIAGNOSTIC")
print("=" * 80)

# Check 1: Burn log data
print("\n1. BURN LOG CHECK")
print("-" * 40)
print(f"Total burns: {len(burn_log)}")
print(f"Burn IDs: {burn_log['Burn ID'].tolist()}")

# Check for CR Box usage
cr_box_burns = burn_log[burn_log['CR Box on'].notna() & (burn_log['CR Box on'] != 'n/a')]
print(f"\nBurns with CR Box: {len(cr_box_burns)}")
print(f"CR Box burn IDs: {cr_box_burns['Burn ID'].tolist()}")

no_cr_box = burn_log[burn_log['CR Box on'].isna() | (burn_log['CR Box on'] == 'n/a')]
print(f"\nBurns WITHOUT CR Box: {len(no_cr_box)}")
print(f"No CR Box IDs: {no_cr_box['Burn ID'].tolist()}")

# Check 2: Peak concentration data
print("\n2. PEAK CONCENTRATION DATA CHECK")
print("-" * 40)

peak_file_path = data_root / "burn_data" / "peak_concentrations.xlsx"
if peak_file_path.exists():
    peak_data = pd.read_excel(peak_file_path)
    print(f"Peak data loaded: {peak_data.shape}")
    print(f"Columns: {peak_data.columns.tolist()}")
    print(f"\nBurns in peak data: {peak_data['Burn_ID'].unique().tolist()}")

    # Check for missing data
    print("\nMissing values per column:")
    for col in peak_data.columns:
        if col != 'Burn_ID':
            missing = peak_data[col].isna().sum()
            if missing > 0:
                print(f"  {col}: {missing}/{len(peak_data)} missing")
else:
    print(f"ERROR: Peak concentration file not found at {peak_file_path}")

# Check 3: AeroTrak data availability
print("\n3. AEROTRAK DATA CHECK")
print("-" * 40)

try:
    aerotrakb_file = get_instrument_path('aerotrak_bedroom') / "all_data.xlsx"
    aerotrakk_file = get_instrument_path('aerotrak_kitchen') / "all_data.xlsx"

    aerotrakb_data = pd.read_excel(aerotrakb_file)
    aerotrakk_data = pd.read_excel(aerotrakk_file)

    print(f"AeroTrakB data: {aerotrakb_data.shape}")
    print(f"AeroTrakK data: {aerotrakk_data.shape}")

    # Check for Date column
    aerotrakb_data['Date'] = pd.to_datetime(aerotrakb_data['Date and Time']).dt.date
    aerotrakk_data['Date'] = pd.to_datetime(aerotrakk_data['Date and Time']).dt.date

    print(f"\nUnique dates in AeroTrakB: {aerotrakb_data['Date'].nunique()}")
    print(f"Unique dates in AeroTrakK: {aerotrakk_data['Date'].nunique()}")

    # Match with burn dates
    burn_dates = pd.to_datetime(burn_log['Date']).dt.date
    aerotrakb_burn_dates = aerotrakb_data['Date'].isin(burn_dates).sum()
    aerotrakk_burn_dates = aerotrakk_data['Date'].isin(burn_dates).sum()

    print(f"\nAeroTrakB rows matching burn dates: {aerotrakb_burn_dates}/{len(aerotrakb_data)}")
    print(f"AeroTrakK rows matching burn dates: {aerotrakk_burn_dates}/{len(aerotrakk_data)}")

    # Check for PM columns
    aerotrakb_pm = [col for col in aerotrakb_data.columns if 'PM' in col and '(µg/m³)' in col]
    aerotrakk_pm = [col for col in aerotrakk_data.columns if 'PM' in col and '(µg/m³)' in col]

    print(f"\nAeroTrakB PM columns: {len(aerotrakb_pm)}")
    print(f"  {aerotrakb_pm}")
    print(f"AeroTrakK PM columns: {len(aerotrakk_pm)}")
    print(f"  {aerotrakk_pm}")

except Exception as e:
    print(f"ERROR loading AeroTrak data: {e}")

# Check 4: QuantAQ data availability
print("\n4. QUANTAQ DATA CHECK")
print("-" * 40)

try:
    quantaqb_file = get_instrument_path('quantaq_bedroom') / "MOD-PM-00194-b0fc215029fa4852b926bc50b28fda5a.csv"
    quantaqk_file = get_instrument_path('quantaq_kitchen') / "MOD-PM-00197-a6dd467a147a4d95a7b98a8a10ab4ea3.csv"

    quantaqb_data = pd.read_csv(quantaqb_file)
    quantaqk_data = pd.read_csv(quantaqk_file)

    print(f"QuantAQB data: {quantaqb_data.shape}")
    print(f"QuantAQK data: {quantaqk_data.shape}")

    # Check columns
    print(f"\nQuantAQB columns: {quantaqb_data.columns.tolist()[:10]}...")
    print(f"QuantAQK columns: {quantaqk_data.columns.tolist()[:10]}...")

    # Check for PM columns
    has_pm1_b = 'pm1' in quantaqb_data.columns or 'PM1 (µg/m³)' in quantaqb_data.columns
    has_pm25_b = 'pm25' in quantaqb_data.columns or 'PM2.5 (µg/m³)' in quantaqb_data.columns
    has_pm10_b = 'pm10' in quantaqb_data.columns or 'PM10 (µg/m³)' in quantaqb_data.columns

    print(f"\nQuantAQB has PM1: {has_pm1_b}")
    print(f"QuantAQB has PM2.5: {has_pm25_b}")
    print(f"QuantAQB has PM10: {has_pm10_b}")

except Exception as e:
    print(f"ERROR loading QuantAQ data: {e}")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
print("\nRecommendations:")
print("1. If CR Box burns are missing, those burns will show as 'skipped'")
print("2. If peak data is missing for some burns, Peak_Ratio_Index will be n/a")
print("3. If date alignment fails, all ratios will be n/a")
print("4. Check that 'Time Since Garage Closed (hours)' column exists in processed data")
print("5. QuantAQ only processes burns 4-10, so burns 1-3 won't have QuantAQ data")
