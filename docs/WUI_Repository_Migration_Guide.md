# WUI Research Repository Migration Guide
## Windows-Ready Setup for Portable, Multi-Machine Data Access

---

## Table of Contents

1. [Overview](#overview)
2. [Current vs. Improved Structure](#current-vs-improved-structure)
3. [Step-by-Step Migration Guide](#step-by-step-migration-guide)
4. [Configuration Files](#configuration-files)
5. [Python Modules](#python-modules)
6. [Script Updates](#script-updates)
7. [Documentation](#documentation)
8. [Usage Examples](#usage-examples)
9. [Troubleshooting](#troubleshooting)
10. [Quick Reference](#quick-reference)

---

## Overview

This guide helps you migrate your existing WUI (Wildland-Urban Interface) fire research repository to use portable, machine-independent data paths. The system allows:

- ✓ **No hardcoded paths** - scripts work on any machine
- ✓ **Data stays local** - never accidentally committed to Git
- ✓ **Windows-friendly** - handles drive letters and Windows paths
- ✓ **Instrument-organized** - data grouped by instrument type
- ✓ **Clear error messages** - tells you exactly what's missing
- ✓ **Team-friendly** - others can use same code with their own data

### Core Concept: Config-Based Data Paths

We'll use a **config file approach** where:
- `data_config.template.json` is committed to Git (template only)
- `data_config.json` is created locally on each machine (ignored by Git)
- Scripts read paths from the config file automatically

---

## Current vs. Improved Structure

### Current (what you have):
```
project-name/
├── wui.yaml
├── scripts/ (23+ analysis scripts)
└── [hardcoded paths in each script]
```

### Improved (what we'll create):
```
project-name/
├── .gitignore
├── README.md
├── LICENSE
├── wui.yaml
├── requirements.txt
├── data_config.template.json     # NEW - Template for others
├── CITATION.md                   # NEW - How to cite
│
├── data/                         # NEW - Local data references only
│   ├── README.md
│   ├── aerotrak/
│   ├── smps/
│   ├── dusttrak/
│   ├── purpleair/
│   ├── quantaq/
│   └── [other instruments]/
│
├── scripts/ (your existing 23 scripts, updated)
│
├── src/                          # NEW - Reusable modules
│   ├── __init__.py
│   └── data_paths.py
│
├── results/                      # NEW - Organized outputs
│   ├── figures/
│   ├── tables/
│   └── output_data/
│
└── docs/                         # NEW - Documentation
    └── methodology.md
```

---

## Step-by-Step Migration Guide

### Step 1: Backup Your Current Repository

```powershell
# Navigate to your existing repo
cd path\to\your\project-name

# Create a backup branch
git checkout -b backup-before-migration
git push origin backup-before-migration

# Return to main branch
git checkout main
```

### Step 2: Create New Directory Structure

```powershell
# Create new directories
mkdir src, data, results, docs
mkdir results\figures, results\tables, results\output_data
mkdir data\aerotrak, data\smps, data\dusttrak, data\purpleair, data\quantaq

# Create placeholder files
type nul > src\__init__.py
type nul > data\README.md
type nul > docs\methodology.md
```

### Step 3: Create `.gitignore` (if not exists)

Create or update `.gitignore`:

```gitignore
# Data configuration (contains local paths - DO NOT COMMIT)
data_config.json

# Data directories (actual data stays local)
data/aerotrak/*
data/smps/*
data/dusttrak/*
data/purpleair/*
data/quantaq/*
!data/**/README.md
!data/**/.gitkeep

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Jupyter Notebook
.ipynb_checkpoints

# Results (optional - decide if you want to commit these)
results/figures/*.png
results/figures/*.html
results/figures/*.pdf
results/tables/*.csv
results/output_data/*
!results/**/.gitkeep

# IDE
.vscode/
.idea/
*.swp

# OS
Thumbs.db
Desktop.ini
$RECYCLE.BIN/

# Temporary files
*.tmp
~$*.xlsx
~$*.docx
```

---

## Configuration Files

### Step 4: Create Instrument-Based Data Config Template

Create `data_config.template.json`:

```json
{
  "description": "WUI Fire Research Data Path Configuration",
  "machine_name": "MACHINE_ID_HERE",
  "data_root": "PATH_TO_YOUR_DATA_ROOT",
  
  "instruments": {
    "aerotrak": {
      "path": "PATH_TO_AEROTRAK_DATA",
      "description": "AeroTrak particle counter data",
      "file_pattern": "*.csv"
    },
    "smps": {
      "path": "PATH_TO_SMPS_DATA",
      "description": "SMPS (Scanning Mobility Particle Sizer) data",
      "file_pattern": "*.txt"
    },
    "dusttrak": {
      "path": "PATH_TO_DUSTTRAK_DATA",
      "description": "DustTrak aerosol monitor data",
      "file_pattern": "*.csv"
    },
    "purpleair": {
      "path": "PATH_TO_PURPLEAIR_DATA",
      "description": "PurpleAir sensor data",
      "file_pattern": "*.csv"
    },
    "quantaq": {
      "path": "PATH_TO_QUANTAQ_DATA",
      "description": "QuantAQ air quality monitor data",
      "file_pattern": "*.csv"
    },
    "mh_relay": {
      "path": "PATH_TO_RELAY_LOGS",
      "description": "MH relay control logs",
      "file_pattern": "*.log"
    },
    "temp_rh": {
      "path": "PATH_TO_TEMP_RH_DATA",
      "description": "Temperature and relative humidity data",
      "file_pattern": "*.csv"
    }
  },
  
  "common_folders": {
    "calibration": "PATH_TO_CALIBRATION_FILES",
    "metadata": "PATH_TO_METADATA",
    "raw_backups": "PATH_TO_RAW_BACKUPS"
  },
  
  "notes": [
    "Use forward slashes (/) or double backslashes (\\\\) in Windows paths",
    "Example: D:/Research/WUI_Data/aerotrak or D:\\\\Research\\\\WUI_Data\\\\aerotrak",
    "Each instrument folder should contain all data files for that instrument"
  ]
}
```

---

## Python Modules

### Step 5: Create Path Resolution Module

Create `src/data_paths.py`:

```python
"""
Data path resolution for WUI fire research multi-instrument, multi-machine setup.
Handles instrument-specific data folders with multiple files per instrument.
"""
import json
from pathlib import Path
import socket
import sys
from typing import List, Optional

class WUIDataPathResolver:
    """Resolves data paths for WUI research instruments across machines."""
    
    def __init__(self, config_file='data_config.json'):
        self.repo_root = Path(__file__).parent.parent.resolve()
        self.config_file = self.repo_root / config_file
        self.machine_name = socket.gethostname()
        self.config = self._load_config()
        
    def _load_config(self):
        """Load configuration file with helpful error messages."""
        if not self.config_file.exists():
            error_msg = f"""
╔════════════════════════════════════════════════════════════════╗
║  ❌ DATA CONFIGURATION NOT FOUND                              ║
╚════════════════════════════════════════════════════════════════╝

Configuration file missing: {self.config_file}

📋 SETUP INSTRUCTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Copy the template to create your local config
    copy data_config.template.json data_config.json

Step 2: Edit data_config.json with YOUR machine's data paths
    notepad data_config.json

Step 3: Example configuration for this machine ({self.machine_name}):

{{
  "machine_name": "{self.machine_name}",
  "data_root": "D:/Research/WUI_Data",
  
  "instruments": {{
    "aerotrak": {{
      "path": "D:/Research/WUI_Data/AeroTrak",
      "description": "AeroTrak particle counter data",
      "file_pattern": "*.csv"
    }},
    "smps": {{
      "path": "D:/Research/WUI_Data/SMPS",
      "description": "SMPS data",
      "file_pattern": "*.txt"
    }},
    "dusttrak": {{
      "path": "D:/Research/WUI_Data/DustTrak",
      "description": "DustTrak data",
      "file_pattern": "*.csv"
    }}
  }}
}}

Step 4: Run setup verification
    python -c "from src.data_paths import resolver; resolver.list_instruments()"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            print(error_msg, file=sys.stderr)
            raise FileNotFoundError(error_msg)
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            print(f"✓ Loaded WUI data config for: {config.get('machine_name', 'unknown')}")
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {self.config_file}: {e}")
    
    def get_instrument_path(self, instrument: str) -> Path:
        """
        Get the data folder path for a specific instrument.
        
        Parameters:
        -----------
        instrument : str
            Instrument name (e.g., 'aerotrak', 'smps', 'dusttrak')
            
        Returns:
        --------
        pathlib.Path
            Resolved absolute path to instrument data folder
        """
        instruments = self.config.get('instruments', {})
        
        if instrument not in instruments:
            available = list(instruments.keys())
            raise KeyError(
                f"Instrument '{instrument}' not configured.\n"
                f"Available instruments: {available}\n"
                f"Update your data_config.json to add this instrument."
            )
        
        path_str = instruments[instrument]['path']
        path = Path(path_str).resolve()
        
        if not path.exists():
            print(f"⚠️  Warning: Instrument path does not exist: {path}", file=sys.stderr)
            print(f"    Create folder or update path in data_config.json", file=sys.stderr)
        
        return path
    
    def get_instrument_files(self, instrument: str, pattern: str = None) -> List[Path]:
        """
        Get all data files for an instrument.
        
        Parameters:
        -----------
        instrument : str
            Instrument name
        pattern : str, optional
            File pattern (e.g., '*.csv'). If None, uses config default.
            
        Returns:
        --------
        list of Path
            List of data files found
        """
        inst_path = self.get_instrument_path(instrument)
        
        if pattern is None:
            pattern = self.config['instruments'][instrument].get('file_pattern', '*.*')
        
        files = sorted(inst_path.glob(pattern))
        
        if not files:
            print(f"⚠️  No files found matching '{pattern}' in {inst_path}", file=sys.stderr)
        
        return files
    
    def get_common_folder(self, folder_name: str) -> Path:
        """
        Get path to common folders (calibration, metadata, etc.).
        
        Parameters:
        -----------
        folder_name : str
            Common folder name (e.g., 'calibration', 'metadata')
            
        Returns:
        --------
        pathlib.Path
            Resolved path
        """
        common = self.config.get('common_folders', {})
        
        if folder_name not in common:
            available = list(common.keys())
            raise KeyError(f"Common folder '{folder_name}' not found. Available: {available}")
        
        return Path(common[folder_name]).resolve()
    
    def list_instruments(self):
        """Print all configured instruments and their status."""
        print(f"\n╔════════════════════════════════════════════════════════════════╗")
        print(f"║  📊 WUI Research Data Configuration                            ║")
        print(f"╚════════════════════════════════════════════════════════════════╝")
        print(f"\nMachine: {self.machine_name}")
        print(f"Config:  {self.config_file}")
        
        instruments = self.config.get('instruments', {})
        if instruments:
            print(f"\n{'Instrument':<15} | Status | Files | Path")
            print("─" * 90)
            
            for name, info in instruments.items():
                path = Path(info['path'])
                exists = "✓" if path.exists() else "✗"
                
                if path.exists():
                    pattern = info.get('file_pattern', '*.*')
                    file_count = len(list(path.glob(pattern)))
                    file_info = f"{file_count:>5}"
                else:
                    file_info = "  N/A"
                
                print(f"{name:<15} | {exists:^6} | {file_info} | {path}")
        else:
            print("\n⚠️  No instruments configured!")
        
        common = self.config.get('common_folders', {})
        if common:
            print(f"\n{'Common Folder':<15} | Status | Path")
            print("─" * 90)
            for name, path_str in common.items():
                path = Path(path_str)
                exists = "✓" if path.exists() else "✗"
                print(f"{name:<15} | {exists:^6} | {path}")
        
        print()
    
    def get_data_root(self) -> Path:
        """Get the root data directory."""
        return Path(self.config.get('data_root', self.repo_root / 'data')).resolve()

# Convenience instance for easy importing
resolver = WUIDataPathResolver()

# Convenience functions
def get_instrument_path(instrument: str) -> Path:
    """Get data folder path for an instrument."""
    return resolver.get_instrument_path(instrument)

def get_instrument_files(instrument: str, pattern: str = None) -> List[Path]:
    """Get all data files for an instrument."""
    return resolver.get_instrument_files(instrument, pattern)

def get_common_folder(folder_name: str) -> Path:
    """Get path to common folder."""
    return resolver.get_common_folder(folder_name)
```

Update `src/__init__.py`:

```python
"""
WUI Fire Research Code Package
"""
from .data_paths import (
    get_instrument_path,
    get_instrument_files,
    get_common_folder,
    WUIDataPathResolver
)

__all__ = [
    'get_instrument_path',
    'get_instrument_files', 
    'get_common_folder',
    'WUIDataPathResolver'
]
```

---

## Script Updates

### Step 6: Update One Script as Example

Let's update `wui_smps_heatmap.py` as a working example:

**Before (with hardcoded paths):**
```python
import pandas as pd
from bokeh.plotting import figure, show

# Hardcoded path - only works on one machine!
data_file = "D:/Research/WUI/SMPS/smps_data_burn8.txt"
df = pd.read_csv(data_file)
# ... rest of script
```

**After (portable paths):**
```python
"""
SMPS Heatmap Visualization
Updated to use portable data path resolution
"""
import sys
from pathlib import Path
import pandas as pd
from bokeh.plotting import figure, save, output_file
from bokeh.models import LinearColorMapper, ColorBar
from bokeh.transform import transform
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data_paths import get_instrument_path, get_instrument_files

def main():
    print("="*70)
    print("SMPS Heatmap Generation")
    print("="*70)
    
    # Get SMPS data folder (works on any configured machine!)
    smps_path = get_instrument_path('smps')
    print(f"\n📂 SMPS data folder: {smps_path}")
    
    # Option 1: Get specific file
    data_file = smps_path / 'smps_data_burn8.txt'
    
    # Option 2: Get all SMPS files and select one
    # all_files = get_instrument_files('smps', '*.txt')
    # data_file = all_files[0]  # or select based on criteria
    
    if not data_file.exists():
        print(f"\n❌ Data file not found: {data_file}")
        print(f"Available files in {smps_path}:")
        for f in sorted(smps_path.glob('*.txt')):
            print(f"  - {f.name}")
        return
    
    print(f"✓ Loading: {data_file.name}")
    
    # Load and process data
    df = pd.read_csv(data_file, sep='\t')  # Adjust separator as needed
    print(f"✓ Loaded {len(df)} records")
    
    # [Your existing heatmap generation code here]
    # ... data processing ...
    # ... heatmap creation ...
    
    # Save to results directory
    output_dir = Path(__file__).parent.parent / 'results' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file(output_dir / 'smps_heatmap.html')
    
    # Create heatmap
    p = figure(
        title='SMPS Particle Size Distribution',
        x_axis_label='Time',
        y_axis_label='Particle Size (nm)',
        width=1200,
        height=600
    )
    
    # [Your plotting code]
    
    save(p)
    print(f"\n✓ Saved heatmap to: {output_dir / 'smps_heatmap.html'}")

if __name__ == '__main__':
    main()
```

### Step 7: Create Migration Script Template

Create `scripts/_TEMPLATE_migration.py` to help update other scripts:

```python
"""
TEMPLATE: How to update existing WUI scripts to use portable paths

BEFORE (hardcoded):
    data_file = "D:/Research/WUI/AeroTrak/data.csv"

AFTER (portable):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from data_paths import get_instrument_path
    
    aerotrak_path = get_instrument_path('aerotrak')
    data_file = aerotrak_path / 'data.csv'
"""

# Standard imports
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from bokeh.plotting import figure, save, output_file

# Add src to path for portable data access
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data_paths import get_instrument_path, get_instrument_files

def main():
    # Script header
    print("="*70)
    print("Script Name Here")
    print("="*70)
    
    # Get instrument data path
    instrument_path = get_instrument_path('aerotrak')  # or 'smps', 'dusttrak', etc.
    print(f"\n📂 Data folder: {instrument_path}")
    
    # Option 1: Get specific file
    data_file = instrument_path / 'specific_file.csv'
    
    # Option 2: Get all files matching pattern
    # all_files = get_instrument_files('aerotrak', '*.csv')
    # data_file = all_files[0]  # Select file
    
    # Option 3: Get files and filter
    # all_files = get_instrument_files('aerotrak')
    # data_file = [f for f in all_files if 'burn8' in f.name][0]
    
    if not data_file.exists():
        print(f"❌ File not found: {data_file}")
        return
    
    print(f"✓ Loading: {data_file.name}")
    df = pd.read_csv(data_file)
    
    # [Your analysis code here]
    
    # Save results to organized output directory
    output_dir = Path(__file__).parent.parent / 'results' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file(output_dir / 'output_name.html')
    # [Your plotting code]
    
    print(f"\n✓ Saved output to: {output_dir / 'output_name.html'}")

if __name__ == '__main__':
    main()
```

---

## Documentation

### Step 8: Update README.md

Create or replace `README.md`:

```markdown
# WUI Fire Research - Indoor Air Quality Analysis

Analysis code for Wildland-Urban Interface (WUI) fire smoke infiltration and indoor air quality research.

## 🔬 Research Overview

This repository contains analysis scripts for multi-instrument air quality monitoring during WUI fire simulation experiments, including:
- Particle size distribution analysis (SMPS)
- Particle counting (AeroTrak)
- PM2.5 monitoring (DustTrak, PurpleAir, QuantAQ)
- Clean Air Delivery Rate (CADR) calculations
- Spatial variation analysis
- Filter performance evaluation

## 🚀 Quick Start for Windows

### 1. Clone Repository

```powershell
git clone https://github.com/yourusername/project-name.git
cd project-name
```

### 2. Set Up Conda Environment

```powershell
# Create environment from specification
conda env create -f wui.yaml

# Activate environment
conda activate wui
```

### 3. Configure Data Paths ⚠️ **CRITICAL STEP**

Your data stays on your local machine - configure where it lives:

```powershell
# Copy template to create local config
copy data_config.template.json data_config.json

# Edit with your machine's data paths
notepad data_config.json
```

**Example `data_config.json` for Office PC:**
```json
{
  "machine_name": "OFFICE-PC",
  "data_root": "D:/Research/WUI_Data",
  
  "instruments": {
    "aerotrak": {
      "path": "D:/Research/WUI_Data/AeroTrak",
      "description": "AeroTrak particle counter data",
      "file_pattern": "*.csv"
    },
    "smps": {
      "path": "D:/Research/WUI_Data/SMPS",
      "description": "SMPS data files",
      "file_pattern": "*.txt"
    },
    "dusttrak": {
      "path": "D:/Research/WUI_Data/DustTrak",
      "description": "DustTrak PM monitor data",
      "file_pattern": "*.csv"
    },
    "purpleair": {
      "path": "D:/Research/WUI_Data/PurpleAir",
      "description": "PurpleAir sensor data",
      "file_pattern": "*.csv"
    },
    "quantaq": {
      "path": "D:/Research/WUI_Data/QuantAQ",
      "description": "QuantAQ monitor data",
      "file_pattern": "*.csv"
    }
  },
  
  "common_folders": {
    "calibration": "D:/Research/WUI_Data/Calibration",
    "metadata": "D:/Research/WUI_Data/Metadata"
  }
}
```

**Example `data_config.json` for Lab PC:**
```json
{
  "machine_name": "LAB-PC-01",
  "data_root": "E:/WUI_Research/Data",
  
  "instruments": {
    "aerotrak": {
      "path": "E:/WUI_Research/Data/AeroTrak_Files",
      "description": "AeroTrak data",
      "file_pattern": "*.csv"
    },
    "smps": {
      "path": "E:/WUI_Research/Data/SMPS_Files",
      "description": "SMPS data",
      "file_pattern": "*.txt"
    }
    // ... other instruments
  }
}
```

### 4. Verify Setup

```powershell
# Check that all instruments are configured correctly
python -c "from src.data_paths import resolver; resolver.list_instruments()"
```

You should see output like:
```
╔════════════════════════════════════════════════════════════════╗
║  📊 WUI Research Data Configuration                            ║
╚════════════════════════════════════════════════════════════════╝

Machine: YOUR-PC-NAME

Instrument      | Status | Files | Path
──────────────────────────────────────────────────────────────────────────────
aerotrak        |   ✓    |    45 | D:/Research/WUI_Data/AeroTrak
smps            |   ✓    |    12 | D:/Research/WUI_Data/SMPS
dusttrak        |   ✓    |    38 | D:/Research/WUI_Data/DustTrak
```

### 5. Run Analysis Scripts

```powershell
# Example: Run SMPS heatmap analysis
python scripts/wui_smps_heatmap.py

# Run CADR calculation
python scripts/wui_clean_air_delivery_rates_update.py

# Run spatial variation analysis
python scripts/wui_spatial_variation_analysis.py
```

## 📁 Repository Structure

```
project-name/
├── wui.yaml               # Conda environment
├── data_config.json       # YOUR local paths (not in Git)
├── scripts/               # Analysis scripts (23+)
│   ├── cadr_comparison_statistical_analysis.py
│   ├── wui_smps_heatmap.py
│   ├── wui_clean_air_delivery_rates_update.py
│   └── ...
├── src/                   # Reusable modules
│   └── data_paths.py     # Path resolution
├── results/               # Generated outputs
│   ├── figures/          # Bokeh plots, images
│   ├── tables/           # CSV results
│   └── output_data/      # Processed datasets
└── docs/                  # Documentation
```

## 📊 Available Analysis Scripts

### CADR (Clean Air Delivery Rate) Analysis
- `wui_clean_air_delivery_rates_update.py` - Main CADR calculations
- `wui_clean_air_delivery_rates_barchart.py` - CADR visualization
- `wui_clean_air_delivery_rates_vs_total_surface_area.py` - CADR vs filter area
- `cadr_comparison_statistical_analysis.py` - Statistical comparison

### Particle Size & Distribution
- `wui_smps_heatmap.py` - SMPS particle size distribution heatmap
- `wui_smps_filterperformance.py` - Filter performance by particle size
- `wui_smps_finepm_comparison.py` - Fine PM comparison
- `wui_smps_mass_vs_conc.py` - Mass concentration analysis

### Instrument Comparisons
- `wui_aerotrak_vs_smps.py` - AeroTrak vs SMPS comparison
- `wui_aham_ac1_comparison.py` - AHAM AC-1 standard comparison
- `wui_purpleair_comparison.py` - PurpleAir sensor validation
- `wui_dusttrak-rh_comparison.py` - DustTrak vs RH analysis
- `wui_temp-rh_comparison.py` - Temperature and RH correlation

### Data Processing
- `process_aerotrak_data.py` - AeroTrak data preprocessing
- `wui_remove_aerotrak_dup_data.py` - Remove duplicate records
- `peak_concentration_script.py` - Peak concentration identification

### Spatial & Temporal Analysis
- `wui_spatial_variation_analysis.py` - Multi-location analysis
- `wui_spatial_variation_analysis_plot.py` - Spatial visualization
- `wui_compartmentalization_strategy_comparison.py` - Room isolation effectiveness
- `wui_conc_increase_to_decrease.py` - Concentration dynamics

### Figures & Visualization
- `toc_figure_script.py` - Table of contents figure
- `wui_general_particle_count_comparison.py` - General particle trends
- `wui_decay_rate_barchart.py` - Decay rate visualization

### Other
- `wui_quantaq_pm2.5_burn8.py` - QuantAQ PM2.5 analysis for Burn 8
- `wui_mh_relay_control_log.py` - Mechanical system control logging

## 🔧 For Developers: Updating Scripts to Use Portable Paths

If you're updating old scripts or creating new ones, use this pattern:

```python
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data_paths import get_instrument_path, get_instrument_files

# Get instrument data folder
aerotrak_path = get_instrument_path('aerotrak')

# Get specific file
data_file = aerotrak_path / 'burn8_aerotrak.csv'

# Or get all files
all_files = get_instrument_files('aerotrak', '*.csv')

# Load data
df = pd.read_csv(data_file)
```

See `scripts/_TEMPLATE_migration.py` for full example.

## 📚 Data Organization

### Data Storage (Local Only - NOT in Git)

Your data should be organized by instrument:

```
Your_Local_Drive/
└── WUI_Data/
    ├── AeroTrak/
    │   ├── burn1_aerotrak.csv
    │   ├── burn2_aerotrak.csv
    │   └── ...
    ├── SMPS/
    │   ├── smps_burn1.txt
    │   ├── smps_burn2.txt
    │   └── ...
    ├── DustTrak/
    ├── PurpleAir/
    ├── QuantAQ/
    └── Calibration/
```

### Repository Data Folder (References Only)

The `data/` folder in the repository contains only:
- README files documenting data sources
- .gitkeep files to preserve folder structure
- NO actual data files

## 📝 Citation

If you use this code in your research, please cite:

[Add your citation details - see CITATION.md]

For WUI fire and IAQ research methodology:
[Add relevant paper citations]

## 📄 License

[MIT / GPL-3.0 / Apache 2.0 - specify in LICENSE file]

## 🤝 Collaboration

### For Lab Members

1. Clone repository
2. Configure `data_config.json` with YOUR machine's paths
3. Never commit `data_config.json` (it's git-ignored)
4. Always commit updates to `data_config.template.json` when adding new instruments

### For External Collaborators

Contact [your name/email] for:
- Data access requests
- Instrument specifications
- Calibration procedures
- Research collaboration

## ⚠️ Important Notes

- **Data stays local** - never committed to GitHub
- **Config stays private** - `data_config.json` is machine-specific
- **Template is shared** - `data_config.template.json` shows what's needed
- **Scripts are portable** - work on any machine with proper config

## 🐛 Troubleshooting

### "Configuration file not found"
```powershell
copy data_config.template.json data_config.json
notepad data_config.json
```

### "Instrument 'xyz' not configured"
Add the instrument to your `data_config.json`:
```json
"instruments": {
  "xyz": {
    "path": "D:/Your/Path/To/XYZ_Data",
    "description": "XYZ instrument data",
    "file_pattern": "*.csv"
  }
}
```

### "Path does not exist" warning
Check:
1. Drive letter is correct (D: vs E:)
2. Folder actually exists at that location
3. Path uses forward slashes or double backslashes

### Scripts can't find modules
Make sure you're in the conda environment:
```powershell
conda activate wui
```

---

**Research Contact:** [Your Name], [Your Institution]  
**Code Questions:** [Your Email] or open an issue on GitHub
```

### Create `data/README.md`

```markdown
# WUI Research Data Documentation

## Overview

This directory structure mirrors your local data organization by instrument type. The actual data files are stored on your local machine and configured via `data_config.json`.

## Instrument Data Organization

### AeroTrak Optical Particle Counter
- **Location:** Configured in `data_config.json` → `instruments.aerotrak.path`
- **File Format:** CSV
- **Columns:** Timestamp, 0.3µm, 0.5µm, 1.0µm, 5.0µm, 10.0µm counts
- **Sampling Rate:** 1-minute intervals
- **Source:** TSI AeroTrak 9306-V2
- **Calibration:** [Date, procedure, reference]

### SMPS (Scanning Mobility Particle Sizer)
- **Location:** Configured in `data_config.json` → `instruments.smps.path`
- **File Format:** Tab-delimited TXT
- **Size Range:** 10-500 nm
- **Columns:** Timestamp, size bins (dN/dlogDp)
- **Sampling Rate:** 5-minute scans
- **Source:** TSI SMPS 3938
- **Calibration:** [Date, procedure, reference]

### DustTrak Aerosol Monitor
- **Location:** Configured in `data_config.json` → `instruments.dusttrak.path`
- **File Format:** CSV
- **Parameters:** PM1, PM2.5, PM4, PM10, TSP (µg/m³)
- **Sampling Rate:** 1-second logging
- **Source:** TSI DustTrak DRX 8533
- **Calibration:** [Date, procedure, reference]

### PurpleAir Sensors
- **Location:** Configured in `data_config.json` → `instruments.purpleair.path`
- **File Format:** CSV
- **Parameters:** PM1, PM2.5, PM10, Temperature, Humidity
- **Sampling Rate:** 2-minute averages
- **Source:** PurpleAir PA-II-SD
- **Notes:** Dual laser sensors (A & B channels)

### QuantAQ Air Quality Monitor
- **Location:** Configured in `data_config.json` → `instruments.quantaq.path`
- **File Format:** CSV
- **Parameters:** PM2.5, CO, NO2, O3, Temperature, RH
- **Sampling Rate:** 1-minute intervals
- **Source:** QuantAQ MODULAIR-PM
- **Calibration:** Factory calibrated [date]

## Data Collection Campaigns

### Burn Experiments
- **Burn 1-10:** Conducted [dates]
- **Burn Configuration:** [describe setup]
- **Instrumentation Layout:** [describe positions]
- **Reference:** [Paper, report, or documentation]

## File Naming Conventions

```
[instrument]_burn[number]_[location]_[date].csv

Examples:
- aerotrak_burn8_bedroom_20240315.csv
- smps_burn8_livingroom_20240315.txt
- dusttrak_burn1_outdoor_20240201.csv
```

## Quality Control

### Data Flags
- **0:** Valid data
- **1:** Out of range (sensor limit)
- **2:** Sensor malfunction
- **3:** Calibration/maintenance period
- **9:** Missing data

### Known Issues
- AeroTrak duplicate timestamp issue (use `wui_remove_aerotrak_dup_data.py`)
- DustTrak RH interference above 90% RH
- PurpleAir A/B channel discrepancies [specify affected burns]

## Metadata & Calibration

Additional supporting files:
- Sensor specifications: `docs/instrument_specs.md`
- Calibration records: Configured via `common_folders.calibration`
- Experimental protocols: `docs/methodology.md`

## Data Access

**This repository does NOT contain data files.** 

For data access:
1. Lab members: Configure your `data_config.json` with local data paths
2. External researchers: Contact [your email] for data sharing agreements

## References

1. [Key paper #1 - methodology]
2. [Key paper #2 - instrumentation]
3. [Dataset DOI if published]
```

### Step 9: Create CITATION.md

```markdown
# Citation

If you use this code or data in your research, please cite:

## APA Format
[Your Name]. (2024). WUI Fire Research: Indoor Air Quality Analysis. GitHub. https://github.com/yourusername/project-name

## BibTeX
```bibtex
@software{yourname2024wui,
  author = {Your Name},
  title = {WUI Fire Research: Indoor Air Quality Analysis},
  year = {2024},
  url = {https://github.com/yourusername/project-name},
  version = {1.0.0}
}
```

## Associated Publications

If this code is associated with a publication:

[Author, A., Author, B. (Year). Paper title. Journal Name, Volume(Issue), pages. https://doi.org/...]

## Key References

### Methodology
[Citation for experimental methodology]

### Instrumentation
[Citation for instrument specifications or validation]

### Standards
[Citation for CADR standards, e.g., AHAM AC-1]
```

---

## Usage Examples

### Example 1: Basic Script Structure

```python
"""
Example: AeroTrak Data Analysis
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data_paths import get_instrument_path, get_instrument_files

def main():
    # Get AeroTrak data folder
    aerotrak_path = get_instrument_path('aerotrak')
    
    # Get all CSV files
    all_files = get_instrument_files('aerotrak', '*.csv')
    
    # Process each file
    for data_file in all_files:
        if 'burn8' in data_file.name:
            df = pd.read_csv(data_file)
            print(f"Processing {data_file.name}: {len(df)} records")
            
            # Your analysis here
            
    # Save results
    output_dir = Path(__file__).parent.parent / 'results' / 'tables'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save output
    # results.to_csv(output_dir / 'analysis_results.csv')

if __name__ == '__main__':
    main()
```

### Example 2: Multiple Instruments

```python
"""
Example: Compare AeroTrak and SMPS data
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data_paths import get_instrument_path

def main():
    # Get both instrument paths
    aerotrak_path = get_instrument_path('aerotrak')
    smps_path = get_instrument_path('smps')
    
    # Load specific files
    aerotrak_file = aerotrak_path / 'burn8_aerotrak.csv'
    smps_file = smps_path / 'burn8_smps.txt'
    
    df_aerotrak = pd.read_csv(aerotrak_file)
    df_smps = pd.read_csv(smps_file, sep='\t')
    
    # Compare and analyze
    # ... your comparison code ...

if __name__ == '__main__':
    main()
```

### Example 3: Using Common Folders

```python
"""
Example: Load calibration data
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data_paths import get_instrument_path, get_common_folder

def main():
    # Get calibration folder
    calibration_path = get_common_folder('calibration')
    
    # Load calibration coefficients
    cal_file = calibration_path / 'smps_calibration.csv'
    calibration = pd.read_csv(cal_file)
    
    # Get SMPS data
    smps_path = get_instrument_path('smps')
    data_file = smps_path / 'burn8_smps.txt'
    data = pd.read_csv(data_file, sep='\t')
    
    # Apply calibration
    # ... your calibration code ...

if __name__ == '__main__':
    main()
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: "Configuration file not found"

**Solution:**
```powershell
# Create your local config from template
copy data_config.template.json data_config.json
notepad data_config.json
```

Edit the file with your actual data paths.

---

#### Issue: "Instrument 'xyz' not configured"

**Solution:**
Add the instrument to your `data_config.json`:

```json
{
  "instruments": {
    "xyz": {
      "path": "D:/Your/Path/To/Data",
      "description": "XYZ instrument description",
      "file_pattern": "*.csv"
    }
  }
}
```

---

#### Issue: "Path does not exist" warning

**Checklist:**
1. ✓ Is the drive letter correct? (D: vs E: vs C:)
2. ✓ Does the folder actually exist?
3. ✓ Are you using forward slashes (/) or double backslashes (\\\\)?
4. ✓ Is the path spelled correctly?

**Test path:**
```powershell
# Check if path exists
Test-Path "D:/Research/WUI_Data/AeroTrak"
```

---

#### Issue: Scripts can't find the `data_paths` module

**Solution:**
Make sure you're in the conda environment:
```powershell
conda activate wui
```

Check that `src/__init__.py` exists and contains:
```python
from .data_paths import get_instrument_path, get_instrument_files
```

---

#### Issue: "No files found" in instrument folder

**Checklist:**
1. ✓ Are files actually in that folder?
2. ✓ Does the `file_pattern` match your files? (*.csv vs *.txt)
3. ✓ Check what files exist:

```python
from pathlib import Path
path = Path("D:/Research/WUI_Data/AeroTrak")
print(list(path.glob("*.*")))
```

---

#### Issue: Git is trying to commit `data_config.json`

**Solution:**
```powershell
# Remove from staging
git reset data_config.json

# Verify it's in .gitignore
type .gitignore | findstr data_config.json
```

Should show: `data_config.json`

---

#### Issue: Can't import pandas/bokeh/other packages

**Solution:**
```powershell
# Make sure environment is activated
conda activate wui

# If packages missing, install
conda install pandas bokeh numpy

# Or use pip
pip install pandas bokeh
```

---

## Quick Reference

### Verification Commands

```powershell
# Check configuration
python -c "from src.data_paths import resolver; resolver.list_instruments()"

# Test specific instrument
python -c "from src.data_paths import get_instrument_path; print(get_instrument_path('aerotrak'))"

# List files in instrument folder
python -c "from src.data_paths import get_instrument_files; print(get_instrument_files('aerotrak', '*.csv'))"

# Check machine name
python -c "import socket; print(socket.gethostname())"
```

### Common Path Patterns

```python
# Get instrument folder
aerotrak_path = get_instrument_path('aerotrak')

# Get specific file
data_file = aerotrak_path / 'burn8_data.csv'

# Get all files matching pattern
all_files = get_instrument_files('aerotrak', '*.csv')

# Filter files
burn8_files = [f for f in all_files if 'burn8' in f.name]

# Get most recent file
latest_file = max(all_files, key=lambda p: p.stat().st_mtime)

# Get common folder
calibration_path = get_common_folder('calibration')
```

### Script Template Checklist

When updating a script, ensure:
- [ ] Import statements at top
- [ ] Add src to sys.path
- [ ] Use `get_instrument_path()` instead of hardcoded paths
- [ ] Check if file exists before loading
- [ ] Save outputs to `results/` directory
- [ ] Add helpful print statements
- [ ] Include error handling

---

## Commit and Push

### Step 10: Commit Changes

```powershell
# Stage new files
git add src/ data/ results/ docs/
git add data_config.template.json CITATION.md
git add .gitignore README.md

# Update if exists
git add requirements.txt

# Commit
git commit -m "Add portable data path system for multi-machine Windows setup

- Added src/data_paths.py for instrument-based path resolution
- Created data_config.template.json for machine-specific configuration
- Updated .gitignore to exclude local data and configs
- Added comprehensive README with setup instructions
- Organized results/ directory for outputs
- Added data/ documentation structure"

# Push to GitHub
git push origin main
```

---

## Configuration Examples

### Office PC Example

```json
{
  "machine_name": "OFFICE-PC",
  "data_root": "D:/Research/WUI_FireStudy/Data",
  
  "instruments": {
    "aerotrak": {
      "path": "D:/Research/WUI_FireStudy/Data/AeroTrak",
      "description": "AeroTrak particle counter",
      "file_pattern": "*.csv"
    },
    "smps": {
      "path": "D:/Research/WUI_FireStudy/Data/SMPS",
      "description": "SMPS data",
      "file_pattern": "*.txt"
    },
    "dusttrak": {
      "path": "D:/Research/WUI_FireStudy/Data/DustTrak",
      "description": "DustTrak monitor",
      "file_pattern": "*.csv"
    },
    "purpleair": {
      "path": "D:/Research/WUI_FireStudy/Data/PurpleAir",
      "description": "PurpleAir sensors",
      "file_pattern": "*.csv"
    },
    "quantaq": {
      "path": "D:/Research/WUI_FireStudy/Data/QuantAQ",
      "description": "QuantAQ monitors",
      "file_pattern": "*.csv"
    }
  },
  
  "common_folders": {
    "calibration": "D:/Research/WUI_FireStudy/Calibration",
    "metadata": "D:/Research/WUI_FireStudy/Metadata",
    "raw_backups": "D:/Research/Backups/WUI_Raw"
  }
}
```

### Lab PC Example (Different Paths)

```json
{
  "machine_name": "LAB-WORKSTATION",
  "data_root": "E:/WUI_Data",
  
  "instruments": {
    "aerotrak": {
      "path": "E:/WUI_Data/AeroTrak_Files",
      "description": "AeroTrak particle counter",
      "file_pattern": "*.csv"
    },
    "smps": {
      "path": "E:/WUI_Data/SMPS_Measurements",
      "description": "SMPS data",
      "file_pattern": "*.txt"
    },
    "dusttrak": {
      "path": "E:/WUI_Data/DustTrak_Logs",
      "description": "DustTrak monitor",
      "file_pattern": "*.csv"
    },
    "purpleair": {
      "path": "E:/WUI_Data/PurpleAir_Sensors",
      "description": "PurpleAir sensors",
      "file_pattern": "*.csv"
    },
    "quantaq": {
      "path": "E:/WUI_Data/QuantAQ_Data",
      "description": "QuantAQ monitors",
      "file_pattern": "*.csv"
    }
  },
  
  "common_folders": {
    "calibration": "E:/WUI_Data/Calibration_Files",
    "metadata": "E:/WUI_Data/Metadata",
    "raw_backups": "F:/Backups/WUI_Raw_Data"
  }
}
```

---

## Migration Checklist

### Phase 1: Setup (Day 1)
- [ ] Backup repository (create backup branch)
- [ ] Create new directory structure
- [ ] Create .gitignore file
- [ ] Create data_config.template.json
- [ ] Create src/data_paths.py
- [ ] Create src/__init__.py
- [ ] Test path resolver with verification command

### Phase 2: Documentation (Day 1-2)
- [ ] Update README.md
- [ ] Create data/README.md
- [ ] Create CITATION.md
- [ ] Create docs/methodology.md (if needed)
- [ ] Create requirements.txt (if needed)

### Phase 3: Script Migration (Day 2-5)
- [ ] Create _TEMPLATE_migration.py
- [ ] Update 2-3 frequently-used scripts first
- [ ] Test updated scripts on Office PC
- [ ] Test updated scripts on Lab PC
- [ ] Update remaining scripts gradually

### Phase 4: Configuration (Day 1, repeat per machine)
- [ ] Copy template to data_config.json
- [ ] Edit with machine-specific paths
- [ ] Verify setup with list_instruments()
- [ ] Test running updated scripts
- [ ] Verify outputs are in results/ directory

### Phase 5: Commit and Collaborate (Day 5+)
- [ ] Commit all changes to Git
- [ ] Push to GitHub
- [ ] Share setup instructions with team
- [ ] Help team members configure their machines
- [ ] Update remaining scripts as needed

---

## Best Practices

### For Daily Use

1. **Always activate conda environment first:**
   ```powershell
   conda activate wui
   ```

2. **Verify configuration before running scripts:**
   ```powershell
   python -c "from src.data_paths import resolver; resolver.list_instruments()"
   ```

3. **Check outputs are saved to results/ directory**

4. **Never commit data_config.json**

### For Collaboration

1. **Update template when adding new instruments:**
   - Edit `data_config.template.json`
   - Commit and push
   - Notify team members

2. **Document data sources in data/README.md**

3. **Keep instrument descriptions current**

4. **Share calibration procedures in docs/**

### For Data Management

1. **Organize data by instrument on local machine**

2. **Use consistent file naming:**
   ```
   [instrument]_burn[number]_[location]_[date].csv
   ```

3. **Keep raw data immutable - create copies for processing**

4. **Document quality control procedures**

---

## Additional Resources

### Python Path Management
- [pathlib documentation](https://docs.python.org/3/library/pathlib.html)
- [Working with files in Python](https://realpython.com/working-with-files-in-python/)

### Conda Environments
- [Conda environment management](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
- [Environment YAML specification](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-file-manually)

### Git Best Practices
- [Git ignore patterns](https://git-scm.com/docs/gitignore)
- [Collaborative Git workflows](https://www.atlassian.com/git/tutorials/comparing-workflows)

### Data Management for Research
- [Research data management guide](https://www.dataone.org/best-practices)
- [FAIR data principles](https://www.go-fair.org/fair-principles/)

---

## Support

### Getting Help

1. **Check this guide's troubleshooting section**
2. **Verify your configuration:**
   ```powershell
   python -c "from src.data_paths import resolver; resolver.list_instruments()"
   ```
3. **Check GitHub Issues** (if repository is public)
4. **Contact repository maintainer:** [your email]

### Reporting Issues

When reporting issues, include:
- Machine name and OS (Windows version)
- Python and conda versions
- Error message (full text)
- Steps to reproduce
- Your `data_config.json` structure (remove sensitive paths)

---

## Appendix: Complete File Listing

### Files to Create (Committed to Git)

```
project-name/
├── .gitignore
├── README.md
├── CITATION.md
├── data_config.template.json
├── requirements.txt (optional)
├── wui.yaml (existing)
├── LICENSE (existing or new)
│
├── src/
│   ├── __init__.py
│   └── data_paths.py
│
├── data/
│   ├── README.md
│   ├── aerotrak/
│   ├── smps/
│   ├── dusttrak/
│   ├── purpleair/
│   └── quantaq/
│
├── scripts/
│   ├── _TEMPLATE_migration.py (new)
│   └── [all your existing scripts]
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── output_data/
│
└── docs/
    └── methodology.md
```

### Files to Create (Local Only - Not in Git)

```
project-name/
└── data_config.json  # Machine-specific configuration
```

---

**End of Migration Guide**

---

**Document Version:** 1.0  
**Last Updated:** December 2024  
**Maintained By:** [Your Name]  
**Contact:** [Your Email]
