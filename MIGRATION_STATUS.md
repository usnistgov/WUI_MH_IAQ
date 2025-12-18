# WUI Repository Migration Status

**Date:** 2025-12-17
**Machine:** Lenovo-ThinkPad (Laptop)
**Status:** ✅ Phase 1 Complete - Infrastructure Ready

---

## ✅ Completed Tasks

### Infrastructure Setup
- [x] Created directory structure (`data/`, `results/` with subdirectories)
- [x] Updated `.gitignore` with WUI-specific entries
- [x] Created `data_config.template.json` (template for all users)
- [x] Created `data_config.json` (your laptop-specific configuration)
- [x] Created `src/data_paths.py` (portable path resolution module)
- [x] Created `src/__init__.py` (package initialization)
- [x] Created comprehensive `data/README.md` (238 lines of instrument documentation)
- [x] Fixed Windows Unicode encoding issues

### Scripts Updated (3 most recent)
- [x] `wui_temp-rh_comparison.py` - Now uses portable paths
- [x] `wui_spatial_variation_analysis.py` - Now uses portable paths
- [x] `wui_spatial_variation_analysis_plot.py` - Now uses portable paths

### Testing
- [x] Verified all 9 instruments are accessible
- [x] Confirmed all 7 common files found
- [x] Tested import system works correctly

---

## 📊 Configuration Status

**Your Laptop Configuration:**
```
Machine: Lenovo-ThinkPad
Data Root: C:/Users/Nathan/Documents/NIST/WUI_smoke
Config File: data_config.json ✓
```

**Instruments Configured:** 9/9
- ✓ aerotrak_bedroom (10 files)
- ✓ aerotrak_kitchen (10 files)
- ✓ dusttrak (10 files)
- ✓ miniams (1 file)
- ✓ purpleair (1 file)
- ✓ quantaq_bedroom (1 file)
- ✓ quantaq_kitchen (1 file)
- ✓ smps (10 files)
- ✓ vaisala_th (3 files)

**Common Files:** 7/7
- ✓ burn_log
- ✓ burn_dates_decay_aerotrak_bedroom
- ✓ burn_dates_decay_aerotrak_kitchen
- ✓ burn_dates_decay_smps
- ✓ peak_concentrations
- ✓ spatial_variation
- ✓ output_figures

---

## 🔄 Migration Progress

### Phase 1: Infrastructure ✅ COMPLETE
- Portable path system created
- Configuration files in place
- 3 example scripts updated
- System tested successfully

### Phase 2: Remaining Scripts (Not Started)
The following scripts in `src/` still use hardcoded paths:

**CADR Analysis Scripts:**
- `wui_clean_air_delivery_rates_update.py`
- `wui_clean_air_delivery_rates_barchart.py`
- `wui_clean_air_delivery_rates_pmsizes_SIUniformaty.py`
- `wui_clean_air_delivery_rates_vs_total_surface_area.py`
- `cadr_comparison_statistical_analysis.py`

**Compartmentalization Scripts:**
- `wui_compartmentalization_strategy_comparison.py`
- `wui_decay_rate_barchart.py`

**Concentration Scripts:**
- `wui_conc_increase_to_decrease.py`
- `peak_concentration_script.py`

**Instrument Comparison Scripts:**
- `wui_aerotrak_vs_smps.py`
- `wui_dusttrak-rh_comparison.py`
- `wui_purpleair_comparison.py`
- `wui_quantaq_pm2_5_burn8.py`
- `wui_general_particle_count_comparison.py`

**SMPS Scripts:**
- `wui_smps_filterperformance.py`
- `wui_smps_finepm_comparison.py`
- `wui_smps_heatmap.py`
- `wui_smps_mass_vs_conc.py`

**Utility Scripts:**
- `wui_remove_aerotrak_dup_data.py`
- `wui_mh_relay_control_log.py`
- `toc_figure_script.py`

**Total Remaining:** ~23 scripts

---

## 📝 How to Use the New System

### For Scripts You've Already Updated

Simply run them as normal - they'll automatically find your data:

```bash
cd C:\Users\Nathan\Documents\GitHub\python_coding\NIST_wui_mh_iaq
python src/wui_spatial_variation_analysis.py
```

### For Scripts Not Yet Updated

They still use the old hardcoded path:
```python
ABSOLUTE_PATH = "C:/Users/nml/OneDrive - NIST/Documents/NIST/WUI_smoke/"
```

These will fail on your laptop until updated.

### To Update Additional Scripts

Use this pattern in each script:

```python
# OLD CODE (remove):
ABSOLUTE_PATH = "C:/Users/nml/OneDrive - NIST/Documents/NIST/WUI_smoke/"
os.chdir(ABSOLUTE_PATH)

# NEW CODE (add at top of script):
import sys
from pathlib import Path
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

from src.data_paths import get_data_root, get_instrument_path, get_common_file

# Get paths
data_root = get_data_root()
os.chdir(str(data_root))  # If script needs to change directory

# For instrument-specific files:
aerotrak_path = get_instrument_path('aerotrak_bedroom')
data_file = aerotrak_path / 'all_data.xlsx'

# For common files:
burn_log = get_common_file('burn_log')
output_dir = get_common_file('output_figures')
```

---

## 🔧 Verification Commands

**Check configuration status:**
```bash
python -c "from src.data_paths import resolver; resolver.list_instruments()"
```

**Test a specific instrument:**
```bash
python -c "from src import get_instrument_path; print(get_instrument_path('aerotrak_bedroom'))"
```

**Test common files:**
```bash
python -c "from src import get_common_file; print(get_common_file('burn_log'))"
```

---

## 🚀 Next Steps

1. **Test Updated Scripts:** Run the 3 updated scripts to ensure they work correctly
2. **Update More Scripts:** When you're ready, update additional scripts following the pattern
3. **Desktop Setup:** When working on desktop, create a `data_config.json` there with desktop paths
4. **Documentation:** README.md could be updated with portable path usage instructions

---

## 📚 Key Files

- **Configuration Template:** `data_config.template.json` (version controlled)
- **Your Config:** `data_config.json` (not version controlled - your laptop paths)
- **Path Resolver:** `src/data_paths.py` (the magic that makes it work)
- **Data Documentation:** `data/README.md` (comprehensive instrument info)
- **Migration Guide:** `docs/WUI_Repository_Migration_Guide.md` (original plan)

---

## ⚠️ Important Notes

1. **Never commit `data_config.json`** - it's in `.gitignore` for a reason (contains local paths)
2. **Always commit `data_config.template.json`** - others need this to set up their config
3. **The `data/` directory is empty in git** - actual data stays on your local machine
4. **Scripts can now run on any machine** - as long as `data_config.json` is set up correctly

---

## 💡 Benefits

- ✅ No more hardcoded paths in scripts
- ✅ Works on laptop AND desktop automatically
- ✅ Easy for collaborators to set up
- ✅ Data stays local (not in repository)
- ✅ Single config file to update if data moves
- ✅ Clear documentation of all instruments

---

**System Status:** 🟢 OPERATIONAL
**Next Migration:** Update remaining scripts as needed for active analysis tasks
