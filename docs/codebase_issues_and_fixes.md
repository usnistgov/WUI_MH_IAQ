# Codebase Issues and Required Fixes

**Date reviewed:** 2026-02-26
**Reviewer:** Claude (AI assistant)
**Context:** Review performed during development of the SMPS size-bin barchart script.

---

## CRITICAL Issues

### 1. `src/clean_air_delivery_rates_update.py` — Broken Python Syntax (SyntaxError)

**Severity:** Critical — script cannot be imported or run at all.

**Problem:** The `from bokeh.models import (` statement on line 83 is broken. Path-setup code and additional imports were inserted in the middle of the multi-line import block (lines 85–93), and the continuation of the bokeh import arguments (`ColumnDataSource, Band, Label, ...`) appear after these insertions. Python cannot parse this.

**Broken code (lines 82–105):**
```python
from bokeh.plotting import figure, show, output_file, output_notebook
from bokeh.models import (     # <-- import opened here

import sys                     # <-- WRONG: these lines are INSIDE the open paren
from pathlib import Path
...
from src.data_paths import get_data_root, get_instrument_path, get_common_file

    ColumnDataSource,          # <-- continuation of the broken import
    Band,
    Label,
    ...
)
```

**Fix:** Reorganize imports so all path-setup code and `sys.path` manipulation come before the bokeh imports. Move the `import sys`, `from pathlib import Path`, path manipulation, and `from src.data_paths import ...` lines above the `from bokeh.models import (` block.

**Correct structure:**
```python
import os
import sys
from pathlib import Path
from datetime import datetime as dt

# Path setup
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

from src.data_paths import get_data_root, get_instrument_path, get_common_file
import numpy as np
import pandas as pd
from scipy import optimize
from bokeh.plotting import figure, show, output_file, output_notebook
from bokeh.models import (
    ColumnDataSource,
    Band,
    Label,
    Arrow,
    OpenHead,
    CrosshairTool,
    Span,
    Legend,
    LegendItem,
    Div,
)
from bokeh.layouts import gridplot, row, column
from functools import reduce
```

**Additional note:** This script also uses old-style file paths (e.g., `"./burn_dates_decay_aerotraks_bedroom.xlsx"`) and old merge logic with `min_since_peak`. It has NOT been fully refactored to use the new `data_paths` module or `INSTRUMENT_CONFIG` pattern — it is effectively an old-style script with new import code grafted onto it incorrectly. A fuller refactor to match `clean_air_delivery_rates_pmsizes.py` style is needed.

---

## HIGH Priority Issues

### 2. `data_config.template.json` — Missing `"burn_calcs"` Key

**Severity:** High — causes a `KeyError` at runtime in `clean_air_delivery_rates_pmsizes.py`.

**Problem:** `src/clean_air_delivery_rates_pmsizes.py` calls `get_common_file("burn_calcs")` (lines 1349 and 2644), but `"burn_calcs"` does not exist as a key in `common_folders` of `data_config.template.json`. Any user running the script will get:
```
KeyError: "Common file 'burn_calcs' not found. Available: [...]"
```

**Fix (Option A — preferred for consistency):** Add `"burn_calcs"` to `common_folders` in `data_config.template.json`:
```json
"burn_calcs": "PATH_TO_DATA_ROOT/burn_data/burn_calcs"
```
And update each machine's `data_config.json` accordingly.

**Fix (Option B — no config change needed):** Change `clean_air_delivery_rates_pmsizes.py` to construct the path from `data_root` like `decay_rate_barchart.py` does:
```python
# Replace:
burn_calcs_path = get_common_file("burn_calcs")
# With:
burn_calcs_path = get_data_root() / "burn_data" / "burn_calcs"
```
This requires updating lines 1349 and 2644 in `clean_air_delivery_rates_pmsizes.py`.

### 3. Path Inconsistency Between `clean_air_delivery_rates_pmsizes.py` and `decay_rate_barchart.py`

**Severity:** High — the two scripts may read/write to different directories.

**Problem:** The two scripts construct the `burn_calcs` path differently:
- `decay_rate_barchart.py` line 65: `str(data_root / "burn_data" / "burn_calcs")`
- `clean_air_delivery_rates_pmsizes.py` lines 1349, 2644: `get_common_file("burn_calcs")`

If `data_config.json` has a `"burn_calcs"` key pointing to a different location than `data_root/burn_data/burn_calcs`, the two scripts will read/write to different directories and the barchart script won't find the Excel files produced by the pmsizes script.

**Fix:** Standardize both scripts to use the same path construction method. Recommend using `get_common_file("burn_calcs")` everywhere (after adding the key to `data_config.template.json`), or using `get_data_root() / "burn_data" / "burn_calcs"` everywhere.

Scripts affected:
- `src/clean_air_delivery_rates_pmsizes.py`
- `src/decay_rate_barchart.py`
- `src/clean_air_delivery_rates_barchart.py` (also uses `data_root / "burn_data" / "burn_calcs"`)
- Any new scripts that read from `burn_calcs`

---

## MEDIUM Priority Issues

### 4. `src/clean_air_delivery_rates_pmsizes.py` — `output_notebook()` Called at Module Level

**Severity:** Medium — causes unexpected behavior when running as a script outside Jupyter.

**Problem:** Line ~90 calls `output_notebook()` at module level. When this script is imported or run outside a Jupyter/IPython environment, Bokeh plots will not display and a `BokehUserWarning` or silent failure may occur.

**Fix:** Replace `output_notebook()` at the top of the file with `output_notebook()` guarded for notebook context, or remove it and let the `plot_pm_data` function handle output configuration. The `main()` function could be modified to accept an `output_mode` argument ("notebook" vs "file").

### 5. `src/clean_air_delivery_rates_pmsizes.py` — Default `dataset = "MiniAMS"` May Be Confusing

**Severity:** Low/Medium — not a bug per se, but the default to MiniAMS may mislead users who want to process SMPS data.

**Problem:** The `dataset` variable on line 93 defaults to `"MiniAMS"`. For SMPS size-bin barchart work, the user must remember to change this to `"SMPS"` before running.

**Fix:** Consider adding a CLI argument parser (`argparse`) or a config dict that makes the selected dataset more explicit. Or document clearly that the `dataset` variable must be changed manually.

---

## LOW Priority Issues / Notes

### 6. `src/clean_air_delivery_rates_update.py` — Old-Style Data Access and Logic

After fixing the syntax error (Issue #1), the script still uses:
- Old file paths: `"./burn_dates_decay_aerotraks_bedroom.xlsx"` (should use `get_common_file(...)`)
- Old merge logic using `min_since_peak` as join key (doesn't match the refactored approach in `clean_air_delivery_rates_pmsizes.py`)
- `os.chdir(str(data_root))` (fragile — changes process working directory)
- `dataset = "QuantAQb"` (note lowercase 'b' — differs from "QuantAQB" used elsewhere)

This script needs a comprehensive refactor to match the patterns in `clean_air_delivery_rates_pmsizes.py`.

### 7. Prerequisite Data Generation Order

The `decay_rate_barchart.py` and any new SMPS size-bin barchart script depend on Excel files produced by `clean_air_delivery_rates_pmsizes.py`. Users must run `clean_air_delivery_rates_pmsizes.py` with `dataset = "SMPS"` (and each other instrument in turn) before running the barchart scripts. This dependency is not documented anywhere in the codebase.

**Suggested fix:** Add a `README` or docstring note documenting the execution order:
1. Run `clean_air_delivery_rates_pmsizes.py` with each `dataset` value to generate `burn_calcs/*.xlsx` files
2. Run `decay_rate_barchart.py` to generate summary barcharts from those files

---

## Summary Table

| # | File | Severity | Issue | Fix Required |
|---|------|----------|-------|--------------|
| 1 | `clean_air_delivery_rates_update.py` | **Critical** | Broken import syntax (SyntaxError) | Restructure imports |
| 2 | `data_config.template.json` | **High** | Missing `"burn_calcs"` key → KeyError | Add key to template + user configs |
| 3 | Multiple files | **High** | Path inconsistency for `burn_calcs` dir | Standardize path construction |
| 4 | `clean_air_delivery_rates_pmsizes.py` | Medium | `output_notebook()` at module level | Guard for notebook context |
| 5 | `clean_air_delivery_rates_pmsizes.py` | Low/Med | Default `dataset = "MiniAMS"` | Document or add CLI arg |
| 6 | `clean_air_delivery_rates_update.py` | Low | Old-style data access throughout | Full refactor needed |
| 7 | All scripts | Low | No documented execution order | Add README/docstring |
