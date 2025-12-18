"""
Script to migrate WUI analysis scripts to use portable data paths.

This script automatically updates Python files in src/ to use the portable
data path system instead of hardcoded absolute paths.

Author: Nathan Lima
Date: 2025-12-17
"""
import re
from pathlib import Path
import shutil
from datetime import datetime

# Scripts already migrated (skip these)
MIGRATED_SCRIPTS = {
    '__init__.py',
    'data_paths.py',
    'wui_temp-rh_comparison.py',
    'wui_spatial_variation_analysis.py',
    'wui_spatial_variation_analysis_plot.py',
}

# Portable imports to add
PORTABLE_IMPORTS = '''import sys
from pathlib import Path

# Add repository root to path for portable data access
script_dir = Path(__file__).parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

from src.data_paths import get_data_root, get_instrument_path, get_common_file'''


def backup_file(file_path):
    """Create a backup of the file before modifying"""
    backup_dir = Path('migration_backups')
    backup_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = backup_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"
    shutil.copy2(file_path, backup_path)
    return backup_path


def update_script(file_path):
    """Update a single script to use portable paths"""
    print(f"\nProcessing: {file_path.name}")

    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    modified = False

    # Pattern 1: Replace hardcoded absolute_path definitions
    patterns_to_replace = [
        (
            r'(?:absolute_path|directory_path)\s*=\s*["\']C:/Users/(?:nml/OneDrive - NIST/Documents/NIST/WUI_smoke|Nathan/Documents/NIST/WUI_smoke)[/\'"]*',
            'data_root = get_data_root()  # Portable path - auto-configured'
        ),
        (
            r'ABSOLUTE_PATH\s*=\s*["\']C:/Users/(?:nml/OneDrive - NIST/Documents/NIST/WUI_smoke|Nathan/Documents/NIST/WUI_smoke)[/\'"]*',
            'data_root = get_data_root()  # Portable path - auto-configured'
        ),
    ]

    for pattern, replacement in patterns_to_replace:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            modified = True
            print(f"  [OK] Replaced hardcoded path")

    # Pattern 2: Replace os.chdir(absolute_path) or os.chdir(ABSOLUTE_PATH)
    chdir_pattern = r'os\.chdir\((?:absolute_path|ABSOLUTE_PATH|directory_path|data_root)\)'
    if re.search(chdir_pattern, content):
        content = re.sub(chdir_pattern, 'os.chdir(str(data_root))', content)
        modified = True
        print(f"  [OK] Updated os.chdir() call")

    # Pattern 3: Check if imports are needed
    if modified and 'from src.data_paths import' not in content:
        # Find where to insert imports (after existing imports)
        import_section_end = 0
        lines = content.split('\n')

        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_section_end = i + 1
            elif import_section_end > 0 and line.strip() and not line.strip().startswith('#'):
                # Found first non-import, non-comment line
                break

        if import_section_end > 0:
            lines.insert(import_section_end, '\n' + PORTABLE_IMPORTS + '\n')
            content = '\n'.join(lines)
            print(f"  [OK] Added portable path imports")

    # Pattern 4: Update specific file path references
    # AeroTrak bedroom
    content = re.sub(
        r'["\']\.?/?burn_data/aerotraks/bedroom2/all_data\.xlsx["\']',
        'str(get_instrument_path(\'aerotrak_bedroom\') / \'all_data.xlsx\')',
        content
    )

    # AeroTrak kitchen
    content = re.sub(
        r'["\']\.?/?burn_data/aerotraks/kitchen/all_data\.xlsx["\']',
        'str(get_instrument_path(\'aerotrak_kitchen\') / \'all_data.xlsx\')',
        content
    )

    # QuantAQ paths
    content = re.sub(
        r'["\']\.?/?burn_data/quantaq/MOD-PM-00194[^"\']*\.csv["\']',
        'str(get_instrument_path(\'quantaq_bedroom\') / \'MOD-PM-00194-b0fc215029fa4852b926bc50b28fda5a.csv\')',
        content
    )

    content = re.sub(
        r'["\']\.?/?burn_data/quantaq/MOD-PM-00197[^"\']*\.csv["\']',
        'str(get_instrument_path(\'quantaq_kitchen\') / \'MOD-PM-00197-a6dd467a147a4d95a7b98a8a10ab4ea3.csv\')',
        content
    )

    # Burn log
    content = re.sub(
        r'["\']\.?/?burn_log\.xlsx["\']',
        'str(get_common_file(\'burn_log\'))',
        content
    )

    # Output figures directory
    content = re.sub(
        r'["\']\.?/?Paper_figures/?["\']',
        'str(get_common_file(\'output_figures\'))',
        content
    )

    # Check if anything changed
    if content != original_content:
        # Create backup
        backup_path = backup_file(file_path)
        print(f"  [OK] Backup created: {backup_path.name}")

        # Write updated content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  [OK] File updated successfully")
        return True
    else:
        print(f"  - No changes needed")
        return False


def main():
    """Main migration function"""
    print("="*80)
    print("WUI Scripts Migration to Portable Paths")
    print("="*80)

    src_dir = Path('src')
    if not src_dir.exists():
        print(f"ERROR: src/ directory not found")
        return

    # Get all Python files
    python_files = list(src_dir.glob('*.py'))
    print(f"\nFound {len(python_files)} Python files in src/")

    # Filter out already migrated scripts
    files_to_migrate = [
        f for f in python_files
        if f.name not in MIGRATED_SCRIPTS
    ]

    print(f"Files to migrate: {len(files_to_migrate)}")
    print(f"Already migrated: {len(MIGRATED_SCRIPTS)}")

    if not files_to_migrate:
        print("\n[OK] All scripts already migrated!")
        return

    # Confirm before proceeding
    print(f"\nFiles that will be updated:")
    for f in files_to_migrate:
        print(f"  - {f.name}")

    response = input("\nProceed with migration? (y/n): ")
    if response.lower() != 'y':
        print("Migration cancelled")
        return

    # Migrate each file
    updated_count = 0
    for file_path in files_to_migrate:
        if update_script(file_path):
            updated_count += 1

    print("\n" + "="*80)
    print(f"Migration Complete!")
    print(f"  Updated: {updated_count} files")
    print(f"  Skipped: {len(files_to_migrate) - updated_count} files")
    print(f"  Backups: migration_backups/")
    print("="*80)

    print("\nNext steps:")
    print("1. Review the changes in your version control system")
    print("2. Test a few scripts to ensure they work correctly")
    print("3. If everything works, you can delete migration_backups/")
    print("4. Commit the changes to git")


if __name__ == '__main__':
    main()
