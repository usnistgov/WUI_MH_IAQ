"""
WUI Fire Research Code Package

This package provides utilities for analyzing wildland-urban interface (WUI)
fire smoke infiltration data from multi-instrument measurements.

Main Components:
    - data_paths: Portable data path resolution for cross-machine compatibility
    - Analysis scripts for CADR, spatial variation, and instrument comparisons

Author: Nathan Lima
Institution: National Institute of Standards and Technology (NIST)
Date: 2025
"""
from .data_paths import (
    get_instrument_path,
    get_instrument_files,
    get_common_file,
    get_data_root,
    WUIDataPathResolver
)

__all__ = [
    'get_instrument_path',
    'get_instrument_files',
    'get_common_file',
    'get_data_root',
    'WUIDataPathResolver'
]

__version__ = '1.0.0'
__author__ = 'Nathan Lima'
__institution__ = 'National Institute of Standards and Technology (NIST)'
