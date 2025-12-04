"""
Configuration settings for PDF comparison tool.
All constants and settings in one place for easy maintenance.
"""

import os
from pathlib import Path

# Directory paths
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Output file names
TABLES_DIFF_FILE = "tables_differences.xlsx"
TEXT_DIFF_FILE = "text_differences.xlsx"
COMPARISON_LOG_FILE = "comparison_log.txt"

# Comparison settings
CASE_SENSITIVE = False  # Ignore case differences
STRIP_WHITESPACE = True  # Remove leading/trailing spaces
NORMALIZE_INTERNAL_SPACES = True  # Convert multiple spaces to single space
MIN_TEXT_LENGTH = 5  # Minimum characters to consider as text (skip very short strings)

# Table settings
TABLE_VALUE_TYPE_TOLERANCE = True  # Convert to string for comparison (handles int vs str)

# Text settings
PARAGRAPH_SEPARATOR = "\n"  # Text line separator