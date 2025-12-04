"""
Utility functions used across modules.
Contains normalization and helper functions.
"""

import re


def normalize_value(value: str) -> str:
    """
    Normalize value for comparison.
    Converts to lowercase, removes extra whitespace.
    
    Args:
        value: String to normalize
        
    Returns:
        Normalized string
    """
    
    if not value:
        return ""
    
    # Convert to string if needed
    value = str(value)
    
    # Convert to lowercase
    value = value.lower()
    
    # Remove leading/trailing whitespace
    value = value.strip()
    
    # Replace multiple spaces with single space
    value = re.sub(r'\s+', ' ', value)
    
    return value


def cell_to_string(cell) -> str:
    """
    Convert cell value to string safely.
    
    Args:
        cell: Cell value (any type)
        
    Returns:
        String representation
    """
    
    if cell is None:
        return ""
    
    if isinstance(cell, str):
        return cell
    
    return str(cell)