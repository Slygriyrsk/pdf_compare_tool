# """
# Text normalization utilities for consistent comparison.
# Handles case, whitespace, and encoding issues.
# """

# import re


# class TextNormalizer:
#     """Normalize text for consistent comparison."""
    
#     def __init__(self, case_sensitive=False, strip_whitespace=True, normalize_spaces=True):
#         self.case_sensitive = case_sensitive
#         self.strip_whitespace = strip_whitespace
#         self.normalize_spaces = normalize_spaces
    
#     def normalize(self, text):
#         """Apply all normalization rules to text."""
#         if text is None:
#             return ""
        
#         # Convert to string if needed
#         text = str(text).strip()
        
#         # Remove non-breaking spaces and other unicode whitespace
#         text = re.sub(r'\s+', ' ', text)
        
#         # Apply case normalization
#         if not self.case_sensitive:
#             text = text.lower()
        
#         # Strip leading/trailing spaces
#         if self.strip_whitespace:
#             text = text.strip()
        
#         # Normalize internal spaces
#         if self.normalize_spaces:
#             text = re.sub(r'\s+', ' ', text)
        
#         return text
    
#     def normalize_row(self, row):
#         """Normalize each cell in a row."""
#         if isinstance(row, dict):
#             return {key: self.normalize(value) for key, value in row.items()}
#         elif isinstance(row, (list, tuple)):
#             return [self.normalize(cell) for cell in row]
#         return row

import re

class TextNormalizer:
    """
    Handles text normalization for robust comparison.
    Removes extra spaces, converts to lowercase, handles special characters.
    """
    
    @staticmethod
    def normalize_text(text):
        """
        Normalize text for comparison:
        - Convert to lowercase
        - Remove extra whitespace
        - Remove special characters but keep important punctuation
        - Trim leading/trailing spaces
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove extra whitespace (multiple spaces, tabs, newlines)
        text = re.sub(r'\s+', ' ', text)
        
        # Trim leading and trailing spaces
        text = text.strip()
        
        return text

def normalize_text(text):
    """Convenience function for normalization"""
    return TextNormalizer.normalize_text(text)