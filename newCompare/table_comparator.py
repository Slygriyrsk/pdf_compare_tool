# # """
# # Compares table data from two PDFs.
# # Uses row-level comparison with normalization for matching.
# # """

# # from typing import List, Dict
# # from utils import normalize_value


# # class TableComparator:
# #     """Compares tables from two PDFs."""
    
# #     def compare(self, table1: List[List[str]], table2: List[List[str]], 
# #                 file1_name: str, file2_name: str) -> List[Dict]:
# #         """
# #         Compare two tables and find differences.
# #         Each row from table1 is checked against ALL rows in table2.
        
# #         Args:
# #             table1: Table from first PDF
# #             table2: Table from second PDF
# #             file1_name: Name of first PDF file
# #             file2_name: Name of second PDF file
            
# #         Returns:
# #             List of rows from table1 that don't match any row in table2
# #         """
        
# #         differences = []
        
# #         # Skip header row, start from row 1
# #         for row_idx in range(1, len(table1)):
# #             row1 = table1[row_idx]
            
# #             # Check if this row exists in any row of table2
# #             if not self._row_exists_in_table(row1, table2):
# #                 differences.append({
# #                     'file_from': file1_name,
# #                     'row_index': row_idx,
# #                     'values': row1,
# #                     'status': 'not_found_in_other_pdf'
# #                 })
        
# #         return differences
    
# #     def _row_exists_in_table(self, row: List[str], table: List[List[str]]) -> bool:
# #         """
# #         Check if row values exist in any row of table.
# #         Uses normalized comparison (case-insensitive, trimmed whitespace).
        
# #         Args:
# #             row: Row to search for
# #             table: Table to search in
            
# #         Returns:
# #             True if row found in table, False otherwise
# #         """
        
# #         # Normalize the search row
# #         normalized_search_row = [normalize_value(cell) for cell in row]
        
# #         # Check against all rows in table
# #         for table_row in table[1:]:  # Skip header
# #             normalized_table_row = [normalize_value(cell) for cell in table_row]
            
# #             # Compare cell by cell
# #             if self._rows_match(normalized_search_row, normalized_table_row):
# #                 return True
        
# #         return False
    
# #     def _rows_match(self, row1: List[str], row2: List[str]) -> bool:
# #         """
# #         Check if two rows match exactly (all cells same).
# #         Handles different row lengths gracefully.
        
# #         Args:
# #             row1: First row (normalized)
# #             row2: Second row (normalized)
            
# #         Returns:
# #             True if rows match, False otherwise
# #         """
        
# #         # Get max length to compare all cells
# #         max_len = max(len(row1), len(row2))
        
# #         # Pad shorter row with empty strings
# #         row1_padded = row1 + [""] * (max_len - len(row1))
# #         row2_padded = row2 + [""] * (max_len - len(row2))
        
# #         # Compare all cells
# #         return all(cell1 == cell2 for cell1, cell2 in zip(row1_padded, row2_padded))

# """
# Compare tables from two PDFs.
# Matches rows across all rows and identifies differences.
# """

# from normalizer import TextNormalizer


# class TableComparator:
#     """Compare tables with row-level matching across all rows."""
    
#     def __init__(self):
#         self.normalizer = TextNormalizer()
#         self.differences = []
    
#     def compare(self, tables_pdf1, tables_pdf2, file_name_1, file_name_2):
#         """
#         Compare tables from two PDFs.
#         Each row from PDF1 is checked against ALL rows in PDF2.
        
#         Args:
#             tables_pdf1: List of table dicts from PDF1
#             tables_pdf2: List of table dicts from PDF2
#             file_name_1: Name of first PDF
#             file_name_2: Name of second PDF
        
#         Returns:
#             List of differences found
#         """
#         self.differences = []
        
#         if not tables_pdf1 and not tables_pdf2:
#             return []
        
#         for table1 in tables_pdf1:
#             table_data_1 = table1.get("data", [])
            
#             for row_idx, row_1 in enumerate(table_data_1):
#                 # Check if this row exists in ANY row of PDF2 tables
#                 row_found = False
                
#                 for table2 in tables_pdf2:
#                     table_data_2 = table2.get("data", [])
                    
#                     for row_2 in table_data_2:
#                         if self._rows_match(row_1, row_2):
#                             row_found = True
#                             break
                    
#                     if row_found:
#                         break
                
#                 # If row not found in any table of PDF2, add to differences
#                 if not row_found:
#                     self.differences.append({
#                         "source": f"Table (Page {table1['page']}, Table {table1['table_index']}) - Row {row_idx + 1}",
#                         "from_file": file_name_1,
#                         "comparison_file": file_name_2,
#                         "row_data": row_1,
#                         "reason": "Row not found in second PDF"
#                     })
        
#         return self.differences
    
#     def _rows_match(self, row_1, row_2):
#         """
#         Check if two rows match (all cell values are equal when normalized).
        
#         Args:
#             row_1: First row (dict)
#             row_2: Second row (dict)
        
#         Returns:
#             True if all values match
#         """
#         # Normalize both rows
#         norm_row_1 = self.normalizer.normalize_row(row_1)
#         norm_row_2 = self.normalizer.normalize_row(row_2)
        
#         # Check if all values in row_1 exist in row_2
#         for key, value in norm_row_1.items():
#             if value and value not in norm_row_2.values():
#                 return False
        
#         return True
import pandas as pd
from normalizer import normalize_text

class TableComparator:
    """
    Compares table data from two PDFs.
    Logic: If ANY cell in a row from PDF1 is NOT found anywhere in PDF2,
    that entire row is marked as different/new.
    """
    
    def __init__(self):
        self.pdf1_tables = []
        self.pdf2_tables = []
    
    def load_tables(self, pdf1_tables, pdf2_tables):
        """Store extracted tables from both PDFs"""
        self.pdf1_tables = pdf1_tables
        self.pdf2_tables = pdf2_tables
    
    def cell_exists_in_pdf2(self, cell_value):
        """
        Check if a cell value exists ANYWHERE in any table in PDF2.
        Normalized comparison (case-insensitive, whitespace-trimmed)
        """
        normalized_search = normalize_text(str(cell_value))
        
        for table in self.pdf2_tables:
            for row in table:
                for cell in row:
                    normalized_cell = normalize_text(str(cell))
                    if normalized_search == normalized_cell:
                        return True
        return False
    
    def compare(self):
        """
        Compare all tables from PDF1 against PDF2.
        Returns: List of rows with NEW/DIFFERENT data
        """
        different_rows = []
        
        for table_idx, table in enumerate(self.pdf1_tables):
            for row_idx, row in enumerate(table):
                row_is_different = False
                
                # Check if ANY cell in this row is NOT found in PDF2
                for cell in row:
                    if not self.cell_exists_in_pdf2(cell):
                        row_is_different = True
                        break
                
                # If row contains at least one new/different cell, add it
                if row_is_different:
                    different_rows.append({
                        'table_number': table_idx + 1,
                        'row_number': row_idx + 1,
                        'row_data': row,
                        'status': 'New/Different'
                    })
        
        return different_rows