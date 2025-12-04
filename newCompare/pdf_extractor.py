# # """
# # Handles PDF text and table extraction.
# # Uses pdfplumber for reliable table detection and text extraction.
# # """

# # import pdfplumber
# # from typing import List, Dict, Tuple


# # class PDFExtractor:
# #     """Extracts tables and text from PDF files."""
    
# #     def extract_all(self, pdf_path: str) -> Tuple[List[List[Dict]], str]:
# #         """
# #         Extract both tables and text from PDF.
        
# #         Args:
# #             pdf_path: Path to PDF file
            
# #         Returns:
# #             Tuple of (tables_list, combined_text)
# #         """
# #         tables = []
# #         text_data = []
        
# #         try:
# #             with pdfplumber.open(pdf_path) as pdf:
# #                 for page_num, page in enumerate(pdf.pages, 1):
# #                     # Extract tables from page
# #                     page_tables = page.extract_tables()
# #                     if page_tables:
# #                         for table in page_tables:
# #                             # Convert table to list of dicts with headers
# #                             if table:
# #                                 tables.append(table)
                    
# #                     # Extract text from page
# #                     page_text = page.extract_text()
# #                     if page_text:
# #                         text_data.append(page_text)
        
# #         except Exception as e:
# #             print(f"Error extracting from PDF {pdf_path}: {str(e)}")
# #             return [], ""
        
# #         # Join all text with page separators
# #         combined_text = "\n---PAGE_BREAK---\n".join(text_data)
        
# #         return tables, combined_text
    
# #     def table_to_records(self, table: List[List[str]]) -> List[Dict[str, str]]:
# #         """
# #         Convert table (list of lists) to list of dictionaries.
# #         Uses first row as headers.
        
# #         Args:
# #             table: 2D list representing table
            
# #         Returns:
# #             List of dictionaries representing rows
# #         """
# #         if not table or len(table) < 1:
# #             return []
        
# #         headers = table[0]
# #         records = []
        
# #         for row in table[1:]:
# #             # Pad row with empty strings if needed
# #             while len(row) < len(headers):
# #                 row.append("")
            
# #             record = {}
# #             for header, value in zip(headers, row):
# #                 record[header] = str(value) if value else ""
            
# #             records.append(record)
        
# #         return records

# """
# Extract text and tables from PDF files.
# Uses pdfplumber for robust extraction.
# """

# import pdfplumber
# from pathlib import Path


# class PDFExtractor:
#     """Extract tables and text from PDF documents."""
    
#     def __init__(self):
#         self.tables = []
#         self.text_paragraphs = []
#         self.file_name = None
    
#     def extract(self, pdf_path):
#         """
#         Extract all tables and text from PDF.
        
#         Args:
#             pdf_path: Path to PDF file
        
#         Returns:
#             dict with 'tables' and 'text' keys
#         """
#         self.file_name = Path(pdf_path).name
#         self.tables = []
#         self.text_paragraphs = []
        
#         with pdfplumber.open(pdf_path) as pdf:
#             for page_num, page in enumerate(pdf.pages, 1):
#                 self._extract_tables_from_page(page, page_num)
#                 self._extract_text_from_page(page, page_num)
        
#         return {
#             "tables": self.tables,
#             "text": self.text_paragraphs,
#             "file_name": self.file_name
#         }
    
#     def _extract_tables_from_page(self, page, page_num):
#         """Extract all tables from a single page."""
#         try:
#             tables = page.extract_tables()
#             if tables:
#                 for table_idx, table in enumerate(tables):
#                     # Convert table to list of dicts (header + rows)
#                     if len(table) > 0:
#                         formatted_table = self._format_table(table)
#                         self.tables.append({
#                             "page": page_num,
#                             "table_index": table_idx,
#                             "data": formatted_table
#                         })
#         except Exception as e:
#             print(f"Warning: Could not extract tables from page {page_num}: {e}")
    
#     def _extract_text_from_page(self, page, page_num):
#         """Extract text paragraphs from a single page."""
#         try:
#             text = page.extract_text()
#             if text:
#                 # Split by multiple line breaks to get paragraphs
#                 paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
#                 for para_idx, paragraph in enumerate(paragraphs):
#                     if len(paragraph) > 5:  # Skip very short strings
#                         self.text_paragraphs.append({
#                             "page": page_num,
#                             "paragraph_index": para_idx,
#                             "text": paragraph
#                         })
#         except Exception as e:
#             print(f"Warning: Could not extract text from page {page_num}: {e}")
    
#     def _format_table(self, table):
#         """
#         Convert raw table to clean format.
#         Returns list of dicts where each row is a dictionary.
#         """
#         if not table:
#             return []
        
#         # Assume first row is header
#         headers = [str(h).strip() if h else "Column" for h in table[0]]
        
#         formatted = []
#         for row in table[1:]:
#             row_dict = {}
#             for idx, cell in enumerate(row):
#                 header = headers[idx] if idx < len(headers) else f"Column_{idx}"
#                 row_dict[header] = str(cell).strip() if cell else ""
#             formatted.append(row_dict)
        
#         return formatted

import PyPDF2
import pandas as pd
from io import StringIO

class PDFExtractor:
    """
    Extracts tables and text from PDF files.
    Uses PyPDF2 for basic extraction and pdfplumber for table detection.
    """
    
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.tables = []
        self.text = ""
    
    def extract_tables(self):
        """
        Extract tables from PDF.
        Returns list of tables (each table is list of rows).
        """
        try:
            import pdfplumber
            
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    
                    if tables:
                        for table in tables:
                            self.tables.append(table)
                            print(f"  - Table found on page {page_num + 1}")
            
            print(f"Total tables extracted: {len(self.tables)}")
        except Exception as e:
            print(f"Error extracting tables: {e}")
        
        return self.tables
    
    def extract_text(self):
        """
        Extract text from PDF.
        Returns full text content.
        """
        try:
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text:
                        self.text += text + " "
            
            print(f"Text extracted: {len(self.text)} characters")
        except Exception as e:
            print(f"Error extracting text: {e}")
        
        return self.text
    
    def extract_all(self):
        """Extract both tables and text"""
        print(f"Extracting from: {self.pdf_path}")
        self.extract_tables()
        self.extract_text()
        return self.tables, self.text