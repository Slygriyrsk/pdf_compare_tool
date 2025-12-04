# # """
# # Main entry point for PDF comparison tool.
# # Orchestrates the extraction, comparison, and result generation workflow.
# # """

# # import sys
# # from pathlib import Path
# # from pdf_extractor import PDFExtractor
# # from table_comparator import TableComparator
# # from text_comparator import TextComparator
# # from output_generator import OutputGenerator


# # def run_comparison(pdf1_path, pdf2_path, output_dir="results"):
# #     """
# #     Main function to execute PDF comparison workflow.
    
# #     Args:
# #         pdf1_path: Path to first PDF file
# #         pdf2_path: Path to second PDF file
# #         output_dir: Directory to save comparison results
    
# #     Returns:
# #         Boolean indicating success or failure
# #     """
    
# #     try:
# #         # Validate input files exist
# #         pdf1_path = Path(pdf1_path)
# #         pdf2_path = Path(pdf2_path)
        
# #         if not pdf1_path.exists() or not pdf2_path.exists():
# #             print(f"Error: One or both PDF files not found")
# #             return False
        
# #         print(f"Starting comparison of:")
# #         print(f"  PDF 1: {pdf1_path.name}")
# #         print(f"  PDF 2: {pdf2_path.name}")
# #         print()
        
# #         # Extract data from both PDFs
# #         print("Extracting data from PDFs...")
# #         extractor = PDFExtractor()
        
# #         pdf1_tables, pdf1_text = extractor.extract_all(str(pdf1_path))
# #         pdf2_tables, pdf2_text = extractor.extract_all(str(pdf2_path))
        
# #         print(f"  Found {len(pdf1_tables)} tables in PDF 1")
# #         print(f"  Found {len(pdf2_tables)} tables in PDF 2")
# #         print()
        
# #         # Create output directory
# #         output_path = Path(output_dir)
# #         output_path.mkdir(exist_ok=True)
        
# #         # Compare tables
# #         print("Comparing table data...")
# #         table_comparator = TableComparator()
# #         table_differences = []
        
# #         if pdf1_tables and pdf2_tables:
# #             for idx, (table1, table2) in enumerate(zip(pdf1_tables, pdf2_tables)):
# #                 differences = table_comparator.compare(
# #                     table1, table2,
# #                     pdf1_path.name, pdf2_path.name
# #                 )
# #                 table_differences.append({
# #                     'table_index': idx,
# #                     'differences': differences
# #                 })
        
# #         # Compare text
# #         print("Comparing text data...")
# #         text_comparator = TextComparator()
# #         text_differences = text_comparator.compare(
# #             pdf1_text, pdf2_text,
# #             pdf1_path.name, pdf2_path.name
# #         )
        
# #         # Generate output files
# #         print("Generating output files...")
# #         generator = OutputGenerator()
        
# #         result_files = []
        
# #         if any(t['differences'] for t in table_differences):
# #             table_output_file = output_path / f"table_differences.xlsx"
# #             generator.generate_table_output(table_differences, table_output_file)
# #             result_files.append(str(table_output_file))
# #             print(f"  Table differences saved: {table_output_file}")
        
# #         if text_differences:
# #             text_output_file = output_path / f"text_differences.xlsx"
# #             generator.generate_text_output(text_differences, text_output_file)
# #             result_files.append(str(text_output_file))
# #             print(f"  Text differences saved: {text_output_file}")
        
# #         if not result_files:
# #             print("  No differences found between PDFs!")
        
# #         print()
# #         print("Comparison completed successfully!")
# #         return True
        
# #     except Exception as e:
# #         print(f"Error during comparison: {str(e)}")
# #         import traceback
# #         traceback.print_exc()
# #         return False


# # if __name__ == "__main__":
# #     if len(sys.argv) < 3:
# #         print("Run: python main.py <pdf1_path> <pdf2_path> [output_directory]")
# #         print("\nExample:")
# #         print("  python main.py document1.pdf document2.pdf results/")
# #         sys.exit(1)
    
# #     pdf1 = sys.argv[1]
# #     pdf2 = sys.argv[2]
# #     output_dir = sys.argv[3] if len(sys.argv) > 3 else "results"
    
# #     success = run_comparison(pdf1, pdf2, output_dir)
# #     sys.exit(0 if success else 1)

# """
# Main orchestrator for PDF comparison tool.
# Entry point that coordinates extraction, comparison, and output.
# """

# import sys
# from pathlib import Path
# from pdf_extractor import PDFExtractor
# from table_comparator import TableComparator
# from text_comparator import TextComparator
# from output_generator import ResultsGenerator


# class PDFComparisonTool:
#     """Orchestrate PDF comparison workflow."""
    
#     def __init__(self, output_dir="results"):
#         self.extractor = PDFExtractor()
#         self.table_comparator = TableComparator()
#         self.text_comparator = TextComparator()
#         self.output_generator = ResultsGenerator(output_dir)
    
#     def compare_pdfs(self, pdf_path_1, pdf_path_2):
#         """
#         Main comparison workflow.
        
#         Args:
#             pdf_path_1: Path to first PDF
#             pdf_path_2: Path to second PDF
        
#         Returns:
#             dict with comparison results
#         """
#         print("\n" + "="*60)
#         print("PDF COMPARISON TOOL")
#         print("="*60)
        
#         # Step 1: Extract from both PDFs
#         print(f"\nExtracting data from {Path(pdf_path_1).name}...")
#         data_1 = self.extractor.extract(pdf_path_1)
#         print(f"  - Found {len(data_1['tables'])} tables")
#         print(f"  - Found {len(data_1['text'])} text paragraphs")
        
#         print(f"\nExtracting data from {Path(pdf_path_2).name}...")
#         data_2 = self.extractor.extract(pdf_path_2)
#         print(f"  - Found {len(data_2['tables'])} tables")
#         print(f"  - Found {len(data_2['text'])} text paragraphs")
        
#         # Step 2: Compare tables
#         print("\nComparing tables...")
#         table_differences = self.table_comparator.compare(
#             data_1['tables'],
#             data_2['tables'],
#             data_1['file_name'],
#             data_2['file_name']
#         )
#         print(f"  - Found {len(table_differences)} table differences")
        
#         # Step 3: Compare text
#         print("\nComparing text...")
#         text_differences = self.text_comparator.compare(
#             data_1['text'],
#             data_2['text'],
#             data_1['file_name'],
#             data_2['file_name']
#         )
#         print(f"  - Found {len(text_differences)} text differences")
        
#         # Step 4: Generate outputs
#         print("\nGenerating results...")
#         table_output = None
#         text_output = None
        
#         if table_differences:
#             table_output = self.output_generator.generate_table_results(
#                 table_differences,
#                 data_1['file_name'],
#                 data_2['file_name']
#             )
#             print(f"  - Tables file: {table_output.name}")
        
#         if text_differences:
#             text_output = self.output_generator.generate_text_results(
#                 text_differences,
#                 data_1['file_name'],
#                 data_2['file_name']
#             )
#             print(f"  - Text file: {text_output.name}")
        
#         # Step 5: Summary
#         print("\n" + "="*60)
#         print("COMPARISON COMPLETE")
#         print("="*60)
#         summary = self.output_generator.generate_comparison_summary(
#             len(table_differences),
#             len(text_differences),
#             data_1['file_name'],
#             data_2['file_name']
#         )
#         print(summary)
        
#         return {
#             "table_differences": table_differences,
#             "text_differences": text_differences,
#             "table_output_file": table_output,
#             "text_output_file": text_output
#         }


# def main():
#     """Command line entry point."""
#     if len(sys.argv) != 3:
#         print("Usage: python main.py <pdf_file_1> <pdf_file_2>")
#         print("\nExample: python main.py document1.pdf document2.pdf")
#         sys.exit(1)
    
#     pdf_1 = sys.argv[1]
#     pdf_2 = sys.argv[2]
    
#     # Validate files exist
#     if not Path(pdf_1).exists():
#         print(f"Error: File not found - {pdf_1}")
#         sys.exit(1)
    
#     if not Path(pdf_2).exists():
#         print(f"Error: File not found - {pdf_2}")
#         sys.exit(1)
    
#     # Run comparison
#     tool = PDFComparisonTool()
#     try:
#         tool.compare_pdfs(pdf_1, pdf_2)
#     except Exception as e:
#         print(f"\nError during comparison: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)


# if __name__ == "__main__":
#     main()

import sys
import os
from pathlib import Path
from pdf_extractor import PDFExtractor
from table_comparator import TableComparator
from text_comparator import TextComparator
from output_generator import OutputGenerator

def main():
    """
    Main orchestrator for PDF comparison tool.
    Usage: python main.py pdf1.pdf pdf2.pdf
    """
    
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python main.py <pdf1.pdf> <pdf2.pdf>")
        print("Example: python main.py document1.pdf document2.pdf")
        sys.exit(1)
    
    pdf1_path = sys.argv[1]
    pdf2_path = sys.argv[2]
    
    # Validate files exist
    if not os.path.exists(pdf1_path) or not os.path.exists(pdf2_path):
        print("Error: PDF files not found")
        sys.exit(1)
    
    # Create results directory
    results_dir = "results"
    Path(results_dir).mkdir(exist_ok=True)
    
    # Get file names without path
    pdf1_name = Path(pdf1_path).stem
    pdf2_name = Path(pdf2_path).stem
    
    print(f"\n{'='*60}")
    print(f"PDF Comparison Tool")
    print(f"{'='*60}")
    print(f"Comparing: {pdf1_name} vs {pdf2_name}\n")
    
    # Extract from PDF1
    print(f"Extracting data from {pdf1_name}...")
    extractor1 = PDFExtractor(pdf1_path)
    tables1, text1 = extractor1.extract_all()
    
    # Extract from PDF2
    print(f"\nExtracting data from {pdf2_name}...")
    extractor2 = PDFExtractor(pdf2_path)
    tables2, text2 = extractor2.extract_all()
    
    # Compare tables
    print(f"\n{'='*60}")
    print("Comparing Tables...")
    print(f"{'='*60}")
    table_comparator = TableComparator()
    table_comparator.load_tables(tables1, tables2)
    different_tables = table_comparator.compare()
    print(f"Found {len(different_tables)} rows with differences")
    
    # Compare text
    print(f"\n{'='*60}")
    print("Comparing Text...")
    print(f"{'='*60}")
    text_comparator = TextComparator()
    text_comparator.load_text(text1, text2)
    different_texts = text_comparator.compare()
    print(f"Found {len(different_texts)} sentences with differences")
    
    # Generate results
    print(f"\n{'='*60}")
    print("Generating Results...")
    print(f"{'='*60}")
    output_gen = OutputGenerator(results_dir, pdf1_name, pdf2_name)
    output_gen.generate_table_results(different_tables)
    output_gen.generate_text_results(different_texts)
    
    print(f"\n{'='*60}")
    print("Comparison Complete!")
    print(f"Results saved in: {results_dir}/")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()