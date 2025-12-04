#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF COMPARISON TOOL v6.0 - ONLY SHOW CHANGES (Red = New/Added)
===============================================================
- Non-linear PDFs fully supported (content can move anywhere)
- Common content (even if moved) is ignored in final report
- Only added, removed, modified content is shown
- Red = added in new PDF | Red strikethrough = removed | Green = replacement in modified
"""

import os
import sys
import hashlib
import re
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Any
import logging
import gc

if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

try:
    import pdfplumber
    from docx import Document
    from docx.shared import RGBColor
    from fuzzywuzzy import fuzz
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing package: {e}\nInstall: pip install pdfplumber python-docx fuzzywuzzy python-Levenshtein tqdm", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ============================== CORE CLASSES ==============================

class TextSegment:
    def __init__(self, text: str, page: int):
        self.text = text.strip()
        self.page = page
        self.hash = hashlib.sha256(self.text.encode('utf-8', errors='replace')).hexdigest()
        norm = re.sub(r'\s+', ' ', self.text.lower())
        self.normalized = re.sub(r'[^\w\s]', '', norm)

    def similarity(self, other) -> float:
        if not self.normalized or not other.normalized:
            return 0.0
        return fuzz.ratio(self.normalized, other.normalized) / 100.0


class TableContent:
    def __init__(self, table: List[List[str]], page: int, table_id: int):
        self.page = page
        self.table_id = table_id
        self.table = [[c.strip() if c else "" for c in row] for row in table if any(c.strip() for c in row)]
        table_str = json.dumps(self.table, ensure_ascii=False)
        self.hash = hashlib.sha256(table_str.encode('utf-8')).hexdigest()
        flat = " ".join(c for row in self.table for c in row if c)
        self.normalized = re.sub(r'[^\w\s]', '', flat.lower())

    def similarity(self, other) -> float:
        return fuzz.ratio(self.normalized, other.normalized) / 100.0


class PDFExtractor:
    def __init__(self, path: str):
        self.path = path

    def extract(self) -> Tuple[List[TextSegment], List[TableContent]]:
        text_segs = []
        tables = []

        with pdfplumber.open(self.path) as pdf:
            for page_num, page in enumerate(tqdm(pdf.pages, desc=f"Extracting {Path(self.path).name}", unit="page")):
                p = page_num + 1

                # Tables
                for tid, tbl in enumerate(page.extract_tables()):
                    if tbl and any(any(cell for cell in row) for row in tbl):
                        tables.append(TableContent(tbl, p, tid))

                # Text - paragraph blocks
                text = page.extract_text()
                if text:
                    blocks = [b.strip() for b in re.split(r'\n\s*\n', text) if len(b.strip()) > 30]
                    for block in blocks:
                        text_segs.append(TextSegment(block, p))

        logger.info(f"Extracted {len(text_segs)} text blocks, {len(tables)} tables from {Path(self.path).name}")
        return text_segs, tables


class SmartComparator:
    FUZZY_THRESHOLD = 0.88   # 88%+ = considered same (handles minor edits)
    MODIFIED_THRESHOLD = 0.99  # Below this + above FUZZY = modified

    def __init__(self, text1, tables1, text2, tables2):
        self.text1, self.tables1 = text1, tables1
        self.text2, self.tables2 = text2, tables2

    def compare(self):
        added_text, removed_text, modified_text = self._compare_list(self.text1, self.text2)
        added_tables, removed_tables, modified_tables = self._compare_list(self.tables1, self.tables2, is_table=True)

        return {
            'added_text': sorted(added_text, key=lambda x: x.page),
            'removed_text': sorted(removed_text, key=lambda x: x.page),
            'modified_text': modified_text,  # list of (old, new, score)
            'added_tables': added_tables,
            'removed_tables': removed_tables,
            'modified_tables': modified_tables,
        }

    def _compare_list(self, list1, list2, is_table=False):
        # Mark exact matches
        hash_to_item2 = {item.hash: item for item in list2}
        matched2 = set()

        for item1 in list1:
            if item1.hash in hash_to_item2:
                matched2.add(hash_to_item2[item1.hash])

        unmatched1 = [item for item in list1 if item.hash not in hash_to_item2 or hash_to_item2[item.hash] not in matched2]
        unmatched2 = [item for item in list2 if item not in matched2]

        modified = []
        used2_indices = set()

        # Fuzzy match remaining
        for item1 in unmatched1:
            best_score = 0
            best_item2 = None
            best_idx = -1

            for idx, item2 in enumerate(unmatched2):
                if idx in used2_indices:
                    continue
                score = item1.similarity(item2)
                if score > best_score:
                    best_score = score
                    best_item2 = item2
                    best_idx = idx

            if best_score >= self.FUZZY_THRESHOLD:
                used2_indices.add(best_idx)
                if best_score < self.MODIFIED_THRESHOLD:
                    modified.append((item1, best_item2, best_score))

        # Final classification
        added = [unmatched2[i] for i in range(len(unmatched2)) if i not in used2_indices]
        removed = [item for item in unmatched1 if not any(item is old for old, _, _ in modified)]

        return added, removed, modified


class ChangesOnlyReport:
    def __init__(self, results: dict, pdf1: str, pdf2: str):
        self.results = results
        self.pdf1_name = Path(pdf1).name
        self.pdf2_name = Path(pdf2).name
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def generate_word_report(self, output_path: str = None):
        output_path = output_path or f"PDF_Changes_Only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
        doc = Document()
        doc.add_heading("PDF Changes Report (Only Differences Shown)", 0)

        doc.add_paragraph(f"Old: {self.pdf1_name}\nNew: {self.pdf2_name}\nGenerated: {self.timestamp}\n")

        total_changes = (
            len(self.results['added_text']) + len(self.results['removed_text']) +
            len(self.results['added_tables']) + len(self.results['removed_tables']) +
            len(self.results['modified_text']) + len(self.results['modified_tables'])
        )
        doc.add_paragraph(f"Total changes detected: {total_changes}", style='Intense Quote')

        # === Added Content (Red) ===
        if self.results['added_text'] or self.results['added_tables']:
            doc.add_heading("NEW CONTENT (Added in new PDF)", level=1)
            for seg in self.results['added_text']:
                p = doc.add_paragraph(f"→ Page {seg.page}: ", style='List Bullet')
                run = p.add_run(seg.text)
                run.font.color.rgb = RGBColor(200, 0, 0)   # Red
                run.bold = True

            for tbl in self.results['added_tables']:
                doc.add_paragraph(f"→ Page {tbl.page} - New Table:", style='List Bullet')
                table = doc.add_table(rows=len(tbl.table), cols=len(tbl.table[0]))
                for i, row in enumerate(tbl.table):
                    cells = table.rows[i].cells
                    for j, cell in enumerate(row):
                        cells[j].text = cell or ""
                        for para in cells[j].paragraphs:
                            for run in para.runs:
                                run.font.color.rgb = RGBColor(200, 0, 0)
                                run.bold = True

        # === Removed Content (Red + Strikethrough) ===
        if self.results['removed_text'] or self.results['removed_tables']:
            doc.add_heading("REMOVED CONTENT (No longer in new PDF)", level=1)
            for seg in self.results['removed_text']:
                p = doc.add_paragraph(f"← Page {seg.page}: ", style='List Bullet')
                run = p.add_run(seg.text)
                run.font.color.rgb = RGBColor(200, 0, 0)
                run.font.strike = True
                run.italic = True

            for tbl in self.results['removed_tables']:
                doc.add_paragraph(f"← Page {tbl.page} - Removed Table:", style='List Bullet')
                table = doc.add_table(rows=len(tbl.table), cols=len(tbl.table[0]))
                for i, row in enumerate(tbl.table):
                    cells = table.rows[i].cells
                    for j, cell in enumerate(row):
                        cells[j].text = cell or ""
                        for para in cells[j].paragraphs:
                            for run in para.runs:
                                run.font.color.rgb = RGBColor(200, 0, 0)
                                run.font.strike = True

        # === Modified Content ===
        if self.results['modified_text'] or self.results['modified_tables']:
            doc.add_heading("MODIFIED CONTENT", level=1)
            for old, new, score in self.results['modified_text']:
                doc.add_paragraph(f"Page {old.page} → Page {new.page} (similarity: {score:.1%})")
                p = doc.add_paragraph("Old: ")
                run = p.add_run(old.text)
                run.font.color.rgb = RGBColor(200, 0, 0)
                run.font.strike = True

                p = doc.add_paragraph("New: ")
                run = p.add_run(new.text)
                run.font.color.rgb = RGBColor(0, 100, 0)
                run.bold = True
                doc.add_paragraph("")

        doc.save(output_path)
        logger.info(f"Changes-only report generated: {output_path}")
        return output_path


# ============================== MAIN ==============================

def compare_pdfs(pdf1_path: str, pdf2_path: str, output_dir: str = "."):
    if not all(os.path.exists(p) for p in [pdf1_path, pdf2_path]):
        print("Error: One or both PDFs not found!")
        return

    logger.info("Starting intelligent PDF comparison...")

    # Extract
    ext1 = PDFExtractor(pdf1_path)
    ext2 = PDFExtractor(pdf2_path)
    text1, tables1 = ext1.extract()
    text2, tables2 = ext2.extract()

    # Compare
    comparator = SmartComparator(text1, tables1, text2, tables2)
    results = comparator.compare()

    # Generate report (only changes)
    reporter = ChangesOnlyReport(results, pdf1_path, pdf2_path)
    report_path = reporter.generate_word_report(os.path.join(output_dir, f"Changes_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"))

    # Summary
    added = len(results['added_text']) + len(results['added_tables'])
    removed = len(results['removed_text']) + len(results['removed_tables'])
    modified = len(results['modified_text']) + len(results['modified_tables'])

    print(f"\nCOMPARISON COMPLETE!")
    print(f"Added content: {added}")
    print(f"Removed content: {removed}")
    print(f"Modified content: {modified}")
    print(f"Report (only changes): {report_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare two PDFs - show ONLY changes in red")
    parser.add_argument("old_pdf", help="Path to old/original PDF")
    parser.add_argument("new_pdf", help="Path to new/modified PDF")
    parser.add_argument("--output", "-o", default=".", help="Output directory")
    args = parser.parse_args()

    compare_pdfs(args.old_pdf, args.new_pdf, args.output)
