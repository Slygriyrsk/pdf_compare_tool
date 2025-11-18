"""
Improved PDF comparison tool
- Paragraph-aware (page and paragraph indices preserved)
- Robust matching with optional TF-IDF / semantic embeddings
- Bipartite matching (Hungarian) to align paragraphs across PDFs so moved content is detected
- Produces a clear PDF report with: page numbers, paragraph indices, similarity%, status (unchanged, modified, moved, added, removed)
- Also writes a JSON details file for downstream processing

Usage (CLI):
    python pdf_diff_tool_improved.py old.pdf new.pdf --mode both --window 1 --threshold 0.45 --output diff

Dependencies (recommended):
    pip install PyMuPDF reportlab scikit-learn scipy sentence-transformers numpy

If optional deps are missing the tool falls back to simpler methods.
"""

import argparse
import json
import html
import os
import re
import sys
import math
import difflib
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict

# PDF text extraction / writing
try:
    import fitz  # PyMuPDF
except Exception as e:
    print("ERROR: PyMuPDF (fitz) is required. Install with: pip install PyMuPDF")
    raise e

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import enums
    from reportlab.lib.units import mm
    from reportlab.lib.colors import black, red, green, grey
except Exception as e:
    print("ERROR: reportlab is required. Install with: pip install reportlab")
    raise e

# optional
_has_sklearn = False
_has_scipy = False
_has_transformers = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _has_sklearn = True
except Exception:
    _has_sklearn = False

try:
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    _has_scipy = True
except Exception:
    _has_scipy = False

try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    _has_transformers = True
except Exception:
    _has_transformers = False

# -------------------------
# Utilities
# -------------------------

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r'\r\n?', '\n', s)
    s = re.sub(r'\u00A0', ' ', s)  # non-breaking space
    # collapse multiple newlines to paragraph separator
    s = re.sub(r'\n{3,}', '\n\n', s)
    # collapse whitespace
    s = re.sub(r'[ \t]+', ' ', s)
    return s.strip()


def split_into_paragraphs(page_text: str, min_words: int = 3) -> List[str]:
    """Split page text into paragraphs. Keep paragraphs with at least min_words words."""
    if not page_text:
        return []
    # first split on double-newline
    parts = [p.strip() for p in re.split(r'\n\s*\n', page_text) if p.strip()]
    out = []

    for p in parts:
    # further split very long paragraphs by sentence boundaries if needed
        if len(p.split()) > 200:
            # naive sentence split by .!? followed by space + capital
            segs = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\)])', p)
            segs = [s.strip() for s in segs if len(s.split()) >= min_words]
            out.extend(segs)
        else:
            if len(p.split()) >= min_words:
                out.append(p)
    return out


def extract_pages_paragraphs(pdf_path: str) -> Tuple[List[str], List[List[str]]]:
    """Return (pages_text, paragraphs_per_page) where paragraphs_per_page is a list of lists.
    Pages are 0-indexed in the returned arrays but user-facing numbers will be 1-indexed.
    """
    doc = fitz.open(pdf_path)
    pages_text = []
    paragraphs_per_page = []
    for p in range(len(doc)):
        raw = doc.load_page(p).get_text("text") or ""
        raw = normalize_text(raw)
        pages_text.append(raw)
        paras = split_into_paragraphs(raw)
        paragraphs_per_page.append(paras)
    doc.close()
    return pages_text, paragraphs_per_page

# -------------------------
# Similarity engines
# -------------------------

def similarity_diff(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def build_tfidf_matrix(corpus_a: List[str], corpus_b: List[str]):
    if not _has_sklearn:
        raise RuntimeError("TF-IDF requires scikit-learn")
    vec = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=1)
    X = vec.fit_transform(corpus_a + corpus_b)
    A = X[:len(corpus_a)]
    B = X[len(corpus_a):]
    sim = cosine_similarity(A, B)
    return sim


def build_semantic_matrix(corpus_a: List[str], corpus_b: List[str], model_name='all-MiniLM-L6-v2'):
    if not _has_transformers:
        raise RuntimeError("Semantic mode requires sentence-transformers")
    model = SentenceTransformer(model_name)
    emb_a = model.encode(corpus_a, convert_to_tensor=True, show_progress_bar=False)
    emb_b = model.encode(corpus_b, convert_to_tensor=True, show_progress_bar=False)
    cos = util.cos_sim(emb_a, emb_b)
    if isinstance(cos, torch.Tensor):
        cos = cos.cpu().numpy()
    return cos, model

# -------------------------
# Matching and classification
# -------------------------

def bipartite_match(sim_matrix: 'np.ndarray', threshold: float = 0.3) -> List[Tuple[int,int,float]]:
    """Return list of matches (i, j, sim) using Hungarian assignment on -sim costs if scipy available.
    Otherwise fall back to greedy matching.
    Only matches with sim >= threshold are returned.
    """
    matches = []
    if _has_scipy:
        # convert to cost (maximize sim => minimize cost=-sim)
        cost = -np.array(sim_matrix)
        # Hungarian handles rectangular matrices
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            sim = float(sim_matrix[r][c])
            if sim >= threshold:
                matches.append((r, c, sim))
    else:
        # greedy
        used_cols = set()
        for i, row in enumerate(sim_matrix):
            best_j = max(range(len(row)), key=lambda j: row[j]) if row else -1
            sim = row[best_j] if row and best_j >= 0 else 0.0
            if sim >= threshold and best_j not in used_cols:
                matches.append((i, best_j, sim))
                used_cols.add(best_j)
    return matches


def classify_matches(
    matches: List[Tuple[Optional[int], Optional[int], float]],
    idx_to_page_para_a: Dict[int, Tuple[int, int]],
    idx_to_page_para_b: Dict[int, Tuple[int, int]]
) -> List[Dict[str, Any]]:
    """
    Label matched or unmatched paragraphs:
      - 'unchanged' if same page and high sim
      - 'modified' if similar but not identical
      - 'moved' if page changed but high sim
      - 'added' if only in PDF B
      - 'removed' if only in PDF A

    idx_to_page_para_*: paragraph-global-index -> (page_num, para_idx_on_page)
    """
    out = []
    for match in matches:
        # Support tuples like (i, j, sim) or (i, j, sim, status)
        if len(match) >= 3:
            i, j, sim = match[:3]
        else:
            continue

        # Handle unmatched paragraphs safely
        if i is None and j is not None:
            page_b, para_b = idx_to_page_para_b[j]
            out.append({
                'a_idx': None, 'b_idx': j,
                'a_page': None, 'b_page': page_b,
                'a_para': None, 'b_para': para_b,
                'similarity': 0.0,
                'status': 'added'
            })
            continue

        if j is None and i is not None:
            page_a, para_a = idx_to_page_para_a[i]
            out.append({
                'a_idx': i, 'b_idx': None,
                'a_page': page_a, 'b_page': None,
                'a_para': para_a, 'b_para': None,
                'similarity': 0.0,
                'status': 'removed'
            })
            continue

        # Normal matched pair
        page_a, para_a = idx_to_page_para_a.get(i, (None, None))
        page_b, para_b = idx_to_page_para_b.get(j, (None, None))

        if page_a is None or page_b is None:
            continue

        # Classification logic
        if sim > 0.95 and page_a == page_b:
            status = 'unchanged'
        elif page_a != page_b and sim >= 0.9:
            status = 'moved'
        elif sim >= 0.5:
            status = 'modified'
        else:
            status = 'different'

        out.append({
            'a_idx': i, 'b_idx': j,
            'a_page': page_a, 'b_page': page_b,
            'a_para': para_a, 'b_para': para_b,
            'similarity': sim,
            'status': status
        })
    return out

# -------------------------
# Human diff at paragraph level
# -------------------------

def human_diff_paragraph(a: str, b: str, min_words: int = 1) -> Tuple[List[str], List[str]]:
    removed, added = [], []
    diff = list(difflib.ndiff(a.splitlines(), b.splitlines()))
    for line in diff:
        if line.startswith('- '):
            c = line[2:].strip()
            if len(c.split()) >= min_words:
                removed.append(c)
        elif line.startswith('+ '):
            c = line[2:].strip()
            if len(c.split()) >= min_words:
                added.append(c)
    # dedupe
    def dedup(lst):
        out = []
        prev = None
        for l in lst:
            if l != prev:
                out.append(l)
            prev = l
        return out
    return dedup(removed), dedup(added)

# -------------------------
# Report generator
# -------------------------

def build_pdf_report(report_path: str, title: str, details: List[Dict[str,Any]], corpus_a: List[str], corpus_b: List[str], idx_to_page_para_a: Dict[int,Tuple[int,int]], idx_to_page_para_b: Dict[int,Tuple[int,int]]):
    doc = SimpleDocTemplate(report_path, pagesize=A4, rightMargin=15*mm, leftMargin=15*mm)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(html.escape(title), styles['Title']))
    story.append(Spacer(1,6))

    # summary counts
    counts = defaultdict(int)
    for d in details:
        counts[d['status']] += 1
    summary = f"Matches: {len(details)} — Unchanged: {counts['unchanged']}, Moved: {counts['moved']}, Modified: {counts['modified']}, Added/Removed: {counts.get('added',0)+counts.get('removed',0)}"
    story.append(Paragraph(summary, styles['Normal']))
    story.append(Spacer(1,8))

    # table header
    hdr_data = [['#', 'PDF1 (page:para)', 'PDF2 (page:para)', 'Similarity', 'Status', 'Removed →', 'Added ←']]
    # we'll append a compact table for each match and then full paragraphs below
    tbl_style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), grey),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.25, black),
    ])

    rows = []
    for idx, d in enumerate(details, start=1):
        a_label = f"p{d['a_page']}:{d['a_para']}"
        b_label = f"p{d['b_page']}:{d['b_para']}"
        sim_pct = d['similarity']*100
        removed, added = human_diff_paragraph(corpus_a[d['a_idx']], corpus_b[d['b_idx']])
        rows.append([str(idx), a_label, b_label, f"{sim_pct:.1f}%", d['status'], '\n'.join(removed[:3]) if removed else '', '\n'.join(added[:3]) if added else ''])
    if rows:
        tbl = Table(hdr_data + rows, colWidths=[20*mm, 40*mm, 40*mm, 30*mm, 30*mm, 50*mm, 50*mm])
        tbl.setStyle(tbl_style)
        story.append(tbl)
        story.append(Spacer(1,10))

    # Detailed sections for each match
    for idx, d in enumerate(details, start=1):
        story.append(Paragraph(f"<b>Match {idx} — Status: {d['status'].upper()} — Sim: {d['similarity']*100:.2f}%</b>", styles['Heading3']))
        story.append(Paragraph(f"PDF1 — page {d['a_page']}, paragraph {d['a_para']}", styles['Normal']))
        story.append(Paragraph(html.escape(corpus_a[d['a_idx']])[:300] + ("..." if len(corpus_a[d['a_idx']])>300 else ""), styles['Normal']))
        story.append(Spacer(1,4))
        story.append(Paragraph(f"PDF2 — page {d['b_page']}, paragraph {d['b_para']}", styles['Normal']))
        story.append(Paragraph(html.escape(corpus_b[d['b_idx']])[:300] + ("..." if len(corpus_b[d['b_idx']])>300 else ""), styles['Normal']))
        story.append(Spacer(1,4))
        removed, added = human_diff_paragraph(corpus_a[d['a_idx']], corpus_b[d['b_idx']])
        if removed:
            story.append(Paragraph("<b>Removed from PDF2:</b>", styles['Normal']))
            for r in removed:
                story.append(Paragraph(f"- {html.escape(r)}", styles['Normal']))
        if added:
            story.append(Paragraph("<b>Added in PDF2:</b>", styles['Normal']))
            for a in added:
                story.append(Paragraph(f"+ {html.escape(a)}", styles['Normal']))
        story.append(Spacer(1,8))

    doc.build(story)
    print(f"[REPORT] Generated: {report_path}")

# -------------------------
# Orchestrator
# -------------------------

def run(pdf1: str, pdf2: str, mode: str = 'both', window_size: int = 1, threshold: float = 0.45, output_prefix: str = 'diff_report', semantic_model_name: str = 'all-MiniLM-L6-v2'):
    print('[INFO] Extracting pages & paragraphs...')
    pages1, paras1 = extract_pages_paragraphs(pdf1)
    pages2, paras2 = extract_pages_paragraphs(pdf2)
    total_paras_a = sum(len(p) for p in paras1)
    total_paras_b = sum(len(p) for p in paras2)
    print(f"PDF1 pages={len(pages1)} paras={total_paras_a}; PDF2 pages={len(pages2)} paras={total_paras_b}")

    # flatten paragraphs into corpus arrays while keeping mapping to page/para indices
    corpus_a = []
    corpus_b = []
    idx_to_page_para_a = {}
    idx_to_page_para_b = {}
    cnt = 0
    for pi, page_paras in enumerate(paras1, start=1):
        for pj, ptxt in enumerate(page_paras, start=1):
            corpus_a.append(ptxt)
            idx_to_page_para_a[cnt] = (pi, pj)
            cnt += 1
    cnt = 0
    for pi, page_paras in enumerate(paras2, start=1):
        for pj, ptxt in enumerate(page_paras, start=1):
            corpus_b.append(ptxt)
            idx_to_page_para_b[cnt] = (pi, pj)
            cnt += 1

    # choose similarity matrix method
    sim_matrix = None
    if mode in ('tfidf','both') and _has_sklearn:
        print('[INFO] Building TF-IDF similarity matrix...')
        sim_matrix = build_tfidf_matrix(corpus_a, corpus_b)
    if mode in ('semantic','both') and _has_transformers:
        print('[INFO] Building semantic similarity matrix...')
        sem_mat, model = build_semantic_matrix(corpus_a, corpus_b, model_name=semantic_model_name)
        # if sim_matrix already exists, average them
        if sim_matrix is None:
            sim_matrix = sem_mat
        else:
            # simple average
            sim_matrix = (np.array(sim_matrix) + np.array(sem_mat)) / 2.0
    if sim_matrix is None:
        print('[WARN] No advanced similarity available — falling back to difflib pairwise (slower).')
        # build pairwise similarity in pure python
        sim_matrix = []
        for a in corpus_a:
            row = [similarity_diff(a, b) for b in corpus_b]
            sim_matrix.append(row)

    # matching
    print('[INFO] Performing bipartite matching (to detect moves rather than add/remove)...')
    matches = bipartite_match(sim_matrix, threshold=threshold)
    matches_info = classify_matches(matches, idx_to_page_para_a, idx_to_page_para_b)

    # determine unmatched items (added/removed)
    matched_a = set(m['a_idx'] for m in matches_info)
    matched_b = set(m['b_idx'] for m in matches_info)
    removed_items = [i for i in range(len(corpus_a)) if i not in matched_a]
    added_items = [j for j in range(len(corpus_b)) if j not in matched_b]

    # Build details list
    details = []
    for m in matches_info:
        details.append(m)
    # add explicit removed/added entries
    for i in removed_items:
        page, para = idx_to_page_para_a[i]
        details.append({'a_idx': i, 'b_idx': None, 'a_page': page, 'b_page': None, 'a_para': para, 'b_para': None, 'similarity': 0.0, 'status': 'removed'})
    for j in added_items:
        page, para = idx_to_page_para_b[j]
        details.append({'a_idx': None, 'b_idx': j, 'a_page': None, 'b_page': page, 'a_para': None, 'b_para': para, 'similarity': 0.0, 'status': 'added'})

    # sort details for stable report order (by a_page then b_page)
    def sort_key(d):
        a_page = d['a_page'] if d['a_page'] is not None else math.inf
        b_page = d['b_page'] if d['b_page'] is not None else math.inf
        return (a_page, b_page)
    details = sorted(details, key=sort_key)

    # output files
    out_pdf = f"{output_prefix}.pdf"
    out_json = f"{output_prefix}.json"

    print(f"[INFO] Generating report PDF -> {out_pdf}")
    build_pdf_report(out_pdf, f"PDF Diff Report: {os.path.basename(pdf1)} → {os.path.basename(pdf2)}", details, corpus_a, corpus_b, idx_to_page_para_a, idx_to_page_para_b)

    # write JSON
    with open(out_json, 'w', encoding='utf8') as fh:
        json.dump({'pdf1': pdf1, 'pdf2': pdf2, 'details': details}, fh, indent=2, ensure_ascii=False)
    print(f"[INFO] JSON details -> {out_json}")
    print('[DONE]')

# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Paragraph-aware PDF diff tool')
    p.add_argument('pdf1')
    p.add_argument('pdf2')
    p.add_argument('--mode', choices=['simple','tfidf','semantic','both'], default='both')
    p.add_argument('--window', type=int, default=1, help='unused currently; placeholder for page-window block creation')
    p.add_argument('--threshold', type=float, default=0.45)
    p.add_argument('--output', default='diff_report')
    p.add_argument('--semantic_model', default='all-MiniLM-L6-v2')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    try:
        run(args.pdf1, args.pdf2, mode=args.mode, window_size=args.window, threshold=args.threshold, output_prefix=args.output, semantic_model_name=args.semantic_model)
    except KeyboardInterrupt:
        print('\n[ABORTED]')
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)