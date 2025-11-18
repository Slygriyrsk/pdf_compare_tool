import argparse
import html
import re
import sys
from collections import OrderedDict, namedtuple
from typing import List, Tuple, Dict, Optional
from reportlab.lib.styles import ParagraphStyle

import fitz  # PyMuPDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepInFrame
)
from reportlab.lib.units import mm
import difflib

# -------------------- Optional deps --------------------
_has_sklearn = False
_has_transformers = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _has_sklearn = True
except Exception:
    pass

try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    _has_transformers = True
except Exception:
    pass

# -------------------- Data structures --------------------
PdfParagraph = namedtuple("PdfParagraph", "page idx text")
Block = namedtuple("Block", "label start_page end_page paragraphs text")

# -------------------- Helpers --------------------
def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\r\n?", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def split_paragraphs(page_text: str) -> List[str]:
    """Very light paragraph detection – split on double newlines or sentence ends."""
    paras = re.split(r"\n\s*\n", page_text)
    out = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        # split long single-line paragraphs into sentences
        if "\n" not in p and len(p.split()) > 30:
            sentences = re.split(r"(?<=[.!?])\s+", p)
            out.extend([s.strip() for s in sentences if s.strip()])
        else:
            out.append(p)
    return [p for p in out if p]

# -------------------- 1. Extraction --------------------
def extract_pages_with_paragraphs(pdf_path: str) -> List[List[Paragraph]]:
    doc = fitz.open(pdf_path)
    pages_paras: List[List[Paragraph]] = []
    for pno in range(len(doc)):
        page = doc.load_page(pno)
        raw = page.get_text("text") or ""
        raw = normalize_text(raw)
        paras = split_paragraphs(raw)
        page_paras = [Paragraph(page=pno + 1, idx=i + 1, text=t) for i, t in enumerate(paras)]
        pages_paras.append(page_paras)
    doc.close()
    return pages_paras

# -------------------- 2. Windowed blocks --------------------
def build_windowed_blocks(pages_paras: List[List[Paragraph]],
                         window_size: int = 2) -> List[Block]:
    blocks: List[Block] = []
    n = len(pages_paras)
    if n == 0:
        return blocks

    for i in range(max(1, n - window_size + 1)):
        start_pg = i + 1
        end_pg = i + window_size
        block_paras: List[Paragraph] = []
        for pg in pages_paras[i:i + window_size]:
            block_paras.extend(pg)
        combined = "\n".join(p.text for p in block_paras)
        label = f"{start_pg}-{end_pg}"
        blocks.append(Block(label=label,
                            start_page=start_pg,
                            end_page=end_pg,
                            paragraphs=block_paras,
                            text=normalize_text(combined)))
    return blocks

# -------------------- 3. Global page alignment (Needleman-Wunsch) --------------------
def global_page_alignment(pages1: List[str], pages2: List[str],
                          gap_penalty: float = -1.0) -> List[Tuple[int, int]]:
    """
    Returns a list of (idx1, idx2) pairs that represent the best monotonic mapping.
    Unmatched pages get (-1, j) or (i, -1).
    """
    import numpy as np
    m, n = len(pages1), len(pages2)
    dp = np.zeros((m + 1, n + 1), dtype=float)
    dp[:, 0] = np.linspace(0, gap_penalty * m, m + 1)
    dp[0, :] = np.linspace(0, gap_penalty * n, n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            score = difflib.SequenceMatcher(None, pages1[i - 1], pages2[j - 1]).ratio()
            match = dp[i - 1, j - 1] + score
            delete = dp[i - 1, j] + gap_penalty
            insert = dp[i, j - 1] + gap_penalty
            dp[i, j] = max(match, delete, insert)

    # back-trace
    align: List[Tuple[int, int]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            score = difflib.SequenceMatcher(None, pages1[i - 1], pages2[j - 1]).ratio()
            if dp[i, j] == dp[i - 1, j - 1] + score:
                align.append((i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and dp[i, j] == dp[i - 1, j] + gap_penalty:
            align.append((i - 1, -1))
            i -= 1
        else:
            align.append((-1, j - 1))
            j -= 1
    align.reverse()
    return align

# -------------------- 4. Similarity engines --------------------
def similarity_simple(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()

def similarity_tfidf(blocks1: List[str], blocks2: List[str]) -> List[List[float]]:
    if not _has_sklearn:
        raise RuntimeError("scikit-learn required for TF-IDF")
    corpus = blocks1 + blocks2
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vec.fit_transform(corpus)
    sim = cosine_similarity(X[: len(blocks1)], X[len(blocks1) :])
    return sim.tolist()

def similarity_semantic(blocks1: List[str], blocks2: List[str],
                        model_name: str = "all-MiniLM-L6-v2"):
    if not _has_transformers:
        raise RuntimeError("sentence-transformers required")
    model = SentenceTransformer(model_name)
    e1 = model.encode(blocks1, convert_to_tensor=True, show_progress_bar=False)
    e2 = model.encode(blocks2, convert_to_tensor=True, show_progress_bar=False)
    cos = util.cos_sim(e1, e2)
    return cos.cpu().numpy().tolist()

# -------------------- 5. Matching --------------------
def greedy_match(sim_matrix: List[List[float]], threshold: float) -> Dict[int, Tuple[int, float]]:
    mapping: Dict[int, Tuple[int, float]] = {}
    used_cols = set()
    for i, row in enumerate(sim_matrix):
        best_j, best = max(enumerate(row), key=lambda x: x[1])
        if best >= threshold and best_j not in used_cols:
            mapping[i] = (best_j, best)
            used_cols.add(best_j)
    return mapping

# -------------------- 6. Diff (sentence aware) --------------------
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

def sentence_split(text: str) -> List[str]:
    return [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]

def diff_added_removed(text1: str, text2: str, min_words: int = 2) -> Tuple[List[str], List[str]]:
    lines1 = sentence_split(text1)
    lines2 = sentence_split(text2)
    removed, added = [], []
    for s in difflib.ndiff(lines1, lines2):
        if s.startswith("- "):
            if len(s[2:].split()) >= min_words:
                removed.append(s[2:].strip())
        elif s.startswith("+ "):
            if len(s[2:].split()) >= min_words:
                added.append(s[2:].strip())
    # dedup consecutive identical lines
    def dedup(lst):
        out, prev = [], None
        for x in lst:
            if x != prev:
                out.append(x)
                prev = x
        return out
    return dedup(removed), dedup(added)

# -------------------- 7. PDF Report --------------------
def _para_ref(p: Paragraph) -> str:
    return f"Page {p.page}, ¶{p.idx}"

def build_pdf_report(path: str, title: str, entries: List[Dict], mode_label: str):
    doc = SimpleDocTemplate(
        path,
        pagesize=A4,
        leftMargin=15*mm,
        rightMargin=15*mm,
        topMargin=20*mm,
        bottomMargin=18*mm
    )
    styles = getSampleStyleSheet()

    # === Create a proper Title style (cloned, not read-only) ===
    title_style = ParagraphStyle(
        name='CustomTitle',
        parent=styles['Title'],
        fontSize=14,
        leading=18,
        spaceAfter=10,
        alignment=1  # Center
    )

    story = []

    # === Header ===
    story.append(PdfParagraph(html.escape(title), title_style))
    story.append(PdfParagraph(f"<i>{html.escape(mode_label)}</i>", styles["Normal"]))
    story.append(Spacer(1, 8))

    if not entries:
        story.append(PdfParagraph("<i>No matches above threshold.</i>", styles["Normal"]))
    else:
        for ent in entries:
            # Block header
            story.append(PdfParagraph(
                html.escape(ent['block_label']), styles["Heading3"]
            ))
            sim_pct = ent['similarity'] * 100
            story.append(PdfParagraph(
                f"Matched <b>{html.escape(ent['match_label'])}</b> (sim {sim_pct:.1f}%)",
                styles["Normal"]
            ))
            story.append(Spacer(1, 4))

            # Changes table
            table_data = [["Location", "Removed", "Added"]]
            for loc, rem, add in ent['changes']:
                rem_para = PdfParagraph(
                    f"<font color='#B00020'>− {html.escape(rem)}</font>", styles["Normal"]
                ) if rem else PdfParagraph("—", styles["Normal"])
                add_para = PdfParagraph(
                    f"<font color='#006400'>+ {html.escape(add)}</font>", styles["Normal"]
                ) if add else PdfParagraph("—", styles["Normal"])
                table_data.append([loc, rem_para, add_para])

            t = Table(table_data, colWidths=[50*mm, 65*mm, 65*mm])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#DDDDDD')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 3),
                ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
            ]))
            story.append(t)
            story.append(Spacer(1, 10))

            if ent.get('unmatched'):
                story.append(PdfParagraph("<b>Unmatched PdfParagraphs:</b>", styles["Normal"]))
                for u in ent['unmatched']:
                    story.append(PdfParagraph(f"• {html.escape(u)}", styles["Normal"]))
                story.append(Spacer(1, 6))

            story.append(PageBreak())

    doc.build(story)
    print(f"[REPORT] Saved: {path}")

# -------------------- 8. Orchestrator --------------------
def run_comparison(pdf1: str, pdf2: str,
                   mode: str = "hybrid",
                   window: int = 2,
                   threshold: float = 0.38,
                   align_pages: bool = True,
                   min_sentence_words: int = 2,
                   output_prefix: str = "diff_report",
                   semantic_model: str = "all-MiniLM-L6-v2"):

    print("[1] Extracting pages + PdfParagraphs …")
    pages_paras1 = extract_pages_with_paragraphs(pdf1)
    pages_paras2 = extract_pages_with_paragraphs(pdf2)
    page_texts1 = [normalize_text("\n".join(p.text for p in pg)) for pg in pages_paras1]
    page_texts2 = [normalize_text("\n".join(p.text for p in pg)) for pg in pages_paras2]

    # ---- Global page alignment (optional) ----
    if align_pages:
        print("[2] Global page alignment …")
        page_map = global_page_alignment(page_texts1, page_texts2, gap_penalty=-0.8)
    else:
        page_map = [(i, i) for i in range(min(len(page_texts1), len(page_texts2)))]

    # Build blocks respecting the alignment
    blocks1 = build_windowed_blocks(pages_paras1, window)
    blocks2 = build_windowed_blocks(pages_paras2, window)

    # ---- Similarity matrix ----
    print(f"[3] Computing {mode.upper()} similarity …")
    if mode in ("simple", "hybrid"):
        sim_matrix = [[similarity_simple(b1.text, b2.text) for b2 in blocks2] for b1 in blocks1]
    elif mode == "tfidf":
        sim_matrix = similarity_tfidf([b.text for b in blocks1], [b.text for b in blocks2])
    elif mode == "semantic":
        sim_matrix = similarity_semantic([b.text for b in blocks1], [b.text for b in blocks2], semantic_model)
    elif mode == "hybrid":
        # TF-IDF filter → semantic re-rank
        tfidf_mat = similarity_tfidf([b.text for b in blocks1], [b.text for b in blocks2])
        candidates = {}
        for i, row in enumerate(tfidf_mat):
            for j, s in enumerate(row):
                if s >= threshold * 0.6:
                    candidates.setdefault(i, []).append((j, s))
        # re-rank candidates with semantic
        sem_mat = similarity_semantic([blocks1[i].text for i in candidates],
                                      [blocks2[j].text for i in candidates for j, _ in candidates[i]],
                                      semantic_model)
        # rebuild full matrix (low scores elsewhere)
        sim_matrix = [[0.0] * len(blocks2) for _ in blocks1]
        sem_idx = 0
        for i in candidates:
            for j, _ in candidates[i]:
                sim_matrix[i][j] = sem_mat[sem_idx]
                sem_idx += 1
    else:
        raise ValueError("Invalid mode")

    # ---- Greedy matching ----
    mapping = greedy_match(sim_matrix, threshold)

    # ---- Build report entries ----
    entries = []
    for i, (j, sim) in mapping.items():
        b1, b2 = blocks1[i], blocks2[j]

        # paragraph-level diff
        changes: List[Tuple[str, str, str]] = []   # (location, removed, added)
        unmatched: List[str] = []

        # map paragraphs by simple string equality first
        p1 = {p.text: p for p in b1.paragraphs}
        p2 = {p.text: p for p in b2.paragraphs}

        # removed
        for txt, p in p1.items():
            if txt not in p2:
                changes.append((_para_ref(p), txt, ""))
        # added
        for txt, p in p2.items():
            if txt not in p1:
                changes.append((_para_ref(p), "", txt))

        # for paragraphs that exist in both but differ → sentence diff
        for txt, pA in p1.items():
            if txt in p2:
                pB = p2[txt]
                if pA.text == pB.text:
                    continue
                rem, add = diff_added_removed(pA.text, pB.text, min_sentence_words)
                for r in rem:
                    changes.append((_para_ref(pA), r, ""))
                for a in add:
                    changes.append((_para_ref(pB), "", a))

        # keep order roughly by page
        changes.sort(key=lambda x: (int(x[0].split()[1].replace(',', '')), x[0]))

        entries.append({
            "block_label": f"PDF1 {b1.label} (pages {b1.start_page}-{b1.end_page})",
            "match_label": f"PDF2 {b2.label} (pages {b2.start_page}-{b2.end_page})",
            "similarity": sim,
            "changes": changes,
            "unmatched": unmatched,
        })

    # ---- Write reports ----
    mode_suffix = mode
    out_path = f"{output_prefix}_{mode_suffix}.pdf"
    build_pdf_report(out_path,
                      f"PDF Diff – {mode.upper()} Mode",
                      entries,
                      f"Window={window} | Threshold={threshold} | Align={'ON' if align_pages else 'OFF'}")

    print("[DONE]")

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Robust PDF Diff Tool (v2)")
    p.add_argument("pdf1", help="Base PDF")
    p.add_argument("pdf2", help="Modified PDF")
    p.add_argument("--mode", choices=["simple","tfidf","semantic","hybrid"], default="hybrid",
                   help="Matching engine")
    p.add_argument("--window", type=int, default=2, help="Pages per block")
    p.add_argument("--threshold", type=float, default=0.38, help="Similarity threshold")
    p.add_argument("--no-align", action="store_false", dest="align",
                   help="Disable global page alignment")
    p.add_argument("--min-sentence", type=int, default=2,
                   help="Min words for a sentence to be reported")
    p.add_argument("--output", default="diff_report", help="Prefix for output PDF")
    p.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_comparison(pdf1=args.pdf1,
                   pdf2=args.pdf2,
                   mode=args.mode,
                   window=args.window,
                   threshold=args.threshold,
                   align_pages=args.align,
                   min_sentence_words=args.min_sentence,
                   output_prefix=args.output,
                   semantic_model=args.model)