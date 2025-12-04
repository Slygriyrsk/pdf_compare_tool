# # """
# # Compares text data from two PDFs.
# # Breaks text into paragraphs and compares each one.
# # """

# # from typing import List, Dict
# # from utils import normalize_value


# # class TextComparator:
# #     """Compares text content from two PDFs."""
    
# #     def compare(self, text1: str, text2: str, 
# #                 file1_name: str, file2_name: str) -> List[Dict]:
# #         """
# #         Compare text from two PDFs.
# #         Each paragraph from text1 is checked against all paragraphs in text2.
        
# #         Args:
# #             text1: Text from first PDF
# #             text2: Text from second PDF
# #             file1_name: Name of first PDF file
# #             file2_name: Name of second PDF file
            
# #         Returns:
# #             List of text segments from text1 not found in text2
# #         """
        
# #         differences = []
        
# #         # Split text into paragraphs (non-empty lines)
# #         paragraphs1 = self._extract_paragraphs(text1)
# #         paragraphs2 = self._extract_paragraphs(text2)
        
# #         # Check each paragraph from text1
# #         for para_idx, paragraph in enumerate(paragraphs1):
# #             if not self._paragraph_exists(paragraph, paragraphs2):
# #                 differences.append({
# #                     'file_from': file1_name,
# #                     'paragraph_index': para_idx,
# #                     'text': paragraph,
# #                     'status': 'not_found_in_other_pdf'
# #                 })
        
# #         return differences
    
# #     def _extract_paragraphs(self, text: str) -> List[str]:
# #         """
# #         Break text into meaningful paragraphs.
# #         Removes empty lines and page breaks.
        
# #         Args:
# #             text: Raw text content
            
# #         Returns:
# #             List of non-empty paragraphs
# #         """
        
# #         # Split by page breaks and double newlines
# #         parts = text.split("\n---PAGE_BREAK---\n")
# #         paragraphs = []
        
# #         for part in parts:
# #             # Split by double newlines (paragraph separator)
# #             paras = part.split("\n\n")
# #             for para in paras:
# #                 # Clean up and check non-empty
# #                 cleaned = para.strip()
# #                 if cleaned:
# #                     paragraphs.append(cleaned)
        
# #         return paragraphs
    
# #     def _paragraph_exists(self, paragraph: str, paragraphs_list: List[str]) -> bool:
# #         """
# #         Check if paragraph exists in list (normalized comparison).
        
# #         Args:
# #             paragraph: Paragraph to search for
# #             paragraphs_list: List of paragraphs to search in
            
# #         Returns:
# #             True if paragraph found, False otherwise
# #         """
        
# #         normalized_search = normalize_value(paragraph)
        
# #         for para in paragraphs_list:
# #             normalized_para = normalize_value(para)
# #             if normalized_search == normalized_para:
# #                 return True
        
# #         return False

# """
# Compare text paragraphs from two PDFs.
# Matches text across all paragraphs and identifies differences.
# """

# from normalizer import TextNormalizer


# class TextComparator:
#     """Compare text with paragraph-level matching across all paragraphs."""
    
#     def __init__(self):
#         self.normalizer = TextNormalizer()
#         self.differences = []
    
#     def compare(self, text_pdf1, text_pdf2, file_name_1, file_name_2):
#         """
#         Compare text from two PDFs.
#         Each paragraph from PDF1 is checked against ALL paragraphs in PDF2.
        
#         Args:
#             text_pdf1: List of text dicts from PDF1
#             text_pdf2: List of text dicts from PDF2
#             file_name_1: Name of first PDF
#             file_name_2: Name of second PDF
        
#         Returns:
#             List of differences found
#         """
#         self.differences = []
        
#         if not text_pdf1 and not text_pdf2:
#             return []
        
#         for para_idx_1, para_obj_1 in enumerate(text_pdf1):
#             para_text_1 = para_obj_1.get("text", "")
            
#             # Normalize paragraph text
#             norm_para_1 = self.normalizer.normalize(para_text_1)
            
#             # Check if this paragraph exists in ANY paragraph of PDF2
#             paragraph_found = False
            
#             for para_obj_2 in text_pdf2:
#                 para_text_2 = para_obj_2.get("text", "")
#                 norm_para_2 = self.normalizer.normalize(para_text_2)
                
#                 if norm_para_1 == norm_para_2:
#                     paragraph_found = True
#                     break
            
#             # If paragraph not found in PDF2, add to differences
#             if not paragraph_found:
#                 self.differences.append({
#                     "source": f"Text (Page {para_obj_1['page']}, Paragraph {para_idx_1 + 1})",
#                     "from_file": file_name_1,
#                     "comparison_file": file_name_2,
#                     "text": para_text_1,
#                     "reason": "Text not found in second PDF"
#                 })
        
#         return self.differences

from normalizer import normalize_text
import re

class TextComparator:
    """
    Compares text content from two PDFs.
    Logic: Extract sentences (up to full stop), number them,
    and identify sentences NOT present in PDF2.
    """
    
    def __init__(self):
        self.pdf1_text = ""
        self.pdf2_text = ""
    
    def load_text(self, pdf1_text, pdf2_text):
        """Store extracted text from both PDFs"""
        self.pdf1_text = pdf1_text
        self.pdf2_text = pdf2_text
    
    def extract_sentences(self, text):
        """
        Extract sentences from text (split by full stop).
        Returns list of cleaned sentences with sentence numbers.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        numbered_sentences = []
        counter = 1
        
        for sentence in sentences:
            cleaned = sentence.strip()
            if cleaned:  # Only add non-empty sentences
                numbered_sentences.append({
                    'number': counter,
                    'text': cleaned,
                    'normalized': normalize_text(cleaned)
                })
                counter += 1
        
        return numbered_sentences
    
    def sentence_exists_in_pdf2(self, sentence_normalized):
        """
        Check if a sentence exists ANYWHERE in PDF2 text.
        Word-by-word comparison for robustness.
        """
        pdf2_normalized = normalize_text(self.pdf2_text)
        return sentence_normalized in pdf2_normalized
    
    def compare(self):
        """
        Compare all sentences from PDF1 against PDF2.
        Returns: List of sentences NOT found in PDF2
        """
        pdf1_sentences = self.extract_sentences(self.pdf1_text)
        different_sentences = []
        
        for sentence_obj in pdf1_sentences:
            # Check if this sentence exists in PDF2
            if not self.sentence_exists_in_pdf2(sentence_obj['normalized']):
                different_sentences.append({
                    'sentence_number': sentence_obj['number'],
                    'text': sentence_obj['text'],
                    'status': 'Not found in PDF2'
                })
        
        return different_sentences