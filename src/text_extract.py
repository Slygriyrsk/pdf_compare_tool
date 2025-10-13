import pymupdf4llm

def extractTxt(file_path: str) -> str:
    print(f"Logging: Extracting text from {file_path}")
    markdown_text = pymupdf4llm.to_markdown(file_path)
    return markdown_text