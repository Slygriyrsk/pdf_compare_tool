import os
from text_extract import extractTxt as extract_text_from_pdf
from display_diff import cmptxt as compare_texts

def main():
    old_pdf = "inPDF/old.pdf"
    new_pdf = "inPDF/new.pdf"
    output_html = "outPDF/diff_report.html"

    os.makedirs("output_pdfs", exist_ok=True)

    print("Starting comparing things below...")

    # Extract text from both PDFs
    old_text = extract_text_from_pdf(old_pdf)
    new_text = extract_text_from_pdf(new_pdf)

    # Compare and create diff
    print("LOG: Differences...")
    diff_html = compare_texts(old_text, new_text)

    # saving the difference report file for reference
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(diff_html)

    print(f"Difference report file saved at {output_html}")

if __name__ == "__main__":
    main()