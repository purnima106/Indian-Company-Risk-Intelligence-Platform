from pathlib import Path
import fitz  # PyMuPDF-fitz is just the legacy name

def parse_pdf(pdf_path:str) -> list[dict]:
    """
        Extract text from every page of a PDF.
        Returns:
        [
        {
            "page_number": 1,
            "text": "Extracted text from page 1"
        },
        {
            "page_number": 2,
            "text": "Extracted text from page 2"
        },
        ]
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pages = []

    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):

            text = page.get_text("text")
            lines = []
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    line_text = "".join(span.get("text", "") for span in spans).strip()
                    if not line_text or not spans:
                        continue
                    lines.append(
                        {
                            "text": line_text,
                            "size": max(span.get("size", 0) for span in spans),
                            "is_heading": any(
                                span.get("size", 0) >= 11
                                and span.get("flags", 0) & 16
                                for span in spans
                            ),
                        }
                    )

            pages.append(
                {
                    "page": page_number,
                    "text": text.strip(),
                    "lines": lines,
                }
            )

    return pages

