"""PyMuPDF fallback parser"""

from typing import List, Dict
from pathlib import Path
import fitz  # PyMuPDF
from src.utils.logging_utils import log_info, log_error


class PyMuPDFParser:
    """Fallback PDF parser using PyMuPDF"""

    def parse_document(self, pdf_path: str) -> Dict[str, any]:
        """
        Parse PDF document with PyMuPDF

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dict containing parsed content and metadata
        """
        try:
            log_info(f"Parsing document with PyMuPDF (fallback): {pdf_path}")

            doc = fitz.open(pdf_path)

            parsed_data = {
                "document_id": Path(pdf_path).stem,
                "text": "",
                "sections": [],
                "tables": [],
                "figures": [],
                "metadata": {
                    "source": pdf_path,
                    "parser": "pymupdf",
                    "pages": len(doc),
                },
            }

            # Extract text from all pages
            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text("text")
                parsed_data["text"] += f"\n--- Page {page_num} ---\n"
                parsed_data["text"] += page_text

            doc.close()

            log_info(f"Successfully parsed {len(doc)} pages with PyMuPDF")

            return parsed_data

        except Exception as e:
            log_error(e, context=f"PyMuPDFParser.parse_document({pdf_path})")
            raise e
