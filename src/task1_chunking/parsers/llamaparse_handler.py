"""LlamaParse document parser"""

from typing import List, Dict, Optional
from pathlib import Path
from llama_parse import LlamaParse
from src.utils.config import config
from src.utils.logging_utils import log_info, log_error


class LlamaParseHandler:
    """Handler for LlamaParse API"""

    def __init__(self):
        self.api_key = config.llamaparse_api_key
        if not self.api_key:
            raise ValueError("LLAMAPARSE_API_KEY not found in environment variables")

        # Get config from llamaparse.yaml
        parse_config = config.llamaparse_config.get("parser", {})

        self.parser = LlamaParse(
            api_key=self.api_key,
            result_type=parse_config.get("result_type", "markdown"),
            use_vendor_multimodal_model=parse_config.get("features", {}).get(
                "use_vendor_multimodal_model", True
            ),
            vendor_multimodal_model_name=parse_config.get("features", {}).get(
                "vendor_multimodal_model_name", "anthropic-sonnet-3.5"
            ),
            parsing_instruction=parse_config.get(
                "parsing_instruction",
                "Extract all text, tables, and figures with structure preserved.",
            ),
            max_timeout=parse_config.get("timeouts", {}).get("max_timeout", 300),
        )

    def parse_document(self, pdf_path: str) -> Dict[str, any]:
        """
        Parse PDF document with LlamaParse

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dict containing parsed content, tables, and metadata
        """
        try:
            log_info(f"Parsing document with LlamaParse: {pdf_path}")

            # Parse document
            documents = self.parser.load_data(pdf_path)

            # Extract content
            parsed_data = {
                "document_id": Path(pdf_path).stem,
                "text": "",
                "sections": [],
                "tables": [],
                "figures": [],
                "metadata": {
                    "source": pdf_path,
                    "parser": "llamaparse",
                    "pages": len(documents),
                },
            }

            # Combine all document text
            for doc in documents:
                parsed_data["text"] += doc.text + "\n\n"

            # Extract tables and figures from markdown
            parsed_data["tables"] = self._extract_tables(parsed_data["text"])
            parsed_data["figures"] = self._extract_figures(parsed_data["text"])

            log_info(
                f"Successfully parsed {len(documents)} pages, {len(parsed_data['tables'])} tables, {len(parsed_data['figures'])} figures"
            )

            return parsed_data

        except Exception as e:
            log_error(e, context=f"LlamaParse.parse_document({pdf_path})")
            raise e

    def _extract_tables(self, markdown_text: str) -> List[Dict]:
        """Extract tables from markdown text"""
        tables = []
        lines = markdown_text.split("\n")

        in_table = False
        current_table = []
        table_id = 0

        for line in lines:
            # Detect table start (line with pipes |)
            if "|" in line and not in_table:
                in_table = True
                current_table = [line]
            elif "|" in line and in_table:
                current_table.append(line)
            elif in_table and "|" not in line:
                # Table ended
                if len(current_table) > 2:  # At least header + separator + 1 row
                    table_markdown = "\n".join(current_table)
                    tables.append(
                        {
                            "table_id": f"table_{table_id}",
                            "markdown": table_markdown,
                            "data": self._parse_markdown_table(current_table),
                        }
                    )
                    table_id += 1
                in_table = False
                current_table = []

        return tables

    def _parse_markdown_table(self, table_lines: List[str]) -> Dict:
        """Parse markdown table into structured data"""
        if len(table_lines) < 2:
            return {}

        # Parse header
        header_line = table_lines[0].strip("|").split("|")
        headers = [h.strip() for h in header_line]

        # Parse rows (skip separator line)
        rows = []
        for line in table_lines[2:]:
            row_data = line.strip("|").split("|")
            row_values = [v.strip() for v in row_data]
            if row_values:
                rows.append(row_values)

        return {"headers": headers, "rows": rows}

    def _extract_figures(self, markdown_text: str) -> List[Dict]:
        """Extract figure references from markdown"""
        figures = []
        lines = markdown_text.split("\n")
        figure_id = 0

        for line in lines:
            # Look for figure patterns: ![caption](url) or "Figure X:"
            if "![" in line or "Figure" in line:
                figures.append(
                    {
                        "figure_id": f"figure_{figure_id}",
                        "caption": line.strip(),
                        "context": "",  # To be filled with surrounding text
                    }
                )
                figure_id += 1

        return figures
