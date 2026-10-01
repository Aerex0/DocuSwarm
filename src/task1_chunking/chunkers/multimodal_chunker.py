"""Multimodal chunker for financial documents"""

from typing import List, Dict
import re
import uuid
from src.utils.logging_utils import log_info
from src.utils.config import config


class MultimodalChunker:
    """Chunk documents preserving multimodal content"""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or config.default_chunk_size
        self.chunk_overlap = chunk_overlap or config.default_chunk_overlap

    def chunk_document(self, parsed_data: Dict) -> List[Dict]:
        """
        Chunk document with multimodal support

        Args:
            parsed_data: Parsed document data from parser

        Returns:
            List of chunks with metadata
        """
        log_info(f"Chunking document: {parsed_data['document_id']}")

        chunks = []
        document_id = parsed_data["document_id"]
        text = parsed_data["text"]
        tables = parsed_data.get("tables", [])

        # 1. Create text chunks
        text_chunks = self._chunk_text(text, document_id)
        chunks.extend(text_chunks)

        # 2. Create table chunks
        table_chunks = self._chunk_tables(tables, document_id)
        chunks.extend(table_chunks)

        log_info(
            f"Created {len(chunks)} chunks ({len(text_chunks)} text, {len(table_chunks)} table)"
        )

        return chunks

    def _document_metadata(self, document_id: str) -> Dict[str, str]:
        """Derive citation metadata from a document id (e.g. AMAZON_2022_10K)."""
        year_match = re.search(r"(19|20)\d{2}", document_id)
        return {
            "document_name": document_id,
            "year": year_match.group(0) if year_match else "unknown",
        }

    def _chunk_text(self, text: str, document_id: str) -> List[Dict]:
        """Chunk text with overlap"""
        chunks = []

        # Split into sentences (simple approach)
        sentences = re.split(r"(?<=[.!?])\s+", text)

        current_chunk = []
        current_size = 0
        chunk_num = 0

        for sentence in sentences:
            sentence_tokens = len(sentence.split())

            if current_size + sentence_tokens > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    {
                        "chunk_id": f"{document_id}_text_{chunk_num}",
                        "document_id": document_id,
                        "content_type": "text",
                        "text": chunk_text,
                        "section": "main",
                        "word_count": len(chunk_text.split()),
                        **self._document_metadata(document_id),
                    }
                )
                chunk_num += 1

                # Keep overlap
                overlap_sentences = []
                overlap_size = 0
                for sent in reversed(current_chunk):
                    sent_tokens = len(sent.split())
                    if overlap_size + sent_tokens < self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_size += sent_tokens
                    else:
                        break

                current_chunk = overlap_sentences
                current_size = overlap_size

            current_chunk.append(sentence)
            current_size += sentence_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                {
                    "chunk_id": f"{document_id}_text_{chunk_num}",
                    "document_id": document_id,
                    "content_type": "text",
                    "text": chunk_text,
                    "section": "main",
                    "word_count": len(chunk_text.split()),
                    **self._document_metadata(document_id),
                }
            )

        return chunks

    def _chunk_tables(self, tables: List[Dict], document_id: str) -> List[Dict]:
        """Create chunks for tables"""
        chunks = []

        for i, table in enumerate(tables):
            # Create table description
            table_markdown = table.get("markdown", "")
            table_data = table.get("data", {})

            # Generate natural language description
            description = self._generate_table_description(table_data)

            # Combine markdown and description for better retrieval
            chunk_text = f"TABLE:\n{table_markdown}\n\nDESCRIPTION: {description}"

            chunks.append(
                {
                    "chunk_id": f"{document_id}_table_{i}",
                    "document_id": document_id,
                    "content_type": "table",
                    "text": chunk_text,
                    "table_markdown": table_markdown,
                    "table_data": table_data,
                    "section": "table",
                    "word_count": len(chunk_text.split()),
                    **self._document_metadata(document_id),
                }
            )

        return chunks

    def _generate_table_description(self, table_data: Dict) -> str:
        """Generate natural language description of table"""
        if not table_data:
            return "Table with data"

        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])

        description = (
            f"Table with {len(rows)} rows and columns: {', '.join(headers[:5])}"
        )

        # Add sample values if available
        if rows and len(rows) > 0:
            first_row = rows[0]
            description += f". Sample data: {', '.join(str(v) for v in first_row[:3])}"

        return description
