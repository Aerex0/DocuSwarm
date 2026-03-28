"""Script to preprocess financial documents"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.task1_chunking.parsers.llamaparse_handler import LlamaParseHandler
from src.task1_chunking.parsers.pymupdf_parser import PyMuPDFParser
from src.task1_chunking.chunkers.multimodal_chunker import MultimodalChunker
from src.task1_chunking.storage.chromadb_manager import ChromaDBManager
from src.utils.logging_utils import log_info, log_error


def preprocess_document(pdf_path: str, use_llamaparse: bool = True):
    """
    Preprocess a single document

    Args:
        pdf_path: Path to PDF file
        use_llamaparse: Whether to use LlamaParse (True) or PyMuPDF (False)
    """
    try:
        # Step 1: Parse document
        if use_llamaparse:
            try:
                parser = LlamaParseHandler()
                parsed_data = parser.parse_document(pdf_path)
            except Exception as e:
                log_error(e, context="LlamaParse failed, falling back to PyMuPDF")
                parser = PyMuPDFParser()
                parsed_data = parser.parse_document(pdf_path)
        else:
            parser = PyMuPDFParser()
            parsed_data = parser.parse_document(pdf_path)

        # Step 2: Chunk document
        chunker = MultimodalChunker()
        chunks = chunker.chunk_document(parsed_data)

        # Step 3: Store in ChromaDB
        db_manager = ChromaDBManager()
        db_manager.add_chunks(chunks)

        log_info(f"✓ Successfully processed: {pdf_path}")
        log_info(f"  - Pages: {parsed_data['metadata']['pages']}")
        log_info(f"  - Tables: {len(parsed_data.get('tables', []))}")
        log_info(f"  - Chunks: {len(chunks)}")

        return True

    except Exception as e:
        log_error(e, context=f"preprocess_document({pdf_path})")
        return False


def preprocess_directory(directory: str, use_llamaparse: bool = True):
    """
    Preprocess all PDF files in a directory

    Args:
        directory: Path to directory containing PDFs
        use_llamaparse: Whether to use LlamaParse
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        print(f"Error: Directory not found: {directory}")
        return

    # Find all PDF files
    pdf_files = list(dir_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {directory}")
        return

    print(f"\nFound {len(pdf_files)} PDF files to process")
    print(f"Parser: {'LlamaParse' if use_llamaparse else 'PyMuPDF'}")
    print("-" * 60)

    successful = 0
    failed = 0

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        if preprocess_document(str(pdf_file), use_llamaparse):
            successful += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"Processing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(pdf_files)}")

    # Show collection stats
    db_manager = ChromaDBManager()
    stats = db_manager.get_collection_stats()
    print(f"\nChromaDB Collection Stats:")
    print(f"  Name: {stats['name']}")
    print(f"  Total chunks: {stats['count']}")
    print(f"  Path: {stats['path']}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess financial documents")
    parser.add_argument("--input", required=True, help="Input PDF file or directory")
    parser.add_argument(
        "--parser",
        default="llamaparse",
        choices=["llamaparse", "pymupdf"],
        help="Parser to use (default: llamaparse)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset ChromaDB collection before processing",
    )

    args = parser.parse_args()

    # Reset collection if requested
    if args.reset:
        print("Resetting ChromaDB collection...")
        db_manager = ChromaDBManager()
        db_manager.reset_collection()
        print("✓ Collection reset\n")

    use_llamaparse = args.parser == "llamaparse"
    input_path = Path(args.input)

    if input_path.is_file():
        # Process single file
        preprocess_document(str(input_path), use_llamaparse)
    elif input_path.is_dir():
        # Process directory
        preprocess_directory(str(input_path), use_llamaparse)
    else:
        print(f"Error: Invalid input path: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()
