"""ChromaDB manager for document storage"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from pathlib import Path
from src.utils.groq_client import GroqEmbeddings
from src.utils.config import config
from src.utils.logging_utils import log_info, log_error


class ChromaDBManager:
    """Manager for ChromaDB operations"""

    def __init__(self, collection_name: str = "financial_documents"):
        self.collection_name = collection_name
        self.chromadb_path = config.chromadb_path

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.chromadb_path),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Initialize embedding function
        self.embedding_function = GroqEmbeddings()

        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name, embedding_function=self.embedding_function
            )
            log_info(f"Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={
                    "description": "Financial documents collection",
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                },
            )
            log_info(f"Created new collection: {self.collection_name}")

    def add_chunks(self, chunks: List[Dict]):
        """
        Add chunks to ChromaDB

        Args:
            chunks: List of chunk dictionaries
        """
        try:
            log_info(f"Adding {len(chunks)} chunks to ChromaDB")

            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []

            for chunk in chunks:
                ids.append(chunk["chunk_id"])
                documents.append(chunk["text"])

                # Prepare metadata (ChromaDB doesn't support nested dicts)
                metadata = {
                    "document_id": chunk["document_id"],
                    "content_type": chunk["content_type"],
                    "section": chunk.get("section", ""),
                    "word_count": chunk.get("word_count", 0),
                }

                # Add table data as JSON string if available
                if chunk.get("table_data"):
                    import json

                    metadata["table_data"] = json.dumps(chunk["table_data"])

                metadatas.append(metadata)

            # Add to collection in batches
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i : i + batch_size]
                batch_docs = documents[i : i + batch_size]
                batch_metas = metadatas[i : i + batch_size]

                self.collection.add(
                    ids=batch_ids, documents=batch_docs, metadatas=batch_metas
                )

            log_info(f"Successfully added {len(chunks)} chunks to ChromaDB")

        except Exception as e:
            log_error(e, context="ChromaDBManager.add_chunks")
            raise e

    def query(
        self, query_text: str, top_k: int = 10, filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Query ChromaDB for relevant chunks

        Args:
            query_text: Query string
            top_k: Number of results to return
            filters: Metadata filters

        Returns:
            List of relevant chunks with metadata
        """
        try:
            # Build where clause for filtering
            where_clause = None
            if filters:
                where_clause = filters

            # Query collection
            results = self.collection.query(
                query_texts=[query_text], n_results=top_k, where=where_clause
            )

            # Format results
            chunks = []
            if results["ids"] and len(results["ids"]) > 0:
                for i in range(len(results["ids"][0])):
                    chunk = {
                        "chunk_id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i]
                        if "distances" in results
                        else None,
                    }
                    chunks.append(chunk)

            log_info(f"Retrieved {len(chunks)} chunks for query: {query_text[:50]}...")

            return chunks

        except Exception as e:
            log_error(e, context="ChromaDBManager.query")
            return []

    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "count": count,
                "path": str(self.chromadb_path),
            }
        except Exception as e:
            log_error(e, context="ChromaDBManager.get_collection_stats")
            return {}

    def reset_collection(self):
        """Delete and recreate collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name, embedding_function=self.embedding_function
            )
            log_info(f"Reset collection: {self.collection_name}")
        except Exception as e:
            log_error(e, context="ChromaDBManager.reset_collection")
            raise e
