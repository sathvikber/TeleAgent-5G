"""
Knowledge Base Builder for TeleAgent RAG.

Offline utility that builds a FAISS vector database
from telecom documents.
"""

from __future__ import annotations

import json
from pathlib import Path

from teleagent.rag.chunker import DataChunker
from teleagent.rag.document_loader import DocumentLoader
from teleagent.rag.embedding_generator import EmbeddingGenerator
from teleagent.rag.faiss_vector_store import FaissVectorStore


class KnowledgeBaseBuilder:

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ) -> None:

        self.loader = DocumentLoader()
        self.chunker = DataChunker()
        self.generator = EmbeddingGenerator(
            model_name=embedding_model
        )

    def build(
        self,
        input_directory: str,
        output_directory: str,
    ) -> FaissVectorStore:
        """
        Build a complete knowledge base.
        """

        # Load documents
        documents = self.loader.load_directory(
            input_directory
        )

        # Chunk documents
        chunks = self.chunker.chunk_documents(
            documents
        )

        # Generate embeddings
        embeddings = self.generator.embed_chunks(
            chunks
        )

        # Build FAISS index
        store = FaissVectorStore(
            self.generator.embedding_dimension
        )

        store.add(
            chunks,
            embeddings,
        )

        # Save vector store
        store.save(output_directory)

        # Save metadata
        self._write_metadata(
            output_directory,
            len(documents),
            len(chunks),
        )

        return store

    def _write_metadata(
        self,
        output_directory: str,
        document_count: int,
        chunk_count: int,
    ) -> None:

        metadata = {
            "embedding_model":
                self.generator.model_name,

            "embedding_dimension":
                self.generator.embedding_dimension,

            "documents":
                document_count,

            "chunks":
                chunk_count,

            "vector_store":
                "FAISS",

            "version":
                "1.0",
        }

        output_path = Path(output_directory)
        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path / "info.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4,
            )
if __name__ == "__main__":

    input_directory = (
        "/workspace/teleagent/data/raw_documents"
    )

    output_directory = (
        "/workspace/teleagent/data/knowledge_base"
    )

    print("=" * 70)
    print("TeleAgent Knowledge Base Builder")
    print("=" * 70)

    print("Input directory :", input_directory)
    print("Output directory:", output_directory)
    print()

    builder = KnowledgeBaseBuilder()

    store = builder.build(
        input_directory=input_directory,
        output_directory=output_directory,
    )

    print()
    print("=" * 70)
    print("KNOWLEDGE BASE BUILD COMPLETED")
    print("Stored chunks:", store.size)
    print("=" * 70)
