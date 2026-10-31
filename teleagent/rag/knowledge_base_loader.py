"""
Knowledge Base Loader for TeleAgent.

Loads an existing FAISS knowledge base from disk.
"""

from __future__ import annotations

import json
from pathlib import Path

from teleagent.rag.faiss_vector_store import FaissVectorStore


class KnowledgeBaseLoader:
    """
    Loads an already-built knowledge base.
    """

    def __init__(self):
        self.metadata = None

    def load(
        self,
        knowledge_base_directory: str,
    ) -> FaissVectorStore:

        path = Path(knowledge_base_directory)

        metadata_file = path / "info.json"

        if not metadata_file.exists():
            raise FileNotFoundError(metadata_file)

        with open(
            metadata_file,
            "r",
            encoding="utf-8",
        ) as f:

            self.metadata = json.load(f)

        

        return FaissVectorStore.load(
            str(path),
          
        )

    @property
    def embedding_dimension(self):

        return self.metadata["embedding_dimension"]

    @property
    def embedding_model(self):

        return self.metadata["embedding_model"]

    @property
    def document_count(self):

        return self.metadata["documents"]

    @property
    def chunk_count(self):

        return self.metadata["chunks"]
