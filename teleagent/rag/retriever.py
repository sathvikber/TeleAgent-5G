"""
Retriever for the TeleAgent RAG pipeline.
"""

from __future__ import annotations

from teleagent.rag.embedding_generator import EmbeddingGenerator
from teleagent.rag.faiss_vector_store import FaissVectorStore
from teleagent.rag.schemas import (
    Chunk,
    RetrievedContext,
)


class Retriever:
    """
    Retrieves the most relevant chunks for a user query.
    """

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_store: FaissVectorStore,
    ) -> None:

        self.embedding_generator = embedding_generator
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievedContext:
        """
        Retrieve top-k relevant chunks.
        """

        query_chunk = Chunk(
            id="query",
            document_id="query",
            content=query,
            metadata={},
        )

        embedding = self.embedding_generator.embed_chunk(
            query_chunk
        )

        results = self.vector_store.search(
            embedding.vector,
            top_k=top_k,
        )

        return RetrievedContext(
            query=query,
            results=results,
        )
