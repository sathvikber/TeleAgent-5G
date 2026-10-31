"""
Loads a knowledge base and provides
semantic retrieval for the Knowledge Agent.
"""

from __future__ import annotations

from teleagent.rag.embedding_generator import EmbeddingGenerator
from teleagent.rag.knowledge_base_loader import KnowledgeBaseLoader
from teleagent.rag.retriever import Retriever
from teleagent.rag.schemas import RetrievedContext


class RAGService:
    """
    High-level interface to the RAG subsystem.
    """

    def __init__(
        self,
        knowledge_base_directory: str,
    ) -> None:

        loader = KnowledgeBaseLoader()

        #
        # Load vector database
        #

        self._vector_store = loader.load(
            knowledge_base_directory
        )

        self.metadata = loader.metadata

        #
        # Embedding model
        #

        self.embedding_generator = EmbeddingGenerator(
            model_name=self.metadata["embedding_model"]
        )

        #
        # Dense retriever
        #

        self._retriever = Retriever(
            self.embedding_generator,
            self._vector_store,
        )

    ####################################################################
    # Public read-only properties
    ####################################################################

    @property
    def vector_store(self):
        """
        Return the underlying FAISS vector store.
        """

        return self._vector_store

    @property
    def retriever(self):
        """
        Return the dense retriever.
        """

        return self._retriever

    ####################################################################
    # Retrieval API
    ####################################################################

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> RetrievedContext:
        """
        Retrieve relevant chunks for a query.
        """

        return self._retriever.retrieve(
            query=query,
            top_k=top_k,
        )

    def retrieve_text(
        self,
        query: str,
        top_k: int = 3,
    ) -> str:
        """
        Retrieve context as a single string.

        Note:
            This method is retained for backward compatibility.
            The IntelligentRetrievalService should be preferred
            for Hybrid RAG workflows.
        """

        context = self.retrieve(
            query=query,
            top_k=top_k,
        )

        if not context.results:
            return ""

        return "\n\n".join(
            result.chunk.content
            for result in context.results
        )

    ####################################################################
    # Metadata
    ####################################################################

    @property
    def stats(self) -> dict:
        """
        Knowledge base metadata.
        """

        return self.metadata
