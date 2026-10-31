"""
Dense Retriever

Performs dense semantic retrieval using the existing
Retriever implementation from the RAG module.
"""

from teleagent.rag.retriever import Retriever
from teleagent.rag.schemas import SearchResult


class DenseRetriever:
    """
    Dense semantic retriever.

    This class wraps the existing Retriever implementation
    without exposing the rest of the RAG subsystem.
    """

    def __init__(
        self,
        retriever: Retriever,
    ) -> None:

        self._retriever = retriever

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Perform dense retrieval.

        """
        context= self._retriever.retrieve(
            query=query,
            top_k=top_k,
        )
        return context.results
