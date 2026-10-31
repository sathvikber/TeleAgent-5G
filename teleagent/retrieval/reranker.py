"""
Re-ranker

Re-orders retrieved search results.

Current Implementation
----------------------
Score-based re-ranking.

Future Implementations
----------------------
- BAAI/bge-reranker-base
- BAAI/bge-reranker-large
- Cross Encoder
- ColBERT
"""

from teleagent.rag.schemas import SearchResult


class ReRanker:
    """
    Re-ranks retrieved search results.
    """

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Re-rank retrieved results.

        Parameters
        ----------
        query : str
            User query.

        results : list[SearchResult]
            Retrieved search results.

        top_k : int
            Number of results to return.

        Returns
        -------
        list[SearchResult]
        """

        #
        # Current implementation:
        # Sort by retrieval score.
        #

        ranked = sorted(
            results,
            key=lambda result: result.score,
            reverse=True,
        )

        return ranked[:top_k]
