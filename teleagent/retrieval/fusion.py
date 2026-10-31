"""
Fusion

Combines the results from multiple retrievers using
Reciprocal Rank Fusion (RRF).
"""

from teleagent.rag.schemas import SearchResult


class Fusion:
    """
    Reciprocal Rank Fusion (RRF).

    Reference:
        Cormack et al.,
        "Reciprocal Rank Fusion Outperforms Condorcet
        and Individual Rank Learning Methods."
    """

    def __init__(
        self,
        k: int = 60,
    ) -> None:
        """
        Parameters
        ----------
        k : int
            RRF constant.
        """

        self._k = k

    def fuse(
        self,
        dense_results: list[SearchResult],
        sparse_results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Fuse dense and sparse retrieval results.

        Parameters
        ----------
        dense_results
            Results from DenseRetriever.

        sparse_results
            Results from SparseRetriever.

        top_k
            Number of final results.

        Returns
        -------
        list[SearchResult]
        """

        #
        # chunk_id -> SearchResult
        #

        chunk_map = {}

        #
        # chunk_id -> fused score
        #

        fused_scores = {}

        #
        # Dense ranking
        #

        for rank, result in enumerate(dense_results):

            chunk_id = result.chunk.id

            chunk_map[chunk_id] = result

            fused_scores.setdefault(chunk_id, 0.0)

            fused_scores[chunk_id] += 1.0 / (
                self._k + rank + 1
            )

        #
        # Sparse ranking
        #

        for rank, result in enumerate(sparse_results):

            chunk_id = result.chunk.id

            chunk_map[chunk_id] = result

            fused_scores.setdefault(chunk_id, 0.0)

            fused_scores[chunk_id] += 1.0 / (
                self._k + rank + 1
            )

        #
        # Sort by fused score
        #

        ranked = sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        fused_results = []

        for chunk_id, score in ranked[:top_k]:

            result = chunk_map[chunk_id]

            fused_results.append(
                SearchResult(
                    chunk=result.chunk,
                    score=score,
                )
            )

        return fused_results
