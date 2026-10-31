"""
Sparse Retriever

Performs lexical retrieval using BM25.
"""

from rank_bm25 import BM25Okapi

from teleagent.rag.faiss_vector_store import FaissVectorStore
from teleagent.rag.schemas import (
    Chunk,
    SearchResult,
)


class SparseRetriever:
    """
    BM25-based sparse retriever.
    """

    def __init__(
        self,
        vector_store: FaissVectorStore,
    ) -> None:

        self._vector_store = vector_store

        #
        # Build corpus
        #

        self._chunks= vector_store.chunks

        tokenized_corpus = []

        for chunk in self._chunks:

            tokenized_corpus.append(
                chunk.content.lower().split()
            )

        self._bm25 = BM25Okapi(
            tokenized_corpus
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Perform sparse retrieval.

        Parameters
        ----------
        query : str

        top_k : int

        Returns
        -------
        list[SearchResult]
        """

        tokens = query.lower().split()

        scores = self._bm25.get_scores(
            tokens
        )

        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )

        results = []

        for index, score in ranked[:top_k]:

            results.append(
                SearchResult(
                    chunk=self._chunks[index],
                    score=float(score),
                )
            )

        return results
