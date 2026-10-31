"""
FAISS vector database for TeleAgent RAG.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np

from teleagent.rag.schemas import (
    Chunk,
    Embedding,
    SearchResult,
)


class FaissVectorStore:
    """
    Simple FAISS-backed vector store.
    """

    def __init__(self, dimension: int):

        self.dimension = dimension

        self.index = faiss.IndexFlatIP(dimension)

        self._chunk_store: Dict[str, Chunk] = {}

        self.id_to_chunk: List[str] = []

    @property
    def chunks(self) -> list[Chunk]:
        """
        Return all stored chunks.
        """
        return list(self._chunk_store.values())

    @property
    def size(self) -> int:
        return self.index.ntotal

    def add(
        self,
        chunks: List[Chunk],
        embeddings: List[Embedding],
    ) -> None:

        if len(chunks) != len(embeddings):
            raise ValueError(
                "chunks and embeddings must have same length"
            )

        vectors = []

        for chunk, embedding in zip(chunks, embeddings):

            if chunk.id != embedding.chunk_id:
                raise ValueError(
                    "Chunk ID and Embedding ID do not match."
                )

            vectors.append(embedding.vector)

            self._chunk_store[chunk.id] = chunk

            self.id_to_chunk.append(chunk.id)

        matrix = np.asarray(
            vectors,
            dtype=np.float32,
        )

        self.index.add(matrix)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
    ) -> List[SearchResult]:

        if self.size == 0:
            return []

        query = np.asarray(
            [query_vector],
            dtype=np.float32,
        )

        scores, ids = self.index.search(query, top_k)

        results = []

        for score, idx in zip(scores[0], ids[0]):

            if idx == -1:
                continue

            chunk_id = self.id_to_chunk[idx]

            results.append(
                SearchResult(
                    chunk=self._chunk_store[chunk_id],
                    score=float(score),
                )
            )

        return results

    def save(
        self,
        directory: str,
    ) -> None:

        directory = Path(directory)

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(directory / "index.faiss"),
        )

        with open(
            directory / "metadata.pkl",
            "wb",
        ) as f:
            pickle.dump(
                (
                    self._chunk_store,
                    self.id_to_chunk,
                ),
                f,
            )

    @classmethod
    def load(
        cls,
        directory: str,
    ) -> "FaissVectorStore":

        directory = Path(directory)

        index = faiss.read_index(
            str(directory / "index.faiss")
        )

        with open(
            directory / "metadata.pkl",
            "rb",
        ) as f:
            chunk_store, id_to_chunk = pickle.load(f)

        store = cls(index.d)

        store.index = index
        store._chunk_store = chunk_store
        store.id_to_chunk = id_to_chunk

        return store
