"""
Embedding generation for the RAG pipeline.
"""

from __future__ import annotations

from typing import List

from sentence_transformers import SentenceTransformer

from teleagent.rag.schemas import Chunk, Embedding


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class EmbeddingGenerator:
    """
    Generates dense embeddings for document chunks.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
    ) -> None:

        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

    @property
    def embedding_dimension(self) -> int:
        """
        Returns the embedding size.
        """
        return self.model.get_embedding_dimension()

    def embed_chunk(
        self,
        chunk: Chunk,
    ) -> Embedding:
        """
        Generate embedding for one chunk.
        """

        vector = self.model.encode(
            chunk.content,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return Embedding(
            chunk_id=chunk.id,
            vector=vector.tolist(),
        )

    def embed_chunks(
        self,
        chunks: List[Chunk],
    ) -> List[Embedding]:
        """
        Generate embeddings for multiple chunks.
        """

        if not chunks:
            return []

        texts = [c.content for c in chunks]

        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = []

        for chunk, vector in zip(chunks, vectors):

            embeddings.append(
                Embedding(
                    chunk_id=chunk.id,
                    vector=vector.tolist(),
                )
            )

        return embeddings
