"""
Schemas used by the RAG subsystem.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class Document(BaseModel):
    """
    Represents a source document loaded into the knowledge base.
    """

    id: str = Field(..., description="Unique document ID")

    source: str = Field(..., description="Source filename or path")

    content: str = Field(..., description="Complete document text")

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class Chunk(BaseModel):
    """
    Represents a chunk generated from a document.
    """

    id: str = Field(..., description="Unique chunk ID")

    document_id: str = Field(
        ...,
        description="Parent document ID"
    )

    content: str = Field(
        ...,
        description="Chunk text"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Inherited metadata"
    )


class Embedding(BaseModel):
    """
    Embedding vector for a chunk.
    """

    chunk_id: str

    vector: List[float]


class SearchResult(BaseModel):
    """
    Result returned from vector search.
    """

    chunk: Chunk

    score: float


class RetrievedContext(BaseModel):
    """
    Collection of retrieved chunks.
    """

    query: str

    results: List[SearchResult]


class KnowledgeBaseStats(BaseModel):
    """
    Statistics of the knowledge base.
    """

    documents: int

    chunks: int

    embedding_dimension: int


__all__ = [
    "Document",
    "Chunk",
    "Embedding",
    "SearchResult",
    "RetrievedContext",
    "KnowledgeBaseStats",
]
