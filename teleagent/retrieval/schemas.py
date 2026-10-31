"""
Schemas for Intelligent Retrieval Layer.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalRequest:
    """
    Request sent to the Intelligent Retrieval Service.
    """
    query: str
    top_k: int = 5


@dataclass
class RetrievedChunk:
    """
    A single retrieved chunk.
    """
    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResponse:
    """
    Response returned by the Intelligent Retrieval Service.
    """
    query: str
    chunks: list[RetrievedChunk]
