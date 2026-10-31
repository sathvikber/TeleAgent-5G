"""
Text chunking for RAG.

Converts loaded documents into overlapping text chunks that
can later be embedded and stored in the vector database.

Supports two ingestion paths:

1. Generic documents
   -> existing character-based overlapping chunking

2. Telecom / 3GPP documents
   -> section-aware telecom ingestion
   -> telecom-aware chunking
   -> existing Chunk schema

The public DataChunker interface remains unchanged.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from teleagent.rag.schemas import Document, Chunk
from teleagent.rag.telecom_ingestion import TelecomIngestionProcessor
from teleagent.rag.telecom_chunker import TelecomSectionChunker


class DataChunker:
    """
    Splits documents into overlapping chunks.

    Parameters
    ----------
    chunk_size:
        Maximum characters per chunk for generic documents.

    chunk_overlap:
        Number of overlapping characters between consecutive
        generic chunks.

    telecom_aware:
        Enable telecom-aware ingestion when a document is
        recognized as a telecom / 3GPP document.

    Notes
    -----
    Generic document chunking preserves the original TeleAgent
    behavior.

    Telecom-aware chunking is used only when the document can
    be conservatively identified as telecom-specific.
    """

    # Conservative indicators used only to decide whether the
    # telecom-aware pipeline should be invoked.
    #
    # We intentionally do not classify a document as telecom
    # simply because it contains words such as "network".
    TELECOM_DOCUMENT_PATTERNS = (
        re.compile(
            r"\b(?:3GPP\s+)?(?:TS|TR)\s*\d{2}\.\d{3}\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b5G\s+(?:Core|Network|System)\b",
            re.IGNORECASE,
        ),
    )

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        telecom_aware: bool = True,
    ) -> None:

        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.telecom_aware = telecom_aware

        # Telecom ingestion components are initialized only once.
        self._telecom_processor = TelecomIngestionProcessor()

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Split a single document into chunks.

        Telecom documents are processed using section-aware
        ingestion.

        Other documents use the original TeleAgent generic
        chunking behavior.
        """

        text = document.content.strip()

        if not text:
            return []

        if (
            self.telecom_aware
            and self._is_telecom_document(document)
        ):
            return self._chunk_telecom_document(document)

        return self._chunk_generic_document(document)

    def _chunk_generic_document(
        self,
        document: Document,
    ) -> List[Chunk]:
        """
        Original TeleAgent character-based chunking behavior.

        This method intentionally preserves the previous
        implementation.
        """

        text = document.content.strip()

        if not text:
            return []

        chunks: List[Chunk] = []

        start = 0
        chunk_id = 0

        while start < len(text):

            end = min(
                start + self.chunk_size,
                len(text),
            )

            chunk_text = text[start:end].strip()

            chunks.append(
                Chunk(
                    id=f"{document.id}_chunk_{chunk_id}",
                    document_id=document.id,
                    content=chunk_text,
                    metadata=dict(document.metadata),
                )
            )

            chunk_id += 1

            if end == len(text):
                break

            start = end - self.chunk_overlap

        return chunks

    def _chunk_telecom_document(
        self,
        document: Document,
    ) -> List[Chunk]:
        """
        Telecom-aware document chunking.

        Processing flow:

        Document
            ↓
        TelecomIngestionProcessor
            ↓
        logical telecom sections
            ↓
        TelecomSectionChunker
            ↓
        existing TeleAgent Chunk schema
        """

        sections = self._telecom_processor.process(
            document.content,
            source=document.source,
        )

        if not sections:
            return []

        # TelecomSectionChunker currently uses word-based sizes.
        #
        # Convert the existing character-oriented DataChunker
        # configuration to conservative word-based values.
        #
        # Approximation:
        #     ~5 characters per word including spacing.
        #
        # This preserves the existing DataChunker configuration
        # interface without introducing another external setting.
        telecom_chunk_size = max(
            1,
            self.chunk_size // 5,
        )

        telecom_chunk_overlap = max(
            0,
            self.chunk_overlap // 5,
        )

        # Ensure overlap is always smaller than chunk size.
        if telecom_chunk_overlap >= telecom_chunk_size:
            telecom_chunk_overlap = max(
                0,
                telecom_chunk_size - 1,
            )

        telecom_chunker = TelecomSectionChunker(
            chunk_size=telecom_chunk_size,
            chunk_overlap=telecom_chunk_overlap,
        )

        telecom_chunks = telecom_chunker.chunk_sections(
            sections
        )

        chunks: List[Chunk] = []

        for telecom_chunk in telecom_chunks:

            metadata = self._build_telecom_metadata(
                document=document,
                telecom_chunk=telecom_chunk,
            )

            chunks.append(
                Chunk(
                    id=(
                        f"{document.id}_"
                        f"{telecom_chunk.chunk_id}"
                    ),
                    document_id=document.id,
                    content=telecom_chunk.text,
                    metadata=metadata,
                )
            )

        return chunks

    @staticmethod
    def _build_telecom_metadata(
        document: Document,
        telecom_chunk: Any,
    ) -> Dict[str, Any]:
        """
        Merge original document metadata with telecom-specific
        chunk metadata.

        Existing document metadata is preserved.

        Telecom metadata is then added to the same dictionary so
        downstream components such as EmbeddingGenerator and
        FAISSVectorStore can continue using the existing Chunk
        schema.
        """

        metadata: Dict[str, Any] = dict(
            document.metadata
        )

        metadata.update(
            {
                "telecom_aware": True,
                "source": telecom_chunk.source,
                "document_type": telecom_chunk.document_type,
                "specification_number":
                    telecom_chunk.specification_number,
                "release": telecom_chunk.release,
                "version": telecom_chunk.version,
                "section_number":
                    telecom_chunk.section_number,
                "section_title":
                    telecom_chunk.section_title,
                "parent_section":
                    telecom_chunk.parent_section,
                "chunk_index":
                    telecom_chunk.chunk_index,
                "network_functions":
                    list(telecom_chunk.network_functions),
                "protocols":
                    list(telecom_chunk.protocols),
                "procedures":
                    list(telecom_chunk.procedures),
            }
        )

        # Remove metadata entries whose value is None.
        #
        # Empty lists are retained intentionally because they
        # indicate that extraction was performed but nothing
        # was detected.
        return {
            key: value
            for key, value in metadata.items()
            if value is not None
        }

    def _is_telecom_document(
        self,
        document: Document,
    ) -> bool:
        """
        Conservatively determine whether telecom-aware ingestion
        should be used.

        Strong 3GPP document identifiers take priority.

        Existing document metadata may also explicitly request
        telecom-aware ingestion.
        """

        # ---------------------------------------------------------
        # Explicit metadata override
        # ---------------------------------------------------------

        if document.metadata.get("telecom_aware") is True:
            return True

        # ---------------------------------------------------------
        # Existing document type metadata
        # ---------------------------------------------------------

        document_type = str(
            document.metadata.get(
                "document_type",
                "",
            )
        ).upper()

        if document_type in {
            "3GPP_TS",
            "3GPP_TR",
            "TELECOM",
        }:
            return True

        # ---------------------------------------------------------
        # Search document text and source name.
        # ---------------------------------------------------------

        searchable_text = (
            f"{document.source}\n"
            f"{document.content[:5000]}"
        )

        for pattern in self.TELECOM_DOCUMENT_PATTERNS:

            if pattern.search(searchable_text):
                return True

        return False

    def chunk_documents(
        self,
        documents: List[Document],
    ) -> List[Chunk]:
        """
        Chunk multiple documents.
        """

        chunks: List[Chunk] = []

        for document in documents:
            chunks.extend(
                self.chunk_document(document)
            )

        return chunks
