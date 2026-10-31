"""
telecom_chunker.py

Section-aware chunking for TeleAgent telecom documents.

The chunker receives structured TelecomDocumentSection objects
and creates smaller chunks while preserving telecom metadata.

Important
---------
- Chunks never span multiple logical document sections.
- Small sections remain intact.
- Large sections are split with configurable overlap.
- Metadata from the source section is preserved.
- Chunk IDs are deterministic and unique even when the same
  section number appears multiple times in a real 3GPP document.

This module does NOT:
- generate embeddings
- access FAISS
- perform retrieval
- modify KnowledgeAgent
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional

from teleagent.rag.telecom_ingestion import (
    TelecomDocumentSection,
)


@dataclass
class TelecomChunk:
    """
    Final structured chunk produced from a telecom document section.
    """

    chunk_id: str
    text: str

    chunk_index: int = 0

    source: Optional[str] = None

    document_type: Optional[str] = None
    specification_number: Optional[str] = None
    release: Optional[str] = None
    version: Optional[str] = None

    section_number: Optional[str] = None
    section_title: Optional[str] = None
    parent_section: Optional[str] = None

    network_functions: List[str] = field(
        default_factory=list
    )

    protocols: List[str] = field(
        default_factory=list
    )

    procedures: List[str] = field(
        default_factory=list
    )


class TelecomSectionChunker:
    """
    Split structured telecom sections into smaller chunks.

    Chunk size is currently measured using whitespace-separated
    words.

    Each section receives a deterministic occurrence index.
    This prevents duplicate chunk IDs when real telecom documents
    contain repeated section numbers, table-of-content entries,
    annex references, or other repeated headings.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 75,
    ) -> None:

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero"
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative"
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ============================================================
    # Public API
    # ============================================================

    def chunk_sections(
        self,
        sections: List[TelecomDocumentSection],
    ) -> List[TelecomChunk]:
        """
        Chunk a collection of structured telecom sections.

        Every section receives a section occurrence index.

        This guarantees that repeated section numbers such as:

            5.2.1
            5.2.1

        do not generate identical chunk IDs.
        """

        chunks: List[TelecomChunk] = []

        for section_index, section in enumerate(
            sections
        ):

            section_chunks = self.chunk_section(
                section=section,
                section_index=section_index,
            )

            chunks.extend(
                section_chunks
            )

        return chunks

    def chunk_section(
        self,
        section: TelecomDocumentSection,
        section_index: int = 0,
    ) -> List[TelecomChunk]:
        """
        Chunk one telecom section.

        A chunk never contains content from another logical section.

        Parameters
        ----------
        section:
            Structured telecom section.

        section_index:
            Occurrence index of the section within the source
            document.

            This value is used only to guarantee deterministic
            unique chunk IDs.
        """

        text = section.text.strip()

        if not text:
            return []

        words = self._split_words(
            text
        )

        # --------------------------------------------------------
        # Small logical section
        #
        # Keep the entire section as one chunk.
        # --------------------------------------------------------

        if len(words) <= self.chunk_size:

            return [
                self._create_chunk(
                    section=section,
                    text=text,
                    section_index=section_index,
                    chunk_index=0,
                )
            ]

        # --------------------------------------------------------
        # Large logical section
        #
        # Split only inside this section.
        # --------------------------------------------------------

        chunks: List[TelecomChunk] = []

        start = 0
        chunk_index = 0

        while start < len(words):

            end = min(
                start + self.chunk_size,
                len(words),
            )

            chunk_words = words[
                start:end
            ]

            chunk_text = " ".join(
                chunk_words
            ).strip()

            if chunk_text:

                chunks.append(
                    self._create_chunk(
                        section=section,
                        text=chunk_text,
                        section_index=section_index,
                        chunk_index=chunk_index,
                    )
                )

                chunk_index += 1

            # ----------------------------------------------------
            # Final chunk reached.
            # ----------------------------------------------------

            if end >= len(words):
                break

            # ----------------------------------------------------
            # Move forward while retaining overlap.
            # ----------------------------------------------------

            start = (
                end
                - self.chunk_overlap
            )

        return chunks

    # ============================================================
    # Chunk creation
    # ============================================================

    def _create_chunk(
        self,
        section: TelecomDocumentSection,
        text: str,
        section_index: int,
        chunk_index: int,
    ) -> TelecomChunk:
        """
        Create one TelecomChunk while preserving all
        section-level metadata.
        """

        chunk_id = self._build_chunk_id(
            section=section,
            section_index=section_index,
            chunk_index=chunk_index,
        )

        return TelecomChunk(
            chunk_id=chunk_id,
            text=text,
            chunk_index=chunk_index,

            source=section.source,

            document_type=(
                section.document_type
            ),

            specification_number=(
                section.specification_number
            ),

            release=section.release,
            version=section.version,

            section_number=(
                section.section_number
            ),

            section_title=(
                section.section_title
            ),

            parent_section=(
                section.parent_section
            ),

            network_functions=list(
                section.network_functions
            ),

            protocols=list(
                section.protocols
            ),

            procedures=list(
                section.procedures
            ),
        )

    # ============================================================
    # Text helpers
    # ============================================================

    @staticmethod
    def _split_words(
        text: str,
    ) -> List[str]:
        """
        Split text conservatively while preserving punctuation
        inside individual tokens.
        """

        return re.findall(
            r"\S+",
            text,
        )

    # ============================================================
    # Chunk ID generation
    # ============================================================

    @staticmethod
    def _build_chunk_id(
        section: TelecomDocumentSection,
        section_index: int,
        chunk_index: int,
    ) -> str:
        """
        Generate a deterministic and unique chunk identifier.

        The section occurrence index prevents duplicate IDs when
        the same section number appears several times in one
        document.

        Example
        -------

        ts_23_501_5_2_1_s0042_000

        Components:

        ts_23_501
            specification

        5_2_1
            section number

        s0042
            occurrence of the section inside the document

        000
            chunk number inside that section
        """

        # --------------------------------------------------------
        # Document component
        # --------------------------------------------------------

        if section.specification_number:

            document_part = (
                "ts_"
                + section.specification_number.replace(
                    ".",
                    "_",
                )
            )

        elif section.source:

            source_hash = hashlib.sha1(
                section.source.encode(
                    "utf-8"
                )
            ).hexdigest()[:8]

            document_part = (
                f"doc_{source_hash}"
            )

        else:

            text_hash = hashlib.sha1(
                section.text.encode(
                    "utf-8"
                )
            ).hexdigest()[:8]

            document_part = (
                f"doc_{text_hash}"
            )

        # --------------------------------------------------------
        # Section component
        # --------------------------------------------------------

        if section.section_number:

            section_part = (
                section.section_number.replace(
                    ".",
                    "_",
                )
            )

        else:

            section_part = "preamble"

        # --------------------------------------------------------
        # Unique deterministic ID
        # --------------------------------------------------------

        return (
            f"{document_part}_"
            f"{section_part}_"
            f"s{section_index:04d}_"
            f"{chunk_index:03d}"
        )
