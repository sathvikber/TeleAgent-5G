"""
telecom_ingestion.py

Combines telecom section parsing and telecom metadata extraction
for the TeleAgent RAG ingestion pipeline.

This module does NOT:
- perform final chunking
- generate embeddings
- access FAISS
- perform retrieval
- modify KnowledgeAgent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from teleagent.rag.telecom_metadata import (
    TelecomMetadata,
    TelecomMetadataExtractor,
)
from teleagent.rag.telecom_parser import (
    TelecomSection,
    TelecomSectionParser,
)


@dataclass
class TelecomDocumentSection:
    """
    Represents one structured telecom document section together
    with document-level and section-level metadata.
    """

    text: str

    section_number: Optional[str] = None
    section_title: Optional[str] = None
    parent_section: Optional[str] = None

    source: Optional[str] = None

    document_type: Optional[str] = None
    specification_number: Optional[str] = None
    release: Optional[str] = None
    version: Optional[str] = None

    network_functions: List[str] = field(default_factory=list)
    protocols: List[str] = field(default_factory=list)
    procedures: List[str] = field(default_factory=list)


class TelecomIngestionProcessor:
    """
    Performs telecom-aware preprocessing of a document.

    Processing flow:

        Raw document
            |
            v
        Document-level metadata extraction
            |
            v
        Section parsing
            |
            v
        Section-level metadata extraction
            |
            v
        Structured telecom sections
    """

    def __init__(self) -> None:

        self._section_parser = TelecomSectionParser()
        self._metadata_extractor = TelecomMetadataExtractor()

    def process(
        self,
        text: str,
        source: Optional[str] = None,
    ) -> List[TelecomDocumentSection]:
        """
        Process one telecom document.

        Parameters
        ----------
        text:
            Complete document text.

        source:
            Optional filename or source identifier.

        Returns
        -------
        List[TelecomDocumentSection]
            Structured sections containing inherited document
            metadata and locally extracted telecom metadata.
        """

        if not text or not text.strip():
            return []

        # ---------------------------------------------------------
        # Extract document-wide metadata.
        #
        # Information such as:
        #   TS 23.502
        #   Release 18
        #   Version 18.3.0
        #
        # often occurs only on the first page / preamble.
        # ---------------------------------------------------------

        document_metadata = self._metadata_extractor.extract(
            text,
            source=source,
        )

        # ---------------------------------------------------------
        # Parse logical document sections.
        # ---------------------------------------------------------

        sections = self._section_parser.parse(text)

        structured_sections: List[TelecomDocumentSection] = []

        for section in sections:

            structured_section = self._process_section(
                section=section,
                document_metadata=document_metadata,
                source=source,
            )

            structured_sections.append(structured_section)

        return structured_sections

    def _process_section(
        self,
        section: TelecomSection,
        document_metadata: TelecomMetadata,
        source: Optional[str],
    ) -> TelecomDocumentSection:
        """
        Add section-local telecom metadata while inheriting
        document-level specification information.
        """

        # Include the heading during local metadata extraction.
        #
        # Example:
        #
        # 4.2.1 Authentication
        #
        # Even if the body does not repeat the word
        # "Authentication", the procedure should still be detected.

        section_text = section.full_text

        section_metadata = self._metadata_extractor.extract(
            section_text,
            source=source,
        )

        return TelecomDocumentSection(
            text=section_text,

            section_number=section.section_number,
            section_title=section.section_title,
            parent_section=section.parent_section,

            source=source,

            # Document-level metadata is inherited.
            document_type=document_metadata.document_type,
            specification_number=document_metadata.specification_number,
            release=document_metadata.release,
            version=document_metadata.version,

            # Section-local metadata is preferred here.
            network_functions=section_metadata.network_functions,
            protocols=section_metadata.protocols,
            procedures=section_metadata.procedures,
        )
