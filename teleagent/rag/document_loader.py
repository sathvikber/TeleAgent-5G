"""
Document Loader

Loads telecom knowledge documents from disk.

Supported formats:
    - PDF (.pdf)
    - Microsoft Word (.docx)
    - Markdown (.md)
    - Plain Text (.txt)

DOCX handling:
    - Extracts paragraph text directly from WordprocessingML
    - Ignores Table-of-Contents paragraphs styled as TOC1 ... TOC9
    - Preserves paragraph boundaries for downstream telecom parsing

Directory handling:
    - Non-recursive by default
    - Prevents accidental ingestion of duplicate nested copies
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import zipfile
import xml.etree.ElementTree as ET

import pypdf

from .schemas import Document


class DocumentLoader:
    """
    Loads telecom documents from disk.
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".txt",
        ".md",
    }

    # ------------------------------------------------------------
    # DOCX XML namespace
    # ------------------------------------------------------------

    WORD_NAMESPACE = (
        "http://schemas.openxmlformats.org/"
        "wordprocessingml/2006/main"
    )

    WORD_NS = {
        "w": WORD_NAMESPACE
    }

    # ============================================================
    # Public API
    # ============================================================

    def load(
        self,
        path: str | Path,
    ) -> Document:
        """
        Load a single document.

        Parameters
        ----------
        path:
            Path to the document.

        Returns
        -------
        Document
            TeleAgent document object.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        if not path.is_file():

            raise ValueError(
                f"Not a file: {path}"
            )

        suffix = path.suffix.lower()

        if suffix not in self.SUPPORTED_EXTENSIONS:

            raise ValueError(
                f"Unsupported file type: {suffix}"
            )

        # --------------------------------------------------------
        # PDF
        # --------------------------------------------------------

        if suffix == ".pdf":

            text = self._load_pdf(
                path
            )

        # --------------------------------------------------------
        # DOCX
        # --------------------------------------------------------

        elif suffix == ".docx":

            text = self._load_docx(
                path
            )

        # --------------------------------------------------------
        # TXT / Markdown
        # --------------------------------------------------------

        else:

            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

        return Document(
            id=path.stem,
            source=path.name,
            content=text,
            metadata={
                "path": str(path),
                "extension": suffix,
            },
        )

    def load_directory(
        self,
        directory: str | Path,
        recursive: bool = False,
    ) -> List[Document]:
        """
        Load supported documents from a directory.

        Parameters
        ----------
        directory:
            Directory containing source documents.

        recursive:
            If False, only files directly inside the supplied
            directory are loaded.

            If True, supported documents from subdirectories
            are also loaded.

        Notes
        -----
        The default is False to prevent accidental duplicate
        ingestion from directories such as:

            raw_documents/
            raw_documents/raw_documents/
            raw_documents/docs/
        """

        directory = Path(
            directory
        )

        if not directory.exists():

            raise FileNotFoundError(
                directory
            )

        if not directory.is_dir():

            raise ValueError(
                f"Not a directory: {directory}"
            )

        documents: List[Document] = []

        if recursive:

            candidates = directory.rglob(
                "*"
            )

        else:

            candidates = directory.iterdir()

        for file in sorted(
            candidates
        ):

            if not file.is_file():
                continue

            if (
                file.suffix.lower()
                not in self.SUPPORTED_EXTENSIONS
            ):
                continue

            documents.append(
                self.load(
                    file
                )
            )

        return documents

    # ============================================================
    # PDF loading
    # ============================================================

    @staticmethod
    def _load_pdf(
        path: Path,
    ) -> str:
        """
        Extract text from a PDF document.
        """

        reader = pypdf.PdfReader(
            str(path)
        )

        pages: List[str] = []

        for page in reader.pages:

            text = page.extract_text()

            if text:

                text = text.strip()

                if text:

                    pages.append(
                        text
                    )

        return "\n".join(
            pages
        )

    # ============================================================
    # DOCX loading
    # ============================================================

    @classmethod
    def _load_docx(
        cls,
        path: Path,
    ) -> str:
        """
        Extract text from a DOCX file.

        Table-of-Contents paragraphs are ignored.

        Examples of ignored styles:

            TOC1
            TOC2
            TOC3
            ...
            TOC9

        This prevents duplicate section headings from entering
        the RAG corpus while preserving the actual document body.
        """

        try:

            with zipfile.ZipFile(
                path,
                "r",
            ) as archive:

                document_xml = archive.read(
                    "word/document.xml"
                )

        except KeyError as exc:

            raise ValueError(
                f"Invalid DOCX file: {path}"
            ) from exc

        except zipfile.BadZipFile as exc:

            raise ValueError(
                f"Corrupted DOCX file: {path}"
            ) from exc

        try:

            root = ET.fromstring(
                document_xml
            )

        except ET.ParseError as exc:

            raise ValueError(
                f"Unable to parse DOCX XML: {path}"
            ) from exc

        paragraphs: List[str] = []

        # --------------------------------------------------------
        # Iterate through paragraphs in document order.
        # --------------------------------------------------------

        for paragraph in root.findall(
            ".//w:p",
            cls.WORD_NS,
        ):

            # ----------------------------------------------------
            # Determine Word paragraph style.
            # ----------------------------------------------------

            style = (
                cls._get_docx_paragraph_style(
                    paragraph
                )
            )

            # ----------------------------------------------------
            # Skip Word-generated Table of Contents.
            #
            # Handles:
            #
            # TOC1
            # TOC2
            # ...
            # TOC9
            #
            # Case-insensitive for safety.
            # ----------------------------------------------------

            if cls._is_toc_style(
                style
            ):

                continue

            # ----------------------------------------------------
            # Extract paragraph text.
            # ----------------------------------------------------

            paragraph_text = (
                cls._extract_docx_paragraph_text(
                    paragraph
                )
            )

            if not paragraph_text:
                continue

            paragraphs.append(
                paragraph_text
            )

        return "\n".join(
            paragraphs
        )

    # ============================================================
    # DOCX helpers
    # ============================================================

    @classmethod
    def _get_docx_paragraph_style(
        cls,
        paragraph: ET.Element,
    ) -> Optional[str]:
        """
        Return the Word paragraph style ID.

        Examples
        --------
        TOC1
        TOC2
        Heading1
        FP
        ZT
        """

        paragraph_properties = (
            paragraph.find(
                "w:pPr",
                cls.WORD_NS,
            )
        )

        if paragraph_properties is None:

            return None

        style_node = (
            paragraph_properties.find(
                "w:pStyle",
                cls.WORD_NS,
            )
        )

        if style_node is None:

            return None

        style = style_node.get(
            (
                f"{{{cls.WORD_NAMESPACE}}}"
                "val"
            )
        )

        if style is None:
            return None

        return style.strip()

    @staticmethod
    def _is_toc_style(
        style: Optional[str],
    ) -> bool:
        """
        Return True when a paragraph uses a Word TOC style.

        Accepted examples:

            TOC1
            TOC2
            ...
            TOC9
        """

        if not style:
            return False

        normalized = (
            style.strip().upper()
        )

        if not normalized.startswith(
            "TOC"
        ):
            return False

        level = normalized[3:]

        if not level.isdigit():
            return False

        level_number = int(
            level
        )

        return 1 <= level_number <= 9

    @classmethod
    def _extract_docx_paragraph_text(
        cls,
        paragraph: ET.Element,
    ) -> str:
        """
        Extract visible text from one DOCX paragraph.

        Handles normal Word text nodes and preserves common
        tab/break elements reasonably for downstream parsing.
        """

        text_parts: List[str] = []

        for element in paragraph.iter():

            tag = element.tag

            # ----------------------------------------------------
            # Normal Word text
            # ----------------------------------------------------

            if tag == (
                f"{{{cls.WORD_NAMESPACE}}}t"
            ):

                if element.text:

                    text_parts.append(
                        element.text
                    )

            # ----------------------------------------------------
            # Tab
            # ----------------------------------------------------

            elif tag == (
                f"{{{cls.WORD_NAMESPACE}}}tab"
            ):

                text_parts.append(
                    " "
                )

            # ----------------------------------------------------
            # Line break
            # ----------------------------------------------------

            elif tag in {
                f"{{{cls.WORD_NAMESPACE}}}br",
                f"{{{cls.WORD_NAMESPACE}}}cr",
            }:

                text_parts.append(
                    "\n"
                )

        text = "".join(
            text_parts
        )

        # --------------------------------------------------------
        # Normalize excess internal whitespace while preserving
        # explicit line breaks.
        # --------------------------------------------------------

        lines = []

        for line in text.splitlines():

            normalized_line = " ".join(
                line.split()
            )

            if normalized_line:

                lines.append(
                    normalized_line
                )

        return "\n".join(
            lines
        ).strip()
