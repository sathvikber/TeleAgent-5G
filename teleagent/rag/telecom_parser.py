"""
telecom_parser.py

Telecom-aware section parser for TeleAgent RAG.

Responsibilities
----------------
- Detect hierarchical sections in telecom documents
- Preserve section number and title
- Determine parent-section relationships
- Preserve text before the first detected section
- Reject common false headings found in telecom specifications

Examples of supported headings
------------------------------
1 Scope
4 Architecture
4.2 Registration Procedures
4.2.1 Initial Registration
5.3.2.1 Authentication Procedure

Examples intentionally rejected
-------------------------------
650 Route des Lucioles
100 ms
2000 ms
200 OK
201 Created
404 Not Found
23.501 System Architecture
29.518 Some change-history entry

This module performs parsing only.

It does NOT:
- perform chunking
- generate embeddings
- access FAISS
- perform retrieval
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TelecomSection:
    """
    Represents one logical section extracted from a telecom document.
    """

    section_number: Optional[str]
    section_title: Optional[str]
    text: str
    parent_section: Optional[str] = None

    @property
    def full_text(self) -> str:
        """
        Return section heading followed by section body.
        """

        if self.section_number and self.section_title:

            heading = (
                f"{self.section_number} "
                f"{self.section_title}"
            )

            if self.text.strip():

                return (
                    f"{heading}\n"
                    f"{self.text.strip()}"
                )

            return heading

        return self.text.strip()


class TelecomSectionParser:
    """
    Detect hierarchical numbered sections in telecom documents.

    The parser is deterministic and lightweight.

    It deliberately rejects several numeric patterns that frequently
    occur in real 3GPP specifications but are not section headings,
    such as:

    - HTTP status codes
    - timer values
    - postal addresses
    - 3GPP specification numbers
    - change-history references
    """

    # ============================================================
    # Candidate section heading
    # ============================================================

    #
    # Examples:
    #
    # 1 Scope
    # 4.2 Registration Procedure
    # 5.3.2.1 Authentication procedure
    #
    # Candidate detection is intentionally broader than final
    # validation. _is_valid_section() performs additional checks.
    #

    SECTION_PATTERN = re.compile(
        r"^\s*"
        r"(?P<number>\d{1,5}(?:\.\d{1,4}){0,6})"
        r"\s+"
        r"(?P<title>.+?)"
        r"\s*$"
    )

    # ============================================================
    # Common 3GPP specification number
    #
    # Examples:
    #
    # 23.501
    # 23.502
    # 24.501
    # 29.509
    # 33.501
    #
    # These must not be interpreted as clause numbers.
    # ============================================================

    SPECIFICATION_NUMBER_PATTERN = re.compile(
        r"^\d{2}\.\d{3}$"
    )

    # ============================================================
    # Common HTTP status codes occurring in 29.xxx documents
    # ============================================================

    HTTP_STATUS_CODES = {
        100,
        101,
        102,
        103,

        200,
        201,
        202,
        203,
        204,
        205,
        206,
        207,
        208,
        226,

        300,
        301,
        302,
        303,
        304,
        305,
        307,
        308,

        400,
        401,
        402,
        403,
        404,
        405,
        406,
        407,
        408,
        409,
        410,
        411,
        412,
        413,
        414,
        415,
        416,
        417,
        418,
        421,
        422,
        423,
        424,
        425,
        426,
        428,
        429,
        431,
        451,

        500,
        501,
        502,
        503,
        504,
        505,
        506,
        507,
        508,
        510,
        511,
    }

    # ============================================================
    # Titles which are clearly units rather than headings
    # ============================================================

    UNIT_ONLY_PATTERN = re.compile(
        r"^(?:"
        r"ms|"
        r"s|"
        r"sec|"
        r"secs|"
        r"second|"
        r"seconds|"
        r"min|"
        r"mins|"
        r"minute|"
        r"minutes|"
        r"hz|"
        r"khz|"
        r"mhz|"
        r"ghz|"
        r"kbps|"
        r"mbps|"
        r"gbps|"
        r"byte|"
        r"bytes|"
        r"bit|"
        r"bits"
        r")"
        r"(?:\s*\(.*\))?"
        r"$",
        re.IGNORECASE,
    )

    # ============================================================
    # HTTP reason phrases
    # ============================================================

    HTTP_REASON_PHRASES = {
        "continue",
        "switching protocols",
        "processing",
        "early hints",

        "ok",
        "created",
        "accepted",
        "no content",

        "multiple choices",
        "moved permanently",
        "found",
        "see other",
        "not modified",
        "temporary redirect",
        "permanent redirect",

        "bad request",
        "unauthorized",
        "forbidden",
        "not found",
        "method not allowed",
        "request timeout",
        "conflict",
        "gone",
        "payload too large",
        "unsupported media type",
        "too many requests",

        "internal server error",
        "not implemented",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
    }

    # ============================================================
    # Public API
    # ============================================================

    def parse(
        self,
        text: str,
    ) -> List[TelecomSection]:
        """
        Parse document text into hierarchical telecom sections.

        If no valid numbered section headings are detected,
        the complete document is returned as one unstructured
        section.
        """

        if not text or not text.strip():
            return []

        lines = text.splitlines()

        sections: List[TelecomSection] = []

        current_number: Optional[str] = None
        current_title: Optional[str] = None
        current_lines: List[str] = []

        preamble_lines: List[str] = []

        section_found = False

        for line in lines:

            match = self._match_section_heading(
                line
            )

            if match is None:

                if section_found:

                    current_lines.append(
                        line
                    )

                else:

                    preamble_lines.append(
                        line
                    )

                continue

            number = match.group(
                "number"
            )

            title = match.group(
                "title"
            ).strip()

            # ----------------------------------------------------
            # New valid section found
            # ----------------------------------------------------

            section_found = True

            # Save previous section first.

            if current_number is not None:

                sections.append(
                    TelecomSection(
                        section_number=current_number,
                        section_title=current_title,
                        text="\n".join(
                            current_lines
                        ).strip(),
                        parent_section=(
                            self._get_parent_section(
                                current_number
                            )
                        ),
                    )
                )

            current_number = number
            current_title = title

            current_lines = []

        # --------------------------------------------------------
        # Save final section
        # --------------------------------------------------------

        if current_number is not None:

            sections.append(
                TelecomSection(
                    section_number=current_number,
                    section_title=current_title,
                    text="\n".join(
                        current_lines
                    ).strip(),
                    parent_section=(
                        self._get_parent_section(
                            current_number
                        )
                    ),
                )
            )

        # --------------------------------------------------------
        # No valid sections
        # --------------------------------------------------------

        if not sections:

            return [
                TelecomSection(
                    section_number=None,
                    section_title=None,
                    text=text.strip(),
                    parent_section=None,
                )
            ]

        # --------------------------------------------------------
        # Preserve document preamble
        # --------------------------------------------------------

        preamble = "\n".join(
            preamble_lines
        ).strip()

        if preamble:

            sections.insert(
                0,
                TelecomSection(
                    section_number=None,
                    section_title=(
                        "Document Preamble"
                    ),
                    text=preamble,
                    parent_section=None,
                ),
            )

        return sections

    # ============================================================
    # Heading detection
    # ============================================================

    def _match_section_heading(
        self,
        line: str,
    ) -> Optional[re.Match]:
        """
        Determine whether a line is a plausible telecom section
        heading.
        """

        match = self.SECTION_PATTERN.match(
            line
        )

        if match is None:
            return None

        number = match.group(
            "number"
        ).strip()

        title = match.group(
            "title"
        ).strip()

        if not self._is_valid_section(
            number=number,
            title=title,
        ):
            return None

        return match

    # ============================================================
    # Section validation
    # ============================================================

    def _is_valid_section(
        self,
        number: str,
        title: str,
    ) -> bool:
        """
        Apply conservative rules to reject common false headings.
        """

        if not title:
            return False

        # --------------------------------------------------------
        # Title must contain at least one alphabetic character.
        # --------------------------------------------------------

        if not any(
            character.isalpha()
            for character in title
        ):
            return False

        # --------------------------------------------------------
        # Reject 3GPP specification numbers.
        #
        # 23.501
        # 23.502
        # 24.501
        # 29.509
        # 33.501
        # --------------------------------------------------------

        if self.SPECIFICATION_NUMBER_PATTERN.match(
            number
        ):
            return False

        parts = number.split(".")

        # --------------------------------------------------------
        # Top-level clause numbers in the specifications we ingest
        # are small integers.
        #
        # This eliminates:
        #
        # 650 Route des Lucioles
        # 23502 Clarification...
        # 2000 ms
        # --------------------------------------------------------

        if len(parts) == 1:

            try:

                top_level_number = int(
                    parts[0]
                )

            except ValueError:

                return False

            if top_level_number > 99:
                return False

            # ----------------------------------------------------
            # Explicit HTTP status-code rejection
            # ----------------------------------------------------

            if (
                top_level_number
                in self.HTTP_STATUS_CODES
            ):
                return False

        # --------------------------------------------------------
        # Reject titles consisting only of measurement units.
        #
        # Example:
        #
        # 5 ms
        # --------------------------------------------------------

        if self.UNIT_ONLY_PATTERN.match(
            title
        ):
            return False

        # --------------------------------------------------------
        # Reject HTTP status lines even if encountered in unusual
        # formatting.
        # --------------------------------------------------------

        normalized_title = (
            " ".join(
                title.lower().split()
            )
        )

        if normalized_title in (
            self.HTTP_REASON_PHRASES
        ):
            return False
        # --------------------------------------------------------
        # Reject standard 3GPP front-matter version explanations.
        #
        # These lines commonly appear as:
        #
        # 1 presented to TSG for information;
        # 2 presented to TSG for approval;
        # 3 or greater indicates TSG approved document
        #   under change control.
        #
        # They are not document clauses.
        # --------------------------------------------------------

        false_front_matter_phrases = (
            "presented to tsg for information",
            "presented to tsg for approval",
            "or greater indicates tsg approved document under change control",
        )

        if any(
            phrase in normalized_title
            for phrase in false_front_matter_phrases
        ):
            return False
        # --------------------------------------------------------
        # Reject table-of-contents entries with long dotted
        # leaders.
        #
        # Example:
        #
        # 5.2 Registration .............. 47
        # --------------------------------------------------------

        if re.search(
            r"\.{3,}",
            title
        ):
            return False

        # --------------------------------------------------------
        # Extremely long titles are usually body text incorrectly
        # matched as headings.
        # --------------------------------------------------------

        if len(title) > 250:
            return False

        return True

    # ============================================================
    # Hierarchy helper
    # ============================================================

    @staticmethod
    def _get_parent_section(
        section_number: str,
    ) -> Optional[str]:
        """
        Determine the immediate parent of a section number.

        Examples
        --------
        4       -> None
        4.2     -> 4
        4.2.1   -> 4.2
        5.3.2.1 -> 5.3.2
        """

        parts = section_number.split(
            "."
        )

        if len(parts) <= 1:
            return None

        return ".".join(
            parts[:-1]
        )
