"""
telecom_metadata.py

Deterministic telecom metadata extraction for TeleAgent RAG.

Responsibilities
----------------
- Identify 3GPP specifications
- Prefer authoritative specification information from filenames
- Avoid misclassifying non-3GPP documents from references in their body
- Extract release and version information
- Detect common 5G network functions
- Detect telecom protocols/interfaces
- Detect common telecom procedures

This module does NOT:
- perform document loading
- perform section parsing
- perform chunking
- generate embeddings
- access FAISS
- perform retrieval
- call the SLM
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class TelecomMetadata:
    """
    Metadata extracted from a telecom document or document section.
    """

    source: Optional[str] = None

    document_type: Optional[str] = None
    specification_number: Optional[str] = None

    release: Optional[str] = None
    version: Optional[str] = None

    network_functions: List[str] = field(
        default_factory=list
    )

    protocols: List[str] = field(
        default_factory=list
    )

    procedures: List[str] = field(
        default_factory=list
    )


class TelecomMetadataExtractor:
    """
    Extract deterministic telecom metadata.

    Document identity rules
    -----------------------
    1. A recognizable 3GPP filename is considered authoritative.

       Example:

           23502-k30.docx

       is identified as:

           3GPP TS 23.502

       even when the document body mentions many other
       specifications.

    2. Clearly non-3GPP filenames are never classified as a 3GPP
       specification merely because their body references one.

       Example:

           T-REC-Y.3172-201906-I!!PDF-E-1.pdf

       may mention TS 28.554, but it is not TS 28.554.

    3. Text-based specification identification is retained only
       when no reliable source identity is available.
    """

    # ============================================================
    # Network functions / telecom entities
    # ============================================================

    NETWORK_FUNCTIONS = (
        "UE",
        "gNB",
        "AMF",
        "SMF",
        "UPF",
        "AUSF",
        "UDM",
        "UDR",
        "PCF",
        "NRF",
        "NSSF",
        "NEF",
        "AF",
        "CHF",
        "NWDAF",
        "SEPP",
        "SCP",
    )

    # ============================================================
    # Protocols / interfaces
    # ============================================================

    PROTOCOLS = (
        "NAS",
        "NGAP",
        "PFCP",
        "GTP-U",
        "GTP-C",
        "HTTP/2",
        "SBI",
        "RRC",
        "PDCP",
        "RLC",
        "MAC",
        "SDAP",
        "N1",
        "N2",
        "N3",
        "N4",
        "N6",
        "N8",
        "N10",
        "N11",
        "N12",
        "N15",
    )

    # ============================================================
    # Telecom procedures
    # ============================================================

    PROCEDURES = {
        "Registration": (
            "registration",
            "initial registration",
            "mobility registration",
            "periodic registration",
        ),

        "Authentication": (
            "authentication",
            "5g-aka",
            "eap-aka",
            "eap-aka'",
        ),

        "PDU Session Establishment": (
            "pdu session establishment",
            "pdu session establishment procedure",
        ),

        "PDU Session Modification": (
            "pdu session modification",
            "pdu session modification procedure",
        ),

        "PDU Session Release": (
            "pdu session release",
            "pdu session release procedure",
        ),

        "Handover": (
            "handover",
            "hand over",
        ),

        "Mobility": (
            "mobility management",
            "mobility",
        ),

        "Paging": (
            "paging",
        ),

        "Service Request": (
            "service request",
        ),

        "Deregistration": (
            "deregistration",
            "de-registration",
        ),

        "Network Slicing": (
            "network slicing",
            "network slice",
        ),

        "QoS": (
            "qos",
            "quality of service",
        ),

        "Policy Control": (
            "policy control",
        ),

        "Session Management": (
            "session management",
        ),
    }

    # ============================================================
    # Specification patterns inside document text
    # ============================================================

    SPEC_PATTERN = re.compile(
        r"\b"
        r"(?:3GPP\s+)?"
        r"(?P<type>TS|TR)"
        r"\s*"
        r"(?P<number>\d{2}\.\d{3})"
        r"\b",
        re.IGNORECASE,
    )

    VERSION_PATTERN = re.compile(
        r"\bV(?P<version>\d+\.\d+\.\d+)\b",
        re.IGNORECASE,
    )

    RELEASE_PATTERN = re.compile(
        r"\b"
        r"(?:Release|Rel[-\s]?)"
        r"\s*"
        r"(?P<release>\d{1,2})"
        r"\b",
        re.IGNORECASE,
    )

    # ============================================================
    # 3GPP filename patterns
    # ============================================================

    #
    # Examples:
    #
    # 23501.docx
    #       -> TS 23.501
    #
    # 23502-k30.docx
    #       -> TS 23.502
    #
    # 24501-k10_2_Main-Body_s05_s0504.docx
    #       -> TS 24.501
    #
    # 29509-k00.docx
    #       -> TS 29.509
    #
    # 33501-k30.docx
    #       -> TS 33.501
    #
    # 245011.docx
    #       -> TS 24.501
    #

    STANDARD_FILENAME_PATTERN = re.compile(
        r"^(?P<number>\d{5})",
        re.IGNORECASE,
    )

    #
    # Official-style ETSI / 3GPP filename:
    #
    # ts_129513v160600p.pdf
    #
    # 1 + 29513 -> TS 29.513
    #

    OFFICIAL_FILENAME_PATTERN = re.compile(
        r"^(?P<type>ts|tr)[_-]"
        r"1?"
        r"(?P<number>\d{5})"
        r"(?:v|[-_]|$)",
        re.IGNORECASE,
    )

    # ============================================================
    # Version patterns in filenames
    # ============================================================

    #
    # Examples:
    #
    # -k30 -> 20.3.0
    # -k10 -> 20.1.0
    # -k00 -> 20.0.0
    #

    CODED_VERSION_PATTERN = re.compile(
        r"-(?P<major>[a-z])"
        r"(?P<minor>\d)"
        r"(?P<patch>\d)",
        re.IGNORECASE,
    )

    #
    # Example:
    #
    # v160600 -> 16.6.0
    #

    NUMERIC_VERSION_PATTERN = re.compile(
        r"v"
        r"(?P<major>\d{2})"
        r"(?P<minor>\d{2})"
        r"(?P<patch>\d{2})",
        re.IGNORECASE,
    )

    # ============================================================
    # Clearly non-3GPP filename patterns
    # ============================================================

    #
    # These documents may legitimately contain references such as
    # "3GPP TS 28.554", but those references must not become the
    # identity of the entire document.
    #

    NON_3GPP_SOURCE_PATTERNS = (
        re.compile(
            r"^T-REC-",
            re.IGNORECASE,
        ),

        re.compile(
            r"^ITU",
            re.IGNORECASE,
        ),

        re.compile(
            r"^TRAI",
            re.IGNORECASE,
        ),

        re.compile(
            r"^5G_Camera_",
            re.IGNORECASE,
        ),

        re.compile(
            r"^5G_IoTSensor_",
            re.IGNORECASE,
        ),
    )

    # ============================================================
    # Public API
    # ============================================================

    def extract(
        self,
        text: str,
        source: Optional[str] = None,
    ) -> TelecomMetadata:
        """
        Extract telecom metadata.

        Parameters
        ----------
        text:
            Document or section text.

        source:
            Source filename when available.

        Returns
        -------
        TelecomMetadata
        """

        text = text or ""

        # --------------------------------------------------------
        # STEP 1:
        # Attempt authoritative document identification from
        # source filename.
        # --------------------------------------------------------

        (
            filename_document_type,
            filename_specification,
        ) = self._extract_specification_from_source(
            source
        )

        # --------------------------------------------------------
        # STEP 2:
        # Resolve document identity.
        # --------------------------------------------------------

        if filename_specification is not None:

            # ----------------------------------------------------
            # Strongest source of truth:
            # recognizable 3GPP filename.
            # ----------------------------------------------------

            document_type = (
                filename_document_type
            )

            specification_number = (
                filename_specification
            )

        elif self._is_clearly_non_3gpp_source(
            source
        ):

            # ----------------------------------------------------
            # Important:
            #
            # Do NOT identify the document from a 3GPP reference
            # found somewhere inside an ITU/TRAI/vendor document.
            #
            # Example:
            #
            # ITU-T Y.3172 may reference 3GPP TS 28.554.
            #
            # That must not make the complete document appear to
            # be TS 28.554.
            # ----------------------------------------------------

            document_type = None
            specification_number = None

        else:

            # ----------------------------------------------------
            # Source is absent or genuinely ambiguous.
            #
            # Retain body-text detection for backward
            # compatibility and manually supplied text.
            # ----------------------------------------------------

            (
                document_type,
                specification_number,
            ) = self._extract_specification_from_text(
                text
            )

        # --------------------------------------------------------
        # STEP 3:
        # Version / Release
        # --------------------------------------------------------

        #
        # For clearly identified 3GPP files, filename version
        # information is preferred.
        #

        filename_version = (
            self._extract_version_from_source(
                source
            )
        )

        if filename_specification is not None:

            if filename_version is not None:

                version = filename_version
                release = version.split(".")[0]

            else:

                version = (
                    self._extract_version_from_text(
                        text
                    )
                )

                release = (
                    self._extract_release_from_text(
                        text
                    )
                )

                if (
                    release is None
                    and version is not None
                ):

                    release = (
                        version.split(".")[0]
                    )

        elif self._is_clearly_non_3gpp_source(
            source
        ):

            # ----------------------------------------------------
            # Do not assign a 3GPP release/version merely because
            # a non-3GPP document contains a reference to one.
            # ----------------------------------------------------

            version = None
            release = None

        else:

            # ----------------------------------------------------
            # Ambiguous / text-only input.
            # ----------------------------------------------------

            version = (
                self._extract_version_from_text(
                    text
                )
            )

            release = (
                self._extract_release_from_text(
                    text
                )
            )

            if (
                release is None
                and version is not None
            ):

                release = (
                    version.split(".")[0]
                )

        # --------------------------------------------------------
        # STEP 4:
        # Extract content-level telecom metadata.
        #
        # These remain valid even for ITU, TRAI, research, and
        # vendor documents.
        # --------------------------------------------------------

        network_functions = (
            self._extract_network_functions(
                text
            )
        )

        protocols = (
            self._extract_protocols(
                text
            )
        )

        procedures = (
            self._extract_procedures(
                text
            )
        )

        return TelecomMetadata(
            source=source,

            document_type=document_type,

            specification_number=(
                specification_number
            ),

            release=release,
            version=version,

            network_functions=(
                network_functions
            ),

            protocols=protocols,

            procedures=procedures,
        )

    # ============================================================
    # Source classification
    # ============================================================

    @classmethod
    def _is_clearly_non_3gpp_source(
        cls,
        source: Optional[str],
    ) -> bool:
        """
        Return True when the filename clearly represents a
        non-3GPP document.

        This prevents body references to 3GPP specifications from
        incorrectly becoming document identity.
        """

        if not source:
            return False

        filename = Path(
            source
        ).name

        for pattern in (
            cls.NON_3GPP_SOURCE_PATTERNS
        ):

            if pattern.search(
                filename
            ):

                return True

        return False

    # ============================================================
    # Specification extraction from source filename
    # ============================================================

    @classmethod
    def _extract_specification_from_source(
        cls,
        source: Optional[str],
    ) -> tuple[
        Optional[str],
        Optional[str],
    ]:
        """
        Extract 3GPP document identity from filename.

        Returns
        -------
        tuple
            (document_type, specification_number)
        """

        if not source:

            return None, None

        filename = Path(
            source
        ).name

        stem = Path(
            filename
        ).stem.lower()

        # --------------------------------------------------------
        # Official 3GPP / ETSI style
        #
        # Example:
        #
        # ts_129513v160600p
        #
        # -> TS 29.513
        # --------------------------------------------------------

        match = (
            cls.OFFICIAL_FILENAME_PATTERN.search(
                stem
            )
        )

        if match:

            digits = match.group(
                "number"
            )

            specification = (
                f"{digits[:2]}."
                f"{digits[2:]}"
            )

            file_type = (
                match.group("type")
                .lower()
            )

            if file_type == "tr":

                document_type = (
                    "3GPP_TR"
                )

            else:

                document_type = (
                    "3GPP_TS"
                )

            return (
                document_type,
                specification,
            )

        # --------------------------------------------------------
        # 3GPP archive-style names
        #
        # Examples:
        #
        # 23501.docx
        # 23502-k30.docx
        # 24501-k10_2_Main-Body.docx
        # 33501-k30.docx
        # --------------------------------------------------------

        match = (
            cls.STANDARD_FILENAME_PATTERN.search(
                stem
            )
        )

        if match:

            digits = match.group(
                "number"
            )

            specification = (
                f"{digits[:2]}."
                f"{digits[2:]}"
            )

            return (
                "3GPP_TS",
                specification,
            )

        return None, None

    # ============================================================
    # Specification extraction from document text
    # ============================================================

    @classmethod
    def _extract_specification_from_text(
        cls,
        text: str,
    ) -> tuple[
        Optional[str],
        Optional[str],
    ]:
        """
        Extract the first explicit 3GPP specification reference
        from text.

        Important
        ---------
        This method is used only when source identity is missing
        or ambiguous.
        """

        match = (
            cls.SPEC_PATTERN.search(
                text
            )
        )

        if match is None:

            return None, None

        specification_type = (
            match.group(
                "type"
            ).upper()
        )

        specification_number = (
            match.group(
                "number"
            )
        )

        return (
            f"3GPP_{specification_type}",
            specification_number,
        )

    # ============================================================
    # Version extraction from source filename
    # ============================================================

    @classmethod
    def _extract_version_from_source(
        cls,
        source: Optional[str],
    ) -> Optional[str]:
        """
        Extract 3GPP version information from known filename
        conventions.
        """

        if not source:

            return None

        stem = Path(
            source
        ).stem.lower()

        # --------------------------------------------------------
        # Numeric version:
        #
        # v160600 -> 16.6.0
        # --------------------------------------------------------

        numeric_match = (
            cls.NUMERIC_VERSION_PATTERN.search(
                stem
            )
        )

        if numeric_match:

            major = int(
                numeric_match.group(
                    "major"
                )
            )

            minor = int(
                numeric_match.group(
                    "minor"
                )
            )

            patch = int(
                numeric_match.group(
                    "patch"
                )
            )

            return (
                f"{major}."
                f"{minor}."
                f"{patch}"
            )

        # --------------------------------------------------------
        # 3GPP archive version:
        #
        # k30
        #
        # a = 10
        # b = 11
        # ...
        # k = 20
        # --------------------------------------------------------

        coded_match = (
            cls.CODED_VERSION_PATTERN.search(
                stem
            )
        )

        if coded_match:

            letter = (
                coded_match.group(
                    "major"
                ).lower()
            )

            if (
                "a"
                <= letter
                <= "z"
            ):

                major = (
                    ord(letter)
                    - ord("a")
                    + 10
                )

                minor = int(
                    coded_match.group(
                        "minor"
                    )
                )

                patch = int(
                    coded_match.group(
                        "patch"
                    )
                )

                return (
                    f"{major}."
                    f"{minor}."
                    f"{patch}"
                )

        return None

    # ============================================================
    # Version / release extraction from text
    # ============================================================

    @classmethod
    def _extract_version_from_text(
        cls,
        text: str,
    ) -> Optional[str]:
        """
        Extract version string such as V20.3.0.
        """

        match = (
            cls.VERSION_PATTERN.search(
                text
            )
        )

        if match is None:

            return None

        return match.group(
            "version"
        )

    @classmethod
    def _extract_release_from_text(
        cls,
        text: str,
    ) -> Optional[str]:
        """
        Extract release such as Release 20 or Rel-20.
        """

        match = (
            cls.RELEASE_PATTERN.search(
                text
            )
        )

        if match is None:

            return None

        return match.group(
            "release"
        )

    # ============================================================
    # Network functions
    # ============================================================

    def _extract_network_functions(
        self,
        text: str,
    ) -> List[str]:
        """
        Detect known 5G network functions/entities.
        """

        detected: List[str] = []

        for network_function in (
            self.NETWORK_FUNCTIONS
        ):

            pattern = (
                self._term_pattern(
                    network_function
                )
            )

            if re.search(
                pattern,
                text,
                re.IGNORECASE,
            ):

                detected.append(
                    network_function
                )

        return detected

    # ============================================================
    # Protocols
    # ============================================================

    def _extract_protocols(
        self,
        text: str,
    ) -> List[str]:
        """
        Detect telecom protocols and interfaces.
        """

        detected: List[str] = []

        for protocol in (
            self.PROTOCOLS
        ):

            pattern = (
                self._term_pattern(
                    protocol
                )
            )

            if re.search(
                pattern,
                text,
                re.IGNORECASE,
            ):

                detected.append(
                    protocol
                )

        return detected

    # ============================================================
    # Procedures
    # ============================================================

    def _extract_procedures(
        self,
        text: str,
    ) -> List[str]:
        """
        Detect known telecom procedures.
        """

        detected: List[str] = []

        text_lower = (
            text.lower()
        )

        for (
            canonical_name,
            phrases,
        ) in self.PROCEDURES.items():

            for phrase in phrases:

                if self._contains_phrase(
                    text_lower,
                    phrase.lower(),
                ):

                    detected.append(
                        canonical_name
                    )

                    break

        return detected

    # ============================================================
    # Helpers
    # ============================================================

    @staticmethod
    def _term_pattern(
        term: str,
    ) -> str:
        """
        Build an exact telecom-term matching expression.

        This prevents matches such as:

            N1

        being accidentally detected inside:

            N11
        """

        return (
            r"(?<![A-Za-z0-9])"
            + re.escape(term)
            + r"(?![A-Za-z0-9])"
        )

    @staticmethod
    def _contains_phrase(
        text: str,
        phrase: str,
    ) -> bool:
        """
        Check for a complete phrase with conservative boundaries.
        """

        pattern = (
            r"(?<![A-Za-z0-9])"
            + re.escape(phrase)
            + r"(?![A-Za-z0-9])"
        )

        return (
            re.search(
                pattern,
                text,
                re.IGNORECASE,
            )
            is not None
        )
