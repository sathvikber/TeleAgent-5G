"""
Query Type Detector

Deterministically identifies the type of information requested
by a TeleAgent KnowledgeAgent query.

This component intentionally does not use an LLM/SLM.

The detected query type can later be used to:

- improve retrieval queries;
- select appropriate evidence;
- determine whether retrieved evidence is sufficient;
- build smaller query-specific RAG prompts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Literal, Optional


QueryType = Literal[
    "definition",
    "role",
    "services",
    "procedure",
    "comparison",
    "general",
]


@dataclass(frozen=True)
class QueryAnalysis:
    """
    Structured interpretation of a user query.

    Attributes
    ----------
    original_query : str
        Original user query.

    query_type : QueryType
        Detected information need.

    target_entity : Optional[str]
        Main entity being asked about, when identifiable.

    requires_explicit_definition : bool
        True when the question specifically requires
        definition evidence.
    """

    original_query: str
    query_type: QueryType
    target_entity: Optional[str]
    requires_explicit_definition: bool = False


class QueryTypeDetector:
    """
    Rule-based query-type detector for telecom RAG queries.

    The detector deliberately uses deterministic rules instead
    of the SLM so that routing decisions remain predictable.
    """

    # ---------------------------------------------------------
    # Definition patterns
    # ---------------------------------------------------------

    _DEFINITION_PATTERNS = [
        re.compile(
            r"\bwhat\s+is\s+(?:the\s+)?"
            r"(?P<target>[A-Za-z0-9][A-Za-z0-9_./*+-]*)"
            r"\s*\??\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bdefine\s+"
            r"(?P<target>[A-Za-z0-9][A-Za-z0-9_./*+-]*)"
            r"\s*\??\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwhat\s+does\s+"
            r"(?P<target>[A-Za-z0-9][A-Za-z0-9_./*+-]*)"
            r"\s+stand\s+for\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bmeaning\s+of\s+"
            r"(?P<target>[A-Za-z0-9][A-Za-z0-9_./*+-]*)",
            re.IGNORECASE,
        ),
    ]

    # ---------------------------------------------------------
    # Role patterns
    # ---------------------------------------------------------

    _ROLE_PATTERNS = [
        re.compile(
            r"\brole\s+of\s+"
            r"(?P<target>[A-Za-z0-9][A-Za-z0-9_./*+-]*)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwhat\s+does\s+"
            r"(?P<target>[A-Za-z0-9][A-Za-z0-9_./*+-]*)"
            r"\s+do\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bresponsibilit(?:y|ies)\s+of\s+"
            r"(?P<target>[A-Za-z0-9][A-Za-z0-9_./*+-]*)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bfunction\s+of\s+"
            r"(?P<target>[A-Za-z0-9][A-Za-z0-9_./*+-]*)",
            re.IGNORECASE,
        ),
    ]

    # ---------------------------------------------------------
    # Service patterns
    # ---------------------------------------------------------

    _SERVICE_PATTERNS = [
        re.compile(
            r"\bwhat\s+services?\s+does\s+"
            r"(?P<target>[A-Za-z0-9][A-Za-z0-9_./*+-]*)"
            r"\s+provide\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bservices?\s+provided\s+by\s+"
            r"(?P<target>[A-Za-z0-9][A-Za-z0-9_./*+-]*)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bservices?\s+of\s+"
            r"(?P<target>[A-Za-z0-9][A-Za-z0-9_./*+-]*)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bservice\s+operations?\s+of\s+"
            r"(?P<target>[A-Za-z0-9][A-Za-z0-9_./*+-]*)",
            re.IGNORECASE,
        ),
    ]

    # ---------------------------------------------------------
    # Comparison patterns
    # ---------------------------------------------------------

    _COMPARISON_PATTERNS = [
        re.compile(
            r"\bdifference\s+between\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bcompare\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bversus\b|\bvs\.?\b",
            re.IGNORECASE,
        ),
    ]

    # ---------------------------------------------------------
    # Procedure indicators
    # ---------------------------------------------------------

    _PROCEDURE_PATTERNS = [
        re.compile(r"\bprocedure\b", re.IGNORECASE),
        re.compile(r"\bprocess\b", re.IGNORECASE),
        re.compile(r"\bsteps?\b", re.IGNORECASE),
        re.compile(r"\bhow\s+does\b", re.IGNORECASE),
        re.compile(r"\bhow\s+is\b", re.IGNORECASE),
        re.compile(r"\bparticipate\b", re.IGNORECASE),
        re.compile(r"\bparticipates?\b", re.IGNORECASE),
        re.compile(r"\bauthentication\s+flow\b", re.IGNORECASE),
    ]

    # Tokens that should not be selected as the primary entity
    # when performing fallback acronym extraction.
    _IGNORED_TARGETS = {
        "TS",
        "3GPP",
        "ETSI",
        "ITU",
        "AKA",
        "RAG",
    }

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def detect(self, query: str) -> QueryAnalysis:
        """
        Detect query type and target entity.

        Parameters
        ----------
        query : str
            User query.

        Returns
        -------
        QueryAnalysis
            Structured query interpretation.
        """

        query = (query or "").strip()

        if not query:
            return QueryAnalysis(
                original_query=query,
                query_type="general",
                target_entity=None,
                requires_explicit_definition=False,
            )

        # -----------------------------------------------------
        # Services
        # -----------------------------------------------------

        target = self._match_target(
            query,
            self._SERVICE_PATTERNS,
        )

        if target:
            return QueryAnalysis(
                original_query=query,
                query_type="services",
                target_entity=self._clean_target(target),
                requires_explicit_definition=False,
            )

        # -----------------------------------------------------
        # Role
        # -----------------------------------------------------

        target = self._match_target(
            query,
            self._ROLE_PATTERNS,
        )

        if target:
            return QueryAnalysis(
                original_query=query,
                query_type="role",
                target_entity=self._clean_target(target),
                requires_explicit_definition=False,
            )

        # -----------------------------------------------------
        # Definition
        # -----------------------------------------------------

        target = self._match_target(
            query,
            self._DEFINITION_PATTERNS,
        )

        if target:
            return QueryAnalysis(
                original_query=query,
                query_type="definition",
                target_entity=self._clean_target(target),
                requires_explicit_definition=True,
            )

        # -----------------------------------------------------
        # Comparison
        # -----------------------------------------------------

        if any(
            pattern.search(query)
            for pattern in self._COMPARISON_PATTERNS
        ):
            return QueryAnalysis(
                original_query=query,
                query_type="comparison",
                target_entity=self._fallback_target(query),
                requires_explicit_definition=False,
            )

        # -----------------------------------------------------
        # Procedure
        # -----------------------------------------------------

        if any(
            pattern.search(query)
            for pattern in self._PROCEDURE_PATTERNS
        ):
            return QueryAnalysis(
                original_query=query,
                query_type="procedure",
                target_entity=self._fallback_target(query),
                requires_explicit_definition=False,
            )

        # -----------------------------------------------------
        # General
        # -----------------------------------------------------

        return QueryAnalysis(
            original_query=query,
            query_type="general",
            target_entity=self._fallback_target(query),
            requires_explicit_definition=False,
        )

    def build_retrieval_queries(
        self,
        analysis: QueryAnalysis,
    ) -> List[str]:
        """
        Build retrieval-query variants according to query type.

        These are retrieval hints only. Existing specification
        filtering and hybrid retrieval remain responsible for
        selecting final evidence.

        Parameters
        ----------
        analysis : QueryAnalysis
            Result produced by detect().

        Returns
        -------
        List[str]
            Retrieval-query variants.
        """

        original = analysis.original_query.strip()
        target = analysis.target_entity

        queries: List[str] = []

        if original:
            queries.append(original)

        if not target:
            return self._deduplicate(queries)

        if analysis.query_type == "definition":

            queries.extend(
                [
                    f"{target} definition",
                    f"{target} abbreviation definition",
                    f"{target} network function definition",
                    f"definition of {target}",
                ]
            )

        elif analysis.query_type == "role":

            queries.extend(
                [
                    f"{target} role",
                    f"{target} responsibilities",
                    f"{target} function responsibilities",
                ]
            )

        elif analysis.query_type == "services":

            queries.extend(
                [
                    f"{target} services",
                    f"services provided by {target}",
                    f"{target} service operations",
                ]
            )

        elif analysis.query_type == "procedure":

            queries.extend(
                [
                    f"{target} procedure",
                    f"{target} process",
                ]
            )

        return self._deduplicate(queries)

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------

    @staticmethod
    def _match_target(
        query: str,
        patterns: List[re.Pattern],
    ) -> Optional[str]:
        """
        Return named target captured by the first matching rule.
        """

        for pattern in patterns:
            match = pattern.search(query)

            if match:
                target = match.groupdict().get("target")

                if target:
                    return target

        return None

    @staticmethod
    def _clean_target(target: str) -> str:
        """
        Normalize a detected target while preserving acronyms.
        """

        target = target.strip()
        target = target.strip(" ?.,:;()[]{}")

        # Preserve acronym case when supplied by the user.
        return target

    def _fallback_target(
        self,
        query: str,
    ) -> Optional[str]:
        """
        Extract a likely telecom acronym when no structured
        target pattern matched.

        This is intentionally conservative.
        """

        candidates = re.findall(
            r"\b(?:[A-Z][A-Z0-9]{1,}(?:/[A-Z][A-Z0-9]{1,})?|[0-9]+G)\b",
            query,
        )

        for candidate in candidates:

            if candidate.upper() in self._IGNORED_TARGETS:
                continue

            return candidate

        return None

    @staticmethod
    def _deduplicate(
        values: List[str],
    ) -> List[str]:
        """
        Preserve query order while removing duplicates.
        """

        seen = set()
        result = []

        for value in values:

            normalized = value.strip()

            if not normalized:
                continue

            key = normalized.lower()

            if key in seen:
                continue

            seen.add(key)
            result.append(normalized)

        return result
