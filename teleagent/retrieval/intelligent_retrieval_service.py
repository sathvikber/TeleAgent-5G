"""
Intelligent Retrieval Service

Hybrid retrieval pipeline for TeleAgent.

Pipeline
--------
User Query
    |
    v
QueryProcessor
    |
    +----------------------+
    |                      |
    v                      v
DenseRetriever       SparseRetriever
    |                      |
    +----------+-----------+
               |
               v
      Specification Filter
               |
               v
            Fusion
               |
               v
           ReRanker
               |
               v
   Hierarchical Expansion
               |
               v
     Compact Evidence Builder
               |
               v
       RAGPromptBuilder

Features
--------
- Dense retrieval
- Sparse retrieval
- Reciprocal-rank fusion
- Reranking
- Explicit 3GPP specification awareness
- Parent/child/neighbor expansion
- Compact evidence construction
- Canonical acronym-definition enrichment
- Metadata-assisted entity grounding
- Optional RAG debugging
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from teleagent.rag.rag_service import RAGService
from teleagent.rag.schemas import Chunk, SearchResult

from .query_processor import QueryProcessor
from .dense_retriever import DenseRetriever
from .sparse_retriever import SparseRetriever
from .fusion import Fusion
from .reranker import ReRanker
from .rag_prompt_builder import RAGPromptBuilder
from .query_type_detector import QueryAnalysis, QueryTypeDetector
from .evidence_sufficiency_checker import EvidenceSufficiencyChecker


@dataclass(frozen=True)
class RAGPreparationResult:
    """
    Result of preparing a knowledge query for grounded inference.

    should_infer:
        True only when retrieved evidence is sufficient.

    prompt:
        Final RAG prompt when SLM inference is allowed.

    direct_response:
        Deterministic response returned when the SLM must be bypassed.

    evidence:
        Compact evidence produced by the original retrieval pipeline.

    query_analysis:
        Query type and target produced by QueryTypeDetector.

    sufficiency_reason:
        Explanation returned by EvidenceSufficiencyChecker.
    """

    should_infer: bool
    prompt: Optional[str]
    direct_response: Optional[str]
    evidence: str
    query_analysis: QueryAnalysis
    sufficiency_reason: Optional[str] = None


class IntelligentRetrievalService:
    """
    Intelligent hybrid retrieval service used by KnowledgeAgent.

    Retrieval pipeline:

        query processing
            ↓
        dense + sparse retrieval
            ↓
        specification filtering
            ↓
        fusion
            ↓
        reranking
            ↓
        hierarchical expansion
            ↓
        compact evidence construction
            ↓
        RAG prompt
    """

    # ============================================================
    # Explicit standards reference
    # ============================================================

    STANDARD_PATTERN = re.compile(
        r"\b"
        r"(?:3gpp\s+)?"
        r"(?:ts|tr)"
        r"\s*"
        r"(?P<spec>\d{2}\.\d{3})"
        r"\b",
        re.IGNORECASE,
    )

    # ============================================================
    # Acronym detection
    #
    # Examples:
    #
    # AMF
    # AUSF
    # UDM
    # ARPF
    # SEAF
    # NAS
    # KSEAF
    # ============================================================

    ACRONYM_PATTERN = re.compile(
        r"\b[A-Z][A-Z0-9_-]{1,15}\b"
    )

    # ============================================================
    # Common query words which should not influence evidence
    # ranking.
    # ============================================================

    QUERY_STOPWORDS = {
        "what",
        "which",
        "who",
        "where",
        "when",
        "why",
        "how",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "from",
        "by",
        "and",
        "or",
        "according",
        "does",
        "do",
        "did",
        "that",
        "this",
        "these",
        "those",
        "its",
        "their",
        "participate",
        "participates",
        "participating",
        "involved",
    }


    # ============================================================
    # Compact-evidence configuration
    # ============================================================

    AUTHENTICATION_VALUES = (
        "XRES*",
        "HXRES*",
        "HRES*",
        "RES*",
        "KAUSF",
        "KSEAF",
        "KAMF",
        "RAND",
        "AUTN",
        "ABBA",
        "CK",
        "IK",
        "ngKSI",
    )

    DEFINITION_SECTION_KEYWORDS = (
        "definition",
        "definitions",
        "abbreviation",
        "abbreviations",
        "network functions and entities",
        "network functions",
        "security entities",
        "security entities in the 5g core network",
        "terminology",
    )

    # Tokens that must never be promoted to canonical telecom
    # entity definitions merely because they are uppercase.
    NON_ENTITY_TOKENS = {
        "TS",
        "TR",
        "NOTE",
        "NF",
        "AKA",
        "NAS",
        "AV",
        "HE",
        "SE",
        "EPS",
        "5G",
        "3GPP",
        "SUPI",
        "SUCI",
        "RAND",
        "AUTN",
        "RES",
        "RES*",
        "XRES*",
        "HXRES*",
        "HRES*",
        "KAUSF",
        "KSEAF",
        "KAMF",
        "ABBA",
        "CK",
        "IK",
        "NGKSI",
    }

    def __init__(
        self,
        rag_service: RAGService,
        query_processor: Optional[QueryProcessor] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        fusion: Optional[Fusion] = None,
        reranker: Optional[ReRanker] = None,
        prompt_builder: Optional[RAGPromptBuilder] = None,
        query_type_detector: Optional[QueryTypeDetector] = None,
        evidence_sufficiency_checker: Optional[EvidenceSufficiencyChecker] = None,
    ) -> None:

        self._rag_service = rag_service

        # --------------------------------------------------------
        # Query processing
        # --------------------------------------------------------

        self._query_processor = (
            query_processor
            if query_processor is not None
            else QueryProcessor()
        )

        # --------------------------------------------------------
        # Dense retrieval
        # --------------------------------------------------------

        self._dense_retriever = (
            dense_retriever
            if dense_retriever is not None
            else DenseRetriever(
                rag_service.retriever
            )
        )

        # --------------------------------------------------------
        # Sparse retrieval
        # --------------------------------------------------------

        self._sparse_retriever = (
            sparse_retriever
            if sparse_retriever is not None
            else SparseRetriever(
                rag_service.vector_store
            )
        )

        # --------------------------------------------------------
        # Fusion
        # --------------------------------------------------------

        self._fusion = (
            fusion
            if fusion is not None
            else Fusion()
        )

        # --------------------------------------------------------
        # Reranker
        # --------------------------------------------------------

        self._reranker = (
            reranker
            if reranker is not None
            else ReRanker()
        )

        # --------------------------------------------------------
        # Prompt builder
        # --------------------------------------------------------

        self._prompt_builder = (
            prompt_builder
            if prompt_builder is not None
            else RAGPromptBuilder()
        )

        # --------------------------------------------------------
        # Query analysis and deterministic evidence gate
        # --------------------------------------------------------

        self._query_type_detector = (
            query_type_detector
            if query_type_detector is not None
            else QueryTypeDetector()
        )

        self._evidence_sufficiency_checker = (
            evidence_sufficiency_checker
            if evidence_sufficiency_checker is not None
            else EvidenceSufficiencyChecker()
        )

        # --------------------------------------------------------
        # Debug mode
        #
        # export TELEAGENT_RAG_DEBUG=1
        # --------------------------------------------------------

        self._debug = (
            os.getenv(
                "TELEAGENT_RAG_DEBUG",
                "0",
            )
            .strip()
            .lower()
            in {
                "1",
                "true",
                "yes",
                "on",
            }
        )

    # ============================================================
    # Public API
    # ============================================================

    def prepare(
        self,
        query: str,
        top_k: int = 5,
    ) -> RAGPreparationResult:
        """
        Prepare a knowledge query for the KnowledgeAgent.

        The original retrieval pipeline is executed unchanged.
        QueryTypeDetector and EvidenceSufficiencyChecker are applied
        only after compact evidence has been constructed.

        If evidence is insufficient, no RAG prompt is built and the
        KnowledgeAgent can bypass the SLM.
        """

        query = (query or "").strip()

        analysis = self._query_type_detector.detect(
            query
        )

        if self._debug:

            print()
            print("=" * 80)
            print("QUERY ANALYSIS")
            print("=" * 80)
            print(
                "Query Type :",
                analysis.query_type,
            )
            print(
                "Target     :",
                analysis.target_entity,
            )
            print(
                "Definition :",
                analysis.requires_explicit_definition,
            )

        retrieved_context = self.build_evidence(
            query=query,
            top_k=top_k,
        )

        # The original compact-context builder returns a textual
        # sentinel when nothing was retrieved. Convert only that
        # sentinel to empty evidence for the deterministic checker.
        evidence_for_check = retrieved_context

        if (
            retrieved_context
            and retrieved_context.strip().lower().startswith(
                "no relevant information was retrieved"
            )
        ):
            evidence_for_check = ""

        sufficiency = (
            self._evidence_sufficiency_checker.check(
                analysis=analysis,
                evidence=evidence_for_check,
            )
        )

        if self._debug:

            print()
            print("=" * 80)
            print("EVIDENCE SUFFICIENCY")
            print("=" * 80)
            print(
                "Query Type :",
                analysis.query_type,
            )
            print(
                "Target     :",
                analysis.target_entity,
            )
            print(
                "Sufficient :",
                sufficiency.is_sufficient,
            )
            print(
                "Reason     :",
                sufficiency.reason,
            )

            matched = getattr(
                sufficiency,
                "matched_evidence",
                None,
            )

            if matched:

                print(
                    "Matched    :",
                    matched,
                )

        if not sufficiency.is_sufficient:

            return RAGPreparationResult(
                should_infer=False,
                prompt=None,
                direct_response=(
                    EvidenceSufficiencyChecker
                    .FALLBACK_RESPONSE
                ),
                evidence=retrieved_context,
                query_analysis=analysis,
                sufficiency_reason=(
                    sufficiency.reason
                ),
            )

        prompt = self._prompt_builder.build(
            query=query,
            retrieved_context=(
                retrieved_context
            ),
        )

        return RAGPreparationResult(
            should_infer=True,
            prompt=prompt,
            direct_response=None,
            evidence=retrieved_context,
            query_analysis=analysis,
            sufficiency_reason=(
                sufficiency.reason
            ),
        )

    def build_prompt(
        self,
        query: str,
        top_k: int = 5,
    ) -> str:
        """
        Compatibility API for existing standalone tests.

        KnowledgeAgent should use prepare() so it can bypass the SLM
        when evidence is insufficient.
        """

        preparation = self.prepare(
            query=query,
            top_k=top_k,
        )

        if not preparation.should_infer:

            return (
                preparation.direct_response
                or EvidenceSufficiencyChecker.FALLBACK_RESPONSE
            )

        return preparation.prompt or ""

    def build_evidence(
        self,
        query: str,
        top_k: int = 5,
    ) -> str:
        """
        Run the original working retrieval pipeline and return
        compact structured evidence before prompt construction.
        """

        # --------------------------------------------------------
        # STEP 1
        # Process query
        # --------------------------------------------------------

        processed_query = (
            self._query_processor.process(
                query
            )
        )

        # --------------------------------------------------------
        # STEP 2
        # Detect explicitly requested specification
        # --------------------------------------------------------

        requested_specification = (
            self._extract_requested_specification(
                query
            )
        )

        # --------------------------------------------------------
        # Retrieve a larger candidate pool.
        # --------------------------------------------------------

        candidate_top_k = max(
            top_k * 6,
            30,
        )

        # --------------------------------------------------------
        # STEP 3
        # Dense retrieval
        # --------------------------------------------------------

        dense_results = (
            self._dense_retriever.retrieve(
                processed_query,
                top_k=candidate_top_k,
            )
        )

        # --------------------------------------------------------
        # STEP 4
        # Sparse retrieval
        # --------------------------------------------------------

        sparse_results = (
            self._sparse_retriever.retrieve(
                processed_query,
                top_k=candidate_top_k,
            )
        )

        # --------------------------------------------------------
        # Debug
        # --------------------------------------------------------

        if self._debug:

            print()
            print("=" * 80)
            print("RAG DEBUG")
            print("=" * 80)

            print("Original Query:")
            print(query)

            print()
            print("Processed Query:")
            print(processed_query)

            print()
            print(
                "Requested Specification:",
                requested_specification,
            )

            self._print_results(
                "DENSE RETRIEVAL",
                dense_results[:10],
            )

            self._print_results(
                "SPARSE RETRIEVAL",
                sparse_results[:10],
            )

        # --------------------------------------------------------
        # STEP 5
        # Specification-aware filtering
        # --------------------------------------------------------

        if requested_specification:

            filtered_dense = (
                self._filter_by_specification(
                    dense_results,
                    requested_specification,
                )
            )

            filtered_sparse = (
                self._filter_by_specification(
                    sparse_results,
                    requested_specification,
                )
            )

            # ----------------------------------------------------
            # Use same-spec evidence whenever either retrieval
            # branch found documents from the requested TS/TR.
            # ----------------------------------------------------

            if (
                filtered_dense
                or filtered_sparse
            ):

                dense_results = (
                    filtered_dense
                )

                sparse_results = (
                    filtered_sparse
                )

                if self._debug:

                    self._print_results(
                        (
                            "DENSE AFTER "
                            "SPECIFICATION FILTER"
                        ),
                        dense_results[:10],
                    )

                    self._print_results(
                        (
                            "SPARSE AFTER "
                            "SPECIFICATION FILTER"
                        ),
                        sparse_results[:10],
                    )

        # --------------------------------------------------------
        # STEP 6
        # Fusion
        # --------------------------------------------------------

        fusion_top_k = max(
            top_k * 4,
            20,
        )

        fused_results = (
            self._fusion.fuse(
                dense_results=dense_results,
                sparse_results=sparse_results,
                top_k=fusion_top_k,
            )
        )

        if self._debug:

            self._print_results(
                "FUSED RESULTS",
                fused_results[:10],
            )

        # --------------------------------------------------------
        # STEP 7
        # Reranking
        # --------------------------------------------------------

        reranked_results = (
            self._reranker.rerank(
                query=processed_query,
                results=fused_results,
                top_k=top_k,
            )
        )

        if self._debug:

            self._print_results(
                "FINAL RERANKED RESULTS",
                reranked_results,
            )

        # --------------------------------------------------------
        # STEP 8
        # Hierarchical / neighbor expansion
        # --------------------------------------------------------

        expanded_results = (
            self._expand_results(
                reranked_results,
                requested_specification=(
                    requested_specification
                ),
                max_results=max(
                    top_k * 2,
                    10,
                ),
            )
        )

        if self._debug:

            self._print_results(
                "EXPANDED CONTEXT RESULTS",
                expanded_results,
            )

        # --------------------------------------------------------
        # STEP 9
        # NEW:
        # Build compact, structured evidence.
        #
        # Instead of giving the SLM many raw chunks, produce:
        #
        #   - metadata-detected network functions
        #   - exact canonical definitions
        #   - only the strongest evidence excerpts
        #
        # This reduces hallucination for the 1.5B SLM.
        # --------------------------------------------------------

        retrieved_context = (
            self._build_compact_context(
                query=query,
                results=expanded_results,
                requested_specification=(
                    requested_specification
                ),
            )
        )

        if self._debug:

            print()
            print("=" * 80)
            print("COMPACT RETRIEVED EVIDENCE")
            print("=" * 80)
            print(retrieved_context)
            print("=" * 80)

        # --------------------------------------------------------
        # Return compact evidence.
        #
        # IMPORTANT:
        # Everything above this point is the original validated
        # retrieval pipeline. Prompt construction now happens in
        # prepare() / build_prompt() so that evidence sufficiency
        # can be checked before invoking the SLM.
        # --------------------------------------------------------

        return retrieved_context

    # ============================================================
    # Specification detection
    # ============================================================

    @classmethod
    def _extract_requested_specification(
        cls,
        query: str,
    ) -> Optional[str]:

        if not query:
            return None

        match = (
            cls.STANDARD_PATTERN.search(
                query
            )
        )

        if not match:
            return None

        return match.group(
            "spec"
        )

    # ============================================================
    # Specification-aware filtering
    # ============================================================

    @staticmethod
    def _filter_by_specification(
        results: List[SearchResult],
        specification: str,
    ) -> List[SearchResult]:

        filtered: List[SearchResult] = []

        for result in results:

            metadata = (
                result.chunk.metadata
                or {}
            )

            chunk_spec = (
                metadata.get(
                    "specification_number"
                )
            )

            if chunk_spec == specification:

                filtered.append(
                    result
                )

        return filtered

    # ============================================================
    # Hierarchical expansion
    # ============================================================

    def _expand_results(
        self,
        results: List[SearchResult],
        requested_specification: Optional[str],
        max_results: int,
    ) -> List[SearchResult]:
        """
        Expand retrieved chunks with same-section continuation,
        child sections and nearby chunks.
        """

        if not results:
            return []

        all_chunks = list(
            self._rag_service
            .vector_store
            .chunks
        )

        if not all_chunks:
            return results

        position_by_id = {
            chunk.id: index
            for index, chunk
            in enumerate(all_chunks)
        }

        expanded: List[SearchResult] = []

        seen_ids: Set[str] = set()

        for result in results:

            if len(expanded) >= max_results:
                break

            anchor = result.chunk

            # ----------------------------------------------------
            # Keep original result
            # ----------------------------------------------------

            self._append_unique_result(
                expanded=expanded,
                seen_ids=seen_ids,
                chunk=anchor,
                score=result.score,
            )

            if len(expanded) >= max_results:
                break

            metadata = (
                anchor.metadata
                or {}
            )

            anchor_spec = (
                metadata.get(
                    "specification_number"
                )
            )

            anchor_section = (
                metadata.get(
                    "section_number"
                )
            )

            anchor_document = (
                anchor.document_id
            )

            if (
                requested_specification
                and anchor_spec
                != requested_specification
            ):
                continue

            # ----------------------------------------------------
            # 1. Same-section continuation
            # ----------------------------------------------------

            if anchor_section:

                same_section_chunks = [
                    chunk
                    for chunk in all_chunks
                    if (
                        chunk.document_id
                        == anchor_document
                        and (
                            chunk.metadata
                            or {}
                        ).get(
                            "section_number"
                        )
                        == anchor_section
                        and chunk.id
                        != anchor.id
                    )
                ]

                same_section_chunks.sort(
                    key=lambda chunk: (
                        (
                            chunk.metadata
                            or {}
                        ).get(
                            "chunk_index",
                            0,
                        )
                    )
                )

                for chunk in (
                    same_section_chunks[:2]
                ):

                    if (
                        len(expanded)
                        >= max_results
                    ):
                        break

                    self._append_unique_result(
                        expanded=expanded,
                        seen_ids=seen_ids,
                        chunk=chunk,
                        score=(
                            result.score
                            * 0.99
                        ),
                    )

            if len(expanded) >= max_results:
                break

            # ----------------------------------------------------
            # 2. Child-section expansion
            # ----------------------------------------------------

            if anchor_section:

                child_prefix = (
                    anchor_section
                    + "."
                )

                child_chunks = []

                for chunk in all_chunks:

                    if (
                        chunk.document_id
                        != anchor_document
                    ):
                        continue

                    child_metadata = (
                        chunk.metadata
                        or {}
                    )

                    child_spec = (
                        child_metadata.get(
                            "specification_number"
                        )
                    )

                    if (
                        requested_specification
                        and child_spec
                        != requested_specification
                    ):
                        continue

                    child_section = (
                        child_metadata.get(
                            "section_number"
                        )
                    )

                    if not child_section:
                        continue

                    if child_section.startswith(
                        child_prefix
                    ):

                        child_chunks.append(
                            chunk
                        )

                child_chunks.sort(
                    key=lambda chunk: (
                        position_by_id.get(
                            chunk.id,
                            10**9,
                        )
                    )
                )

                for chunk in (
                    child_chunks[:4]
                ):

                    if (
                        len(expanded)
                        >= max_results
                    ):
                        break

                    self._append_unique_result(
                        expanded=expanded,
                        seen_ids=seen_ids,
                        chunk=chunk,
                        score=(
                            result.score
                            * 0.98
                        ),
                    )

            if len(expanded) >= max_results:
                break

            # ----------------------------------------------------
            # 3. Neighbor expansion for heading-only chunks
            # ----------------------------------------------------

            if self._is_short_heading_chunk(
                anchor
            ):

                anchor_position = (
                    position_by_id.get(
                        anchor.id
                    )
                )

                if (
                    anchor_position
                    is not None
                ):

                    neighbor_end = min(
                        anchor_position + 4,
                        len(all_chunks),
                    )

                    for index in range(
                        anchor_position + 1,
                        neighbor_end,
                    ):

                        if (
                            len(expanded)
                            >= max_results
                        ):
                            break

                        neighbor = (
                            all_chunks[index]
                        )

                        if (
                            neighbor.document_id
                            != anchor_document
                        ):
                            continue

                        neighbor_metadata = (
                            neighbor.metadata
                            or {}
                        )

                        neighbor_spec = (
                            neighbor_metadata.get(
                                "specification_number"
                            )
                        )

                        if (
                            requested_specification
                            and neighbor_spec
                            != requested_specification
                        ):
                            continue

                        self._append_unique_result(
                            expanded=expanded,
                            seen_ids=seen_ids,
                            chunk=neighbor,
                            score=(
                                result.score
                                * 0.97
                            ),
                        )

        return expanded

    # ============================================================
    # Compact evidence construction
    # ============================================================

    def _build_compact_context(
        self,
        query: str,
        results: List[SearchResult],
        requested_specification: Optional[str],
    ) -> str:
        """
        Convert retrieved results into compact, structured evidence.

        Important design rules:

        - metadata entities are hints, not authoritative participants;
        - canonical definitions are accepted only from definition-like
          sections and strict acronym-definition forms;
        - authentication values/keys/parameters are separated from
          network entities;
        - AMF is explicitly disambiguated when the retrieved procedure
          contains "Authentication Management Field (AMF)";
        - only the strongest evidence excerpts are supplied to the SLM.
        """

        if not results:
            return (
                "No relevant information was retrieved "
                "from the current knowledge base."
            )

        # --------------------------------------------------------
        # Select only the most query-relevant chunks.
        # --------------------------------------------------------

        selected_results = (
            self._select_evidence_results(
                query=query,
                results=results,
                max_results=6,
            )
        )

        if not selected_results:
            return (
                "No relevant information was retrieved "
                "from the current knowledge base."
            )

        # --------------------------------------------------------
        # Combined raw evidence text.
        # --------------------------------------------------------

        raw_text = "\n\n".join(
            (
                result.chunk.content
                or ""
            ).strip()
            for result in selected_results
            if (
                result.chunk.content
                or ""
            ).strip()
        )

        # --------------------------------------------------------
        # Choose the primary section.
        #
        # Prefer the highest-ranked substantive chunk rather than
        # a heading-only parent section.
        # --------------------------------------------------------

        primary_result = None

        for result in selected_results:

            if not self._is_short_heading_chunk(
                result.chunk
            ):
                primary_result = result
                break

        if primary_result is None:
            primary_result = selected_results[0]

        primary_chunk = primary_result.chunk

        primary_metadata = (
            primary_chunk.metadata
            or {}
        )

        primary_section = (
            primary_metadata.get(
                "section_number"
            )
        )

        primary_title = (
            primary_metadata.get(
                "section_title"
            )
        )

        # --------------------------------------------------------
        # Metadata-derived entity hints.
        #
        # These are explicitly labelled as hints so the SLM does not
        # assume every metadata token is a procedure participant.
        # --------------------------------------------------------

        entity_hints = (
            self._collect_network_functions(
                selected_results
            )
        )

        # --------------------------------------------------------
        # Candidate entity acronyms.
        #
        # Start with metadata hints and add uppercase tokens from
        # retrieved evidence. They are candidates only; they become
        # canonical definitions only if a strict definition is found
        # in a definition-like knowledge-base section.
        # --------------------------------------------------------

        candidate_entities: Set[str] = set(
            entity_hints
        )

        candidate_entities.update(
            self.ACRONYM_PATTERN.findall(
                raw_text
            )
        )

        candidate_entities = {
            str(item).strip().upper()
            for item in candidate_entities
            if str(item).strip()
        }

        candidate_entities -= (
            self.NON_ENTITY_TOKENS
        )

        canonical_definitions = (
            self._find_canonical_definitions(
                acronyms=candidate_entities,
                requested_specification=(
                    requested_specification
                ),
                retrieved_text=raw_text,
            )
        )

        # --------------------------------------------------------
        # AMF ambiguity.
        # --------------------------------------------------------

        has_authentication_management_field = (
            re.search(
                r"\bAuthentication\s+"
                r"Management\s+Field\s*"
                r"\(\s*AMF\s*\)",
                raw_text,
                flags=re.IGNORECASE,
            )
            is not None
        )

        # --------------------------------------------------------
        # Authentication values / keys / parameters.
        # --------------------------------------------------------

        authentication_values = []

        for token in self.AUTHENTICATION_VALUES:

            escaped = re.escape(
                token
            )

            if re.search(
                escaped,
                raw_text,
                flags=re.IGNORECASE,
            ):
                authentication_values.append(
                    token
                )

        # --------------------------------------------------------
        # Build structured compact evidence.
        # --------------------------------------------------------

        sections: List[str] = []

        if requested_specification:

            sections.append(
                "REQUESTED SPECIFICATION\n"
                f"TS {requested_specification}"
            )

        # --------------------------------------------------------
        # Primary retrieved section.
        # --------------------------------------------------------

        if (
            primary_section
            or primary_title
        ):

            if (
                primary_section
                and primary_title
            ):

                primary_value = (
                    f"Section {primary_section} "
                    f"— {primary_title}"
                )

            elif primary_section:

                primary_value = (
                    f"Section {primary_section}"
                )

            else:

                primary_value = str(
                    primary_title
                )

            sections.append(
                "PRIMARY RETRIEVED SECTION\n"
                + primary_value
            )

        # --------------------------------------------------------
        # Metadata hints.
        # --------------------------------------------------------

        if entity_hints:

            sections.append(
                "ENTITY HINTS FROM RETRIEVED METADATA\n"
                + ", ".join(
                    entity_hints
                )
                + "\n\n"
                "These values are retrieval metadata hints only. "
                "Do NOT classify an item as a network function "
                "solely because it appears in this list. Use the "
                "retrieved procedure text and canonical definitions "
                "to determine the entity type."
            )

        # --------------------------------------------------------
        # Strict canonical definitions.
        # --------------------------------------------------------

        if canonical_definitions:

            definition_lines = []

            for acronym in sorted(
                canonical_definitions
            ):

                definition_lines.append(
                    f"{acronym}: "
                    f"{canonical_definitions[acronym]}"
                )

            sections.append(
                "CANONICAL DEFINITIONS FOUND "
                "IN DEFINITION SECTIONS\n"
                + "\n".join(
                    definition_lines
                )
                + "\n\n"
                "These definitions were copied from explicit "
                "definition/entity sections in the knowledge base. "
                "Use them exactly and do not invent alternative "
                "expansions."
            )

        # --------------------------------------------------------
        # AMF disambiguation.
        # --------------------------------------------------------

        if has_authentication_management_field:

            sections.append(
                "IMPORTANT AMF DISAMBIGUATION\n"
                'The retrieved authentication procedure contains '
                'the phrase "Authentication Management Field (AMF)". '
                "In that phrase, AMF is an authentication parameter "
                "and NOT the Access and Mobility Management Function. "
                "Do not classify that occurrence as a network "
                "function."
            )

        # --------------------------------------------------------
        # Authentication values / keys / parameters.
        # --------------------------------------------------------

        if authentication_values:

            sections.append(
                "AUTHENTICATION VALUES / KEYS / PARAMETERS "
                "PRESENT IN THE EVIDENCE\n"
                + ", ".join(
                    authentication_values
                )
                + "\n\n"
                "These items must NOT be classified as network "
                "functions merely because they appear in the "
                "authentication procedure."
            )

        # --------------------------------------------------------
        # Retrieved evidence.
        # --------------------------------------------------------

        evidence_blocks: List[str] = []

        for index, result in enumerate(
            selected_results,
            start=1,
        ):

            chunk = result.chunk

            metadata = (
                chunk.metadata
                or {}
            )

            source = (
                metadata.get(
                    "source"
                )
                or chunk.document_id
                or "unknown"
            )

            specification = (
                metadata.get(
                    "specification_number"
                )
            )

            section_number = (
                metadata.get(
                    "section_number"
                )
            )

            section_title = (
                metadata.get(
                    "section_title"
                )
            )

            content = (
                chunk.content
                or ""
            ).strip()

            if not content:
                continue

            excerpt = (
                self._truncate_text(
                    content,
                    max_chars=1600,
                )
            )

            header_parts = [
                f"Evidence {index}",
                f"Source={source}",
            ]

            if specification:

                header_parts.append(
                    f"TS={specification}"
                )

            if section_number:

                header_parts.append(
                    f"Section={section_number}"
                )

            if section_title:

                header_parts.append(
                    f"Title={section_title}"
                )

            evidence_blocks.append(
                "["
                + " | ".join(
                    header_parts
                )
                + "]\n"
                + excerpt
            )

        if evidence_blocks:

            sections.append(
                "RETRIEVED EVIDENCE\n\n"
                + "\n\n".join(
                    evidence_blocks
                )
            )

        if not sections:

            return (
                "No relevant information was retrieved "
                "from the current knowledge base."
            )

        return "\n\n".join(
            sections
        )

    # ============================================================
    # Evidence selection
    # ============================================================

    def _select_evidence_results(
        self,
        query: str,
        results: List[SearchResult],
        max_results: int,
    ) -> List[SearchResult]:
        """
        Rank expanded chunks again for compact evidence
        construction.

        This is deliberately lightweight and query-aware.

        It does NOT replace the main reranker.
        """

        if not results:
            return []

        query_terms = (
            self._extract_query_terms(
                query
            )
        )

        scored: List[
            Tuple[float, int, SearchResult]
        ] = []

        for index, result in enumerate(
            results
        ):

            chunk = result.chunk

            metadata = (
                chunk.metadata
                or {}
            )

            content = (
                chunk.content
                or ""
            ).lower()

            section_title = str(
                metadata.get(
                    "section_title"
                )
                or ""
            ).lower()

            network_functions = {
                str(item).lower()
                for item in (
                    metadata.get(
                        "network_functions"
                    )
                    or []
                )
            }

            score = 0.0

            # ----------------------------------------------------
            # Preserve influence from retrieval ranking.
            # ----------------------------------------------------

            score += float(
                result.score
            )

            # ----------------------------------------------------
            # Query-term overlap
            # ----------------------------------------------------

            for term in query_terms:

                if term in section_title:

                    score += 4.0

                content_count = (
                    content.count(term)
                )

                score += min(
                    content_count,
                    3,
                ) * 1.0

                if term in network_functions:

                    score += 5.0

            # ----------------------------------------------------
            # Prefer substantive chunks over heading-only chunks.
            # ----------------------------------------------------

            if len(content) > 250:

                score += 1.5

            if self._is_short_heading_chunk(
                chunk
            ):

                score -= 1.0

            scored.append(
                (
                    score,
                    index,
                    result,
                )
            )

        scored.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        selected = [
            item[2]
            for item in scored[
                :max_results
            ]
        ]

        return selected

    # ============================================================
    # Query terms
    # ============================================================

    @classmethod
    def _extract_query_terms(
        cls,
        query: str,
    ) -> Set[str]:

        raw_terms = re.findall(
            r"[A-Za-z0-9*'-]+",
            query.lower(),
        )

        terms: Set[str] = set()

        for term in raw_terms:

            if (
                len(term) < 3
                and term != "5g"
            ):
                continue

            if term in cls.QUERY_STOPWORDS:
                continue

            # ----------------------------------------------------
            # Exclude numeric components of specification
            # identifiers from lexical scoring.
            # ----------------------------------------------------

            if term.isdigit():
                continue

            terms.add(
                term
            )

        return terms

    # ============================================================
    # Metadata entity collection
    # ============================================================

    @staticmethod
    def _collect_network_functions(
        results: List[SearchResult],
    ) -> List[str]:
        """
        Collect network functions identified by the telecom
        metadata extractor.

        Metadata is safer than inferring functions from every
        uppercase token in procedure text.
        """

        seen: Set[str] = set()

        ordered: List[str] = []

        for result in results:

            metadata = (
                result.chunk.metadata
                or {}
            )

            functions = (
                metadata.get(
                    "network_functions"
                )
                or []
            )

            for function in functions:

                function = str(
                    function
                ).strip()

                if not function:
                    continue

                if function in seen:
                    continue

                seen.add(
                    function
                )

                ordered.append(
                    function
                )

        return ordered

    # ============================================================
    # Candidate acronym collection
    # ============================================================

    @classmethod
    def _collect_candidate_acronyms(
        cls,
        query: str,
        results: List[SearchResult],
    ) -> Set[str]:
        """
        Collect possible acronym tokens.

        Important:
        These are NOT automatically classified as network
        functions.

        They are used only to search for canonical definitions.
        """

        acronyms: Set[str] = set()

        # --------------------------------------------------------
        # Query acronyms
        # --------------------------------------------------------

        for match in (
            cls.ACRONYM_PATTERN.findall(
                query
            )
        ):

            acronyms.add(
                match
            )

        # --------------------------------------------------------
        # Evidence acronyms
        # --------------------------------------------------------

        for result in results:

            content = (
                result.chunk.content
                or ""
            )

            for match in (
                cls.ACRONYM_PATTERN.findall(
                    content
                )
            ):

                acronyms.add(
                    match
                )

            metadata = (
                result.chunk.metadata
                or {}
            )

            for function in (
                metadata.get(
                    "network_functions"
                )
                or []
            ):

                value = str(
                    function
                ).strip()

                if value:

                    acronyms.add(
                        value
                    )

        # --------------------------------------------------------
        # Avoid excessive scanning if a large number of uppercase
        # tokens occur in standards text.
        # --------------------------------------------------------

        if len(acronyms) > 80:

            acronyms = set(
                sorted(acronyms)[
                    :80
                ]
            )

        return acronyms

    # ============================================================
    # Canonical definitions
    # ============================================================

    def _find_canonical_definitions(
        self,
        acronyms: Set[str],
        requested_specification: Optional[str],
        retrieved_text: str = "",
    ) -> Dict[str, str]:
        """
        Search only definition-like knowledge-base sections for
        strict canonical acronym definitions.

        Accepted forms include:

            AUSF: AUthentication Server Function

        and:

            Access and Mobility Management Function (AMF)

        Arbitrary sentences are intentionally excluded.
        """

        if not acronyms:
            return {}

        candidates = {
            str(acronym).strip().upper()
            for acronym in acronyms
            if str(acronym).strip()
        }

        candidates -= (
            self.NON_ENTITY_TOKENS
        )

        if not candidates:
            return {}

        all_chunks = list(
            self._rag_service
            .vector_store
            .chunks
        )

        definitions: Dict[str, str] = {}

        # --------------------------------------------------------
        # AMF collision detection.
        #
        # When the retrieved procedure contains Authentication
        # Management Field (AMF), do not inject the network-function
        # AMF expansion into the same compact evidence.
        # --------------------------------------------------------

        ambiguous_amf = (
            re.search(
                r"\bAuthentication\s+"
                r"Management\s+Field\s*"
                r"\(\s*AMF\s*\)",
                retrieved_text,
                flags=re.IGNORECASE,
            )
            is not None
        )

        for chunk in all_chunks:

            metadata = (
                chunk.metadata
                or {}
            )

            specification = (
                metadata.get(
                    "specification_number"
                )
            )

            # ----------------------------------------------------
            # For an explicitly requested specification, definitions
            # must come from that same specification.
            # ----------------------------------------------------

            if (
                requested_specification
                and specification
                and specification
                != requested_specification
            ):
                continue

            section_title = str(
                metadata.get(
                    "section_title"
                )
                or ""
            ).strip()

            if not self._is_definition_section(
                section_title
            ):
                continue

            content = (
                chunk.content
                or ""
            )

            if not content:
                continue

            for acronym in candidates:

                if (
                    acronym == "AMF"
                    and ambiguous_amf
                ):
                    continue

                if acronym in definitions:
                    continue

                definition = (
                    self._extract_definition_for_acronym(
                        acronym=acronym,
                        text=content,
                    )
                )

                if definition:

                    definitions[
                        acronym
                    ] = definition

        return definitions

    # ------------------------------------------------------------

    def _extract_definition_for_acronym(
        self,
        acronym: str,
        text: str,
    ) -> Optional[str]:
        """
        Extract one strict acronym definition from definition-like
        text.
        """

        if not acronym:
            return None

        if not text:
            return None

        escaped = re.escape(
            acronym
        )

        # --------------------------------------------------------
        # Form 1:
        #
        # AUSF: AUthentication Server Function
        # --------------------------------------------------------

        colon_pattern = re.compile(
            rf"(?im)^\s*"
            rf"{escaped}"
            rf"\s*:\s*"
            rf"(?P<definition>"
            rf"[^\n.;]{{3,200}}"
            rf")"
        )

        match = colon_pattern.search(
            text
        )

        if match:

            definition = (
                match.group(
                    "definition"
                )
                .strip(
                    " .,-"
                )
            )

            if self._valid_definition_text(
                definition
            ):

                return definition

        # --------------------------------------------------------
        # Form 2:
        #
        # Authentication Server Function (AUSF)
        # --------------------------------------------------------

        parenthesis_pattern = re.compile(
            rf"(?im)"
            rf"(?P<definition>"
            rf"[A-Za-z]"
            rf"[A-Za-z0-9 /&,\-]{{3,120}}"
            rf")"
            rf"\s*\(\s*"
            rf"{escaped}"
            rf"\s*\)"
        )

        for match in (
            parenthesis_pattern.finditer(
                text
            )
        ):

            definition = (
                match.group(
                    "definition"
                )
                .strip(
                    " .,-:"
                )
            )

            # Keep the nearest phrase only.
            if "\n" in definition:

                definition = (
                    definition
                    .splitlines()[-1]
                    .strip()
                )

            if self._valid_definition_text(
                definition
            ):

                return definition

        return None

    # ------------------------------------------------------------

    @staticmethod
    def _valid_definition_text(
        text: str,
    ) -> bool:
        """
        Reject obvious false canonical definitions.
        """

        value = (
            text
            or ""
        ).strip()

        if len(value) < 3:
            return False

        if len(value) > 200:
            return False

        # Canonical expansions should remain phrase-like.
        if len(
            value.split()
        ) > 15:
            return False

        lower = value.lower()

        rejected_prefixes = (
            "if ",
            "when ",
            "where ",
            "note ",
            "note:",
            "the response",
            "the request",
            "stage ",
            "clause ",
            "section ",
            "requirements for ",
        )

        if lower.startswith(
            rejected_prefixes
        ):

            return False

        # Reject obvious message/API/result fragments.
        rejected_fragments = (
            "_request",
            "_response",
            "request message",
            "response message",
            "shall ",
            "will ",
            "can ",
            "may ",
            "specified in",
        )

        if any(
            fragment in lower
            for fragment
            in rejected_fragments
        ):

            return False

        return True

    # ------------------------------------------------------------

    def _is_definition_section(
        self,
        title: Optional[str],
    ) -> bool:
        """
        Return True only for explicit definition/entity sections.
        """

        if not title:
            return False

        normalized = (
            str(title)
            .strip()
            .lower()
        )

        return any(
            keyword in normalized
            for keyword
            in self.DEFINITION_SECTION_KEYWORDS
        )

    # ============================================================
    # Context text truncation
    # ============================================================

    @staticmethod
    def _truncate_text(
        text: str,
        max_chars: int,
    ) -> str:

        text = (
            " ".join(
                text.split()
            )
        ).strip()

        if len(text) <= max_chars:

            return text

        truncated = (
            text[:max_chars]
            .rsplit(
                " ",
                1,
            )[0]
        )

        return (
            truncated
            + " ..."
        )

    # ============================================================
    # Heading-only detection
    # ============================================================

    @staticmethod
    def _is_short_heading_chunk(
        chunk: Chunk,
    ) -> bool:

        content = (
            chunk.content
            or ""
        ).strip()

        if not content:
            return True

        if len(content) <= 220:
            return True

        metadata = (
            chunk.metadata
            or {}
        )

        section_number = (
            metadata.get(
                "section_number"
            )
        )

        section_title = (
            metadata.get(
                "section_title"
            )
        )

        if (
            section_number
            and section_title
        ):

            expected_heading = (
                f"{section_number} "
                f"{section_title}"
            )

            normalized_content = (
                " ".join(
                    content.split()
                )
                .lower()
            )

            normalized_heading = (
                " ".join(
                    expected_heading.split()
                )
                .lower()
            )

            if (
                normalized_content
                == normalized_heading
            ):

                return True

        return False

    # ============================================================
    # Result helper
    # ============================================================

    @staticmethod
    def _append_unique_result(
        expanded: List[SearchResult],
        seen_ids: Set[str],
        chunk: Chunk,
        score: float,
    ) -> None:

        if chunk.id in seen_ids:
            return

        seen_ids.add(
            chunk.id
        )

        expanded.append(
            SearchResult(
                chunk=chunk,
                score=float(score),
            )
        )

    # ============================================================
    # Debugging
    # ============================================================

    @staticmethod
    def _print_results(
        title: str,
        results: List[SearchResult],
    ) -> None:

        print()
        print("=" * 80)
        print(title)
        print("=" * 80)

        if not results:

            print(
                "No results."
            )

            return

        for index, result in enumerate(
            results,
            start=1,
        ):

            chunk = result.chunk

            metadata = (
                chunk.metadata
                or {}
            )

            print()
            print(
                f"[Result {index}]"
            )

            print(
                "Score          :",
                f"{result.score:.6f}",
            )

            print(
                "Chunk ID       :",
                chunk.id,
            )

            print(
                "Document ID    :",
                chunk.document_id,
            )

            print(
                "Telecom Aware  :",
                metadata.get(
                    "telecom_aware"
                ),
            )

            print(
                "Source         :",
                metadata.get(
                    "source"
                ),
            )

            print(
                "Specification  :",
                metadata.get(
                    "specification_number"
                ),
            )

            print(
                "Release        :",
                metadata.get(
                    "release"
                ),
            )

            print(
                "Section        :",
                metadata.get(
                    "section_number"
                ),
            )

            print(
                "Section Title  :",
                metadata.get(
                    "section_title"
                ),
            )

            print(
                "Parent Section :",
                metadata.get(
                    "parent_section"
                ),
            )

            print(
                "Network Funcs  :",
                metadata.get(
                    "network_functions"
                ),
            )

            print(
                "Protocols      :",
                metadata.get(
                    "protocols"
                ),
            )

            print(
                "Procedures     :",
                metadata.get(
                    "procedures"
                ),
            )

            print(
                "Content:"
            )

            print(
                chunk.content
            )
