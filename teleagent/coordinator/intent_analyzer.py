"""
Intent Analyzer

Hybrid intent routing for TeleAgent.

Routing strategy
----------------
1. Apply strong deterministic rules for clear user intent.
2. Avoid using telecom entities such as AMF, SMF, UE, gNB,
   PFCP, NGAP, etc. as intent indicators by themselves.
3. Route explicit standards-based informational questions
   to the KnowledgeAgent.
4. Use the SLM IntentClassifier only for ambiguous queries.
5. Fall back safely to knowledge when classification fails.

Supported intents
-----------------
- network
- log
- knowledge
- monitoring
"""

from __future__ import annotations

import re
from typing import Optional

from .intent_classifier import IntentClassifier


class IntentAnalyzer:
    """
    Hybrid intent analyzer for TeleAgent.

    Strong deterministic rules handle obvious cases.
    Ambiguous queries are delegated to an optional IntentClassifier,
    normally SLMIntentClassifier.
    """

    ALLOWED_INTENTS = {
        "network",
        "log",
        "knowledge",
        "monitoring",
    }

    # ============================================================
    # Explicit 3GPP specification reference
    #
    # Examples:
    #
    # TS 23.502
    # TS23.502
    # 3GPP TS 33.501
    # TR 28.805
    # ============================================================

    STANDARD_PATTERN = re.compile(
        r"\b"
        r"(?:3gpp\s+)?"
        r"(?:ts|tr)"
        r"\s*"
        r"\d{2}\.\d{3}"
        r"\b",
        re.IGNORECASE,
    )

    # ============================================================
    # Log indicators
    #
    # These deliberately focus on explicit log/trace artifacts.
    #
    # Generic words such as "error" or "failure" are NOT enough
    # by themselves because:
    #
    #     "UE registration failure"
    #
    # should normally be a network troubleshooting request rather
    # than automatically becoming a LogAgent request.
    # ============================================================

    LOG_TERMS = (
        "log",
        "logs",
        "log file",
        "log files",
        "trace",
        "traces",
        "trace file",
        "event log",
        "event logs",
        "diagnostic log",
        "diagnostic logs",
        "journal",
        "journalctl",
    )

    # ============================================================
    # Monitoring indicators
    # ============================================================

    MONITORING_ACTIONS = (
        "monitor",
        "monitoring",
        "show",
        "display",
        "measure",
        "check",
        "report",
        "get current",
        "show current",
    )

    MONITORING_METRICS = (
        "throughput",
        "uplink throughput",
        "downlink throughput",
        "latency",
        "jitter",
        "packet loss",
        "packet-loss",
        "pdr",
        "rssi",
        "snr",
        "cpu",
        "cpu usage",
        "cpu utilization",
        "memory",
        "memory usage",
        "memory utilization",
        "disk",
        "disk usage",
        "utilization",
        "usage",
        "kpi",
        "kpis",
        "health",
        "status",
        "active ue",
        "active ues",
        "active gnb",
        "active gnbs",
        "network state",
    )

    MONITORING_STATE_TERMS = (
        "current",
        "currently",
        "live",
        "real-time",
        "realtime",
        "right now",
        "at present",
    )

    # ============================================================
    # Knowledge indicators
    # ============================================================

    KNOWLEDGE_PATTERNS = (
        "what is",
        "what are",
        "who is",
        "which",
        "explain",
        "define",
        "describe",
        "meaning of",
        "role of",
        "purpose of",
        "function of",
        "functions of",
        "how does",
        "how do",
        "procedure",
        "architecture",
        "protocol used",
        "which protocol",
        "according to",
        "difference between",
        "relationship between",
    )

    # ============================================================
    # Network troubleshooting indicators
    # ============================================================

    NETWORK_PROBLEM_TERMS = (
        "fail",
        "fails",
        "failed",
        "failure",
        "failing",
        "issue",
        "issues",
        "problem",
        "problems",
        "unable",
        "cannot",
        "can't",
        "not able",
        "not working",
        "timeout",
        "timed out",
        "reject",
        "rejected",
        "drop",
        "dropped",
        "disconnect",
        "disconnected",
        "unstable",
        "instability",
        "degraded",
        "degradation",
        "unreachable",
        "low throughput",
        "high latency",
        "packet loss",
        "authentication failure",
        "registration failure",
        "registration failed",
        "pdu session failure",
        "handover failure",
        "signalling failure",
        "signaling failure",
        "connectivity problem",
        "root cause",
        "troubleshoot",
        "troubleshooting",
    )

    NETWORK_QUESTION_PATTERNS = (
        "why is",
        "why are",
        "why does",
        "why did",
        "why has",
        "why was",
        "what caused",
        "what causes",
        "root cause",
    )

    # ============================================================
    # Constructor
    # ============================================================

    def __init__(
        self,
        classifier: Optional[IntentClassifier] = None,
        confidence_threshold: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        classifier:
            Optional fallback classifier.

            In TeleAgent this is normally SLMIntentClassifier.

        confidence_threshold:
            Retained for backward compatibility with the earlier
            score-based IntentAnalyzer interface.

            Strong deterministic rules now take precedence, so this
            value is not used for entity-keyword scoring.
        """

        self._classifier = classifier

        # Retain existing constructor/API compatibility.
        self._confidence_threshold = confidence_threshold

    # ============================================================
    # Public API
    # ============================================================

    def analyze(
        self,
        query: str,
    ) -> str:
        """
        Determine the most appropriate TeleAgent intent.

        Returns one of:

            network
            log
            knowledge
            monitoring
        """

        if not query:
            return "knowledge"

        query = query.strip()

        if not query:
            return "knowledge"

        query_lower = self._normalize(
            query
        )

        # --------------------------------------------------------
        # 1. Explicit log request
        #
        # Examples:
        #
        # "check AMF logs"
        # "analyze UE registration logs"
        # "inspect NGAP traces"
        #
        # Explicit log artifacts have the highest precedence.
        # --------------------------------------------------------

        if self._is_log_query(
            query_lower
        ):
            return "log"

        # --------------------------------------------------------
        # 2. Explicit monitoring request
        #
        # Examples:
        #
        # "monitor gNB throughput"
        # "show UE throughput"
        # "show current CPU utilization"
        # "monitor the network"
        #
        # This rule intentionally runs before general knowledge
        # detection.
        #
        # Therefore:
        #
        #     "What is throughput?"
        #         -> knowledge
        #
        # but:
        #
        #     "Show current throughput"
        #         -> monitoring
        # --------------------------------------------------------

        if self._is_monitoring_query(
            query_lower
        ):
            return "monitoring"

        # --------------------------------------------------------
        # 3. Explicit telecom-standard knowledge question
        #
        # This is the new high-priority rule required for queries
        # such as:
        #
        # "According to TS 23.502, which network functions are
        #  involved in UE authentication?"
        #
        # Previously such a query could reach the SLM classifier
        # and occasionally be misclassified as monitoring.
        # --------------------------------------------------------

        if self._is_standards_knowledge_query(
            query_lower
        ):
            return "knowledge"

        # --------------------------------------------------------
        # 4. General knowledge / explanation request
        #
        # Examples:
        #
        # "What is AMF?"
        # "Explain NGAP"
        # "Role of SMF"
        # "Describe UE registration procedure"
        # --------------------------------------------------------

        if self._is_knowledge_query(
            query_lower
        ):
            return "knowledge"

        # --------------------------------------------------------
        # 5. Network troubleshooting request
        #
        # Examples:
        #
        # "UE registration failed"
        # "Why is throughput low?"
        # "Authentication failure during registration"
        # --------------------------------------------------------

        if self._is_network_query(
            query_lower
        ):
            return "network"

        # --------------------------------------------------------
        # 6. Ambiguous query -> SLM classifier
        #
        # Examples:
        #
        # "AMF CPU"
        # "NGAP signalling"
        # "UE state"
        #
        # Let the classifier resolve these rather than forcing
        # intent based on telecom entity names.
        # --------------------------------------------------------

        classifier_intent = (
            self._classify_with_fallback(
                query
            )
        )

        if classifier_intent:
            return classifier_intent

        # --------------------------------------------------------
        # 7. Safe default
        # --------------------------------------------------------

        return "knowledge"

    # ============================================================
    # Standards knowledge detection
    # ============================================================

    def _is_standards_knowledge_query(
        self,
        query: str,
    ) -> bool:
        """
        Detect informational questions that explicitly reference
        a 3GPP TS/TR specification.

        Examples
        --------
        According to TS 23.502, which network functions are
        involved in UE authentication?

        What does TS 33.501 specify for authentication?

        Explain the AMF role according to TS 23.501.
        """

        if not self.STANDARD_PATTERN.search(
            query
        ):
            return False

        informational_patterns = (
            "according to",
            "what is",
            "what are",
            "which",
            "explain",
            "describe",
            "define",
            "role of",
            "purpose of",
            "function of",
            "functions of",
            "how does",
            "how do",
            "procedure",
            "architecture",
            "specify",
            "specified",
        )

        return any(
            pattern in query
            for pattern in informational_patterns
        )

    # ============================================================
    # Log detection
    # ============================================================

    def _is_log_query(
        self,
        query: str,
    ) -> bool:
        """
        Detect explicit log/trace analysis requests.
        """

        return any(
            self._contains_term(
                query,
                term,
            )
            for term in self.LOG_TERMS
        )

    # ============================================================
    # Monitoring detection
    # ============================================================

    def _is_monitoring_query(
        self,
        query: str,
    ) -> bool:
        """
        Detect requests for live/current network measurements.

        Telecom entities alone do not imply monitoring.
        """

        # --------------------------------------------------------
        # Explicit "monitor ..." command
        # --------------------------------------------------------

        if re.search(
            r"\bmonitor(?:ing)?\b",
            query,
            re.IGNORECASE,
        ):
            return True

        has_action = any(
            self._contains_term(
                query,
                action,
            )
            for action in self.MONITORING_ACTIONS
        )

        has_metric = any(
            self._contains_term(
                query,
                metric,
            )
            for metric in self.MONITORING_METRICS
        )

        has_live_state = any(
            self._contains_term(
                query,
                state,
            )
            for state in self.MONITORING_STATE_TERMS
        )

        # --------------------------------------------------------
        # "show UE throughput"
        # "check CPU utilization"
        # --------------------------------------------------------

        if (
            has_action
            and has_metric
        ):
            return True

        # --------------------------------------------------------
        # "current UE throughput"
        # "live gNB throughput"
        # --------------------------------------------------------

        if (
            has_live_state
            and has_metric
        ):
            return True

        # --------------------------------------------------------
        # Common direct state queries
        # --------------------------------------------------------

        direct_state_patterns = (
            "network status",
            "network health",
            "current status",
            "current health",
            "system status",
        )

        if any(
            pattern in query
            for pattern in direct_state_patterns
        ):
            return True

        return False

    # ============================================================
    # Knowledge detection
    # ============================================================

    def _is_knowledge_query(
        self,
        query: str,
    ) -> bool:
        """
        Detect conceptual / informational telecom questions.
        """

        return any(
            pattern in query
            for pattern in self.KNOWLEDGE_PATTERNS
        )

    # ============================================================
    # Network troubleshooting detection
    # ============================================================

    def _is_network_query(
        self,
        query: str,
    ) -> bool:
        """
        Detect operational network troubleshooting requests.
        """

        if any(
            pattern in query
            for pattern in self.NETWORK_QUESTION_PATTERNS
        ):
            return True

        return any(
            self._contains_term(
                query,
                term,
            )
            for term in self.NETWORK_PROBLEM_TERMS
        )

    # ============================================================
    # SLM fallback
    # ============================================================

    def _classify_with_fallback(
        self,
        query: str,
    ) -> Optional[str]:
        """
        Use the optional IntentClassifier only when deterministic
        rules cannot confidently determine intent.

        Invalid classifier responses are ignored and the caller
        falls back to knowledge.
        """

        if self._classifier is None:
            return None

        try:

            intent = self._classifier.classify(
                query
            )

        except Exception:

            # Intent classification should never crash the
            # Coordinator. Use the safe fallback instead.
            return None

        if not isinstance(
            intent,
            str,
        ):
            return None

        intent = (
            intent
            .strip()
            .lower()
        )

        if intent not in self.ALLOWED_INTENTS:
            return None

        return intent

    # ============================================================
    # Helpers
    # ============================================================

    @staticmethod
    def _normalize(
        query: str,
    ) -> str:
        """
        Normalize whitespace and case.
        """

        return " ".join(
            query.lower().split()
        )

    @staticmethod
    def _contains_term(
        query: str,
        term: str,
    ) -> bool:
        """
        Match a complete term conservatively.

        This avoids accidental substring matches where possible.
        """

        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(term.lower())
            + r"(?![a-z0-9])"
        )

        return (
            re.search(
                pattern,
                query,
                re.IGNORECASE,
            )
            is not None
        )
