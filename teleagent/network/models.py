"""
Network Data Models

Shared data models used by the NetworkAgent pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


# ------------------------------------------------------------------
# Issue Classification
# ------------------------------------------------------------------

@dataclass
class IssueClassification:
    """
    Result of classifying a network troubleshooting query.
    """

    category: str
    confidence: float = 1.0


# ------------------------------------------------------------------
# Troubleshooting Step
# ------------------------------------------------------------------

@dataclass
class TroubleshootingStep:
    """
    Represents a recommended troubleshooting action.
    """

    step_number: int
    description: str


# ------------------------------------------------------------------
# Network Analysis
# ------------------------------------------------------------------

@dataclass
class NetworkAnalysis:
    """
    Structured information extracted from the user query.
    """

    affected_network_functions: list[str] = field(default_factory=list)

    interfaces: list[str] = field(default_factory=list)

    protocols: list[str] = field(default_factory=list)

    possible_causes: list[str] = field(default_factory=list)


# ------------------------------------------------------------------
# Network Analysis Result
# ------------------------------------------------------------------

@dataclass
class NetworkAnalysisResult:
    """
    Output of the Network Analysis Service.
    """

    classification: IssueClassification

    analysis: NetworkAnalysis

    troubleshooting_steps: list[TroubleshootingStep] = field(default_factory=list)

    # Optional extension fields with default None (100% Backward Compatible)
    topology_context: Optional[Dict[str, Any]] = None

    diagnostic_evidence: Optional[Dict[str, Any]] = None
