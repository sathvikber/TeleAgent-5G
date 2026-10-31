"""
Network Analysis Service

Coordinates the complete deterministic network analysis pipeline.

Pipeline

Issue Classifier
        ↓
Network Analyzer
        ↓
Recommendation Engine
        ↓
Prompt Builder
"""

from typing import Optional, Dict, Any

from teleagent.network.issue_classifier import IssueClassifier
from teleagent.network.network_analyzer import NetworkAnalyzer
from teleagent.network.recommendation_engine import RecommendationEngine
from teleagent.network.network_prompt_builder import NetworkPromptBuilder
from teleagent.network.models import NetworkAnalysisResult


class NetworkAnalysisService:
    """
    Coordinates the network analysis pipeline.
    """

    def __init__(self):

        self._classifier = IssueClassifier()

        self._analyzer = NetworkAnalyzer()

        self._recommendation_engine = RecommendationEngine()

        self._prompt_builder = NetworkPromptBuilder()

    def analyze(
        self,
        query: str,
        topology_context: Optional[Dict[str, Any]] = None,
        diagnostic_evidence: Optional[Dict[str, Any]] = None,
    ) -> NetworkAnalysisResult:
        """
        Analyze a network troubleshooting request.
        """

        classification = self._classifier.classify(query)

        analysis = self._analyzer.analyze(classification)

        troubleshooting_steps = self._recommendation_engine.generate(analysis)

        return NetworkAnalysisResult(
            classification=classification,
            analysis=analysis,
            troubleshooting_steps=troubleshooting_steps,
            topology_context=topology_context,
            diagnostic_evidence=diagnostic_evidence,
        )

    def build_prompt(
        self,
        user_query: str,
        analysis: NetworkAnalysisResult,
    ) -> str:
        """
        Build the final prompt for the Telecom SLM.
        """

        return self._prompt_builder.build(
            user_query=user_query,
            analysis=analysis,
        )