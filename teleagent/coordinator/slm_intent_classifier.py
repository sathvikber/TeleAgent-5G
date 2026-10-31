"""
slm_intent_classifier.py

Telecom SLM based intent classifier.

Uses the Core Reasoning Service to classify a user query
into one of the supported TeleAgent intents.

Author : TeleAgent-5G
Version : 0.3
"""

from __future__ import annotations

import json

from teleagent.config import DEFAULT_ADAPTER

from teleagent.reasoning.reasoning_service import ReasoningService
from teleagent.reasoning.schemas import ReasoningRequest

from teleagent.coordinator.intent_classifier import IntentClassifier


class SLMIntentClassifier(IntentClassifier):
    """
    Telecom SLM based intent classifier.
    """

    ALLOWED_INTENTS = {
        "network",
        "log",
        "knowledge",
        "monitoring",
    }

    def __init__(
        self,
        reasoning_service: ReasoningService,
    ) -> None:
        self._reasoning_service = reasoning_service

    def classify(
        self,
        query: str,
    ) -> str:
        """
        Classify a user query using the default LoRA adapter.
        """

        request = ReasoningRequest(
            task="intent_classification",
            query=query,
            adapter_name=DEFAULT_ADAPTER,
        )

        response = self._reasoning_service.generate(
            request
        )

        return self._parse_response(
            response.response
        )

    @classmethod
    def _parse_response(
        cls,
        response: str,
    ) -> str:
        """
        Parse the model response.
        """

        try:
            result = json.loads(response)

            intent = result["intent"].strip().lower()

            if intent in cls.ALLOWED_INTENTS:
                return intent

        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
        ):
            pass

        return "knowledge"
