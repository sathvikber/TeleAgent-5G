"""
reasoning_service.py

Core Reasoning Service (CRS)

This module orchestrates the complete reasoning pipeline.

Pipeline
--------
ReasoningRequest
        │
        ▼
PromptBuilder
        │
        ▼
InferenceEngine
        │
        ▼
ReasoningResponse

Author : TeleAgent-5G
Version : 0.2
"""

import json

from teleagent.reasoning.schemas import (
    ReasoningRequest,
    ReasoningResponse,
)

from teleagent.reasoning.prompt_builder import PromptBuilder
from teleagent.reasoning.inference_engine import InferenceEngine
from teleagent.reasoning.model_manager import ModelManager


class ReasoningService:
    """
    Facade for the Telecom Small Language Model.

    This class exposes a single interface for performing
    reasoning tasks. It coordinates prompt construction
    and inference while hiding the internal implementation.
    """

    def __init__(self, model_manager: ModelManager):

        self.model_manager = model_manager

        self.prompt_builder = PromptBuilder()

        self.inference_engine = InferenceEngine(
            model_manager
        )

    # ---------------------------------------------------------
    # Generate
    # ---------------------------------------------------------

    def generate(
        self,
        request: ReasoningRequest
    ) -> ReasoningResponse:
        """
        Execute the complete reasoning pipeline.

        Parameters
        ----------
        request : ReasoningRequest

        Returns
        -------
        ReasoningResponse
        """

        messages = self.prompt_builder.build(
            request
        )

        response = self.inference_engine.generate(
            request=request,
            messages=messages,
        )

        return response

    # ---------------------------------------------------------
    # Generate JSON
    # ---------------------------------------------------------

    def generate_json(
        self,
        request: ReasoningRequest,
    ) -> dict:
        """
        Execute reasoning and parse the generated response as JSON.

        Parameters
        ----------
        request : ReasoningRequest

        Returns
        -------
        dict
            Parsed JSON. Empty dictionary is returned if parsing fails.
        """

        reasoning_response = self.generate(request)

        try:
            return json.loads(reasoning_response.response)

        except json.JSONDecodeError:
            return {}
