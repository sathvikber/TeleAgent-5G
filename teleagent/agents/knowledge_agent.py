"""
Handles telecom knowledge-related user requests using
Retrieval-Augmented Generation (RAG).
"""

from typing import Any, Optional

import teleagent.config as config

from teleagent.agents.base_agent import BaseAgent

from teleagent.reasoning.reasoning_service import (
    ReasoningService,
)

from teleagent.reasoning.schemas import (
    ReasoningRequest,
)

from teleagent.retrieval.intelligent_retrieval_service import (
    IntelligentRetrievalService,
)


class KnowledgeAgent(BaseAgent):
    """
    Telecom Knowledge Agent backed by Hybrid RAG.
    """

    def __init__(
        self,
        reasoning_service: ReasoningService,
        retrieval_service: IntelligentRetrievalService,
    ):

        self._reasoning_service = reasoning_service

        self._retrieval = retrieval_service

    def execute(
        self,
        request: Any,
        adapter_name: Optional[str] = None,
    ) -> Any:
        """
        Answer telecom knowledge questions using RAG
        and the selected LoRA adapter.
        """

        query = str(request)

        # Build the complete RAG prompt.
        prompt = self._retrieval.build_prompt(
            query=query,
            top_k=config.RAG_TOP_K,
        )

        # Send the prompt and selected adapter to the SLM.
        reasoning_request = ReasoningRequest(
            task="knowledge",
            query=prompt,
            adapter_name=adapter_name,
        )

        return self._reasoning_service.generate(
            reasoning_request
        )
