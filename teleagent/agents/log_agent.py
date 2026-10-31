"""
Log Agent

Handles telecom log analysis requests.
"""

from typing import Any, Optional

from teleagent.agents.base_agent import BaseAgent
from teleagent.reasoning.reasoning_service import ReasoningService
from teleagent.reasoning.schemas import ReasoningRequest


class LogAgent(BaseAgent):
    """
    Agent responsible for telecom log analysis.
    """

    def __init__(
        self,
        reasoning_service: ReasoningService,
    ):
        self._reasoning_service = reasoning_service

    def execute(
        self,
        request: Any,
        adapter_name: Optional[str] = None,
    ) -> Any:
        """
        Execute a log analysis request using the selected LoRA adapter.
        """

        reasoning_request = ReasoningRequest(
            task="log",
            query=str(request),
            adapter_name=adapter_name,
        )

        return self._reasoning_service.generate(
            reasoning_request
        )
