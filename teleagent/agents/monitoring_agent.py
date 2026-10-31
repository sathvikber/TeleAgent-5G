"""
Monitoring Agent

Handles monitoring-related user requests.
"""

from typing import Any, Optional

from teleagent.agents.base_agent import BaseAgent
from teleagent.reasoning.reasoning_service import ReasoningService
from teleagent.reasoning.schemas import ReasoningRequest


class MonitoringAgent(BaseAgent):
    """
    Agent responsible for network monitoring.
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
        Execute a monitoring request using the selected LoRA adapter.
        """

        reasoning_request = ReasoningRequest(
            task="monitoring",
            query=str(request),
            adapter_name=adapter_name,
        )

        return self._reasoning_service.generate(
            reasoning_request
        )
