"""
Dispatcher

Routes a Task to the appropriate agent using the AgentRegistry
and selects the corresponding LoRA adapter.

Author : TeleAgent-5G
Version : 0.3
"""

from teleagent.config import (
    AGENT_ADAPTER_MAP,
    ADAPTER_PATHS,
    DEFAULT_ADAPTER,
)

from teleagent.coordinator.agent_registry import AgentRegistry
from teleagent.coordinator.task import Task
from teleagent.coordinator.task_result import TaskResult


class Dispatcher:
    """
    Dispatches tasks to registered agents and selects
    the appropriate LoRA adapter.
    """

    def __init__(self, registry: AgentRegistry) -> None:
        """
        Parameters
        ----------
        registry : AgentRegistry
            Registry containing available agents.
        """
        self._registry = registry

    def _select_adapter(self, task: Task) -> str:
        """
        Select a LoRA adapter using the task intent and query.
        """

        query = str(task.query).lower()
        intent = str(task.intent).lower()

        # Specialized adapter selection for beamforming.
        beam_keywords = (
            "beamforming",
            "beam forming",
            "beam management",
            "beam steering",
            "beam failure",
            "massive mimo",
        )

        if (
            "adapter_beam" in ADAPTER_PATHS
            and any(keyword in query for keyword in beam_keywords)
        ):
            return "adapter_beam"

        # Specialized adapter selection for waveform and OFDM topics.
        wave_keywords = (
            "waveform",
            "ofdm",
            "ofdma",
            "modulation",
            "demodulation",
            "cyclic prefix",
            "subcarrier",
        )

        if (
            "adapter_wave" in ADAPTER_PATHS
            and any(keyword in query for keyword in wave_keywords)
        ):
            return "adapter_wave"

        # Use the adapter mapped to the task's agent intent.
        adapter_name = AGENT_ADAPTER_MAP.get(
            intent,
            DEFAULT_ADAPTER,
        )

        # Fall back safely if the configured adapter is unavailable.
        if adapter_name not in ADAPTER_PATHS:
            adapter_name = DEFAULT_ADAPTER

        return adapter_name

    def dispatch(self, task: Task) -> TaskResult:
        """
        Dispatch a task to the appropriate agent.
        """

        agent = self._registry.get_agent(task.intent)

        adapter_name = self._select_adapter(task)

        # Pass the selected adapter to agents that support it.
        # Agent execute methods must accept adapter_name and pass it
        # to their ReasoningRequest.
        response = agent.execute(
            task.query,
            adapter_name=adapter_name,
        )

        return TaskResult(
            task=task,
            agent=agent.__class__.__name__,
            response=response,
            success=True,
            metadata={
                "adapter_name": adapter_name,
            },
        )
