"""
Base Agent

Defines the interface that every TeleAgent agent must implement.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """
    Abstract base class for all TeleAgent agents.
    """

    @abstractmethod
    def execute(self, request: Any) -> Any:
        """
        Execute the agent.

        Parameters
        ----------
        request : Any
            User request.

        Returns
        -------
        Any
            Agent response.
        """
        raise NotImplementedError
