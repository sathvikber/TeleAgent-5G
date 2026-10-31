"""
Agent Registry

Maintains a registry of all available agents in the TeleAgent platform.
The Coordinator uses this registry to retrieve the appropriate agent
based on the detected intent.
"""

from typing import Any

from teleagent.exceptions import AgentNotFoundError


class AgentRegistry:
    """
    Registry for all available TeleAgent agents.

    The registry maps an intent name (e.g., 'network', 'log')
    to an instantiated agent object.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._agents: dict[str, Any] = {}

    def register(self, name: str, agent: Any) -> None:
        """
        Register an agent.

        Parameters
        ----------
        name : str
            Unique intent name.
        agent : Any
            Agent instance.
        """
        self._agents[name] = agent

    def unregister(self, name: str) -> None:
        """
        Remove an agent from the registry.

        Parameters
        ----------
        name : str
            Agent name.
        """
        self._agents.pop(name, None)

    def get_agent(self, name: str) -> Any:
        """
        Retrieve an agent.

        Parameters
        ----------
        name : str
            Agent name.

        Returns
        -------
        Any
            Registered agent instance.

        Raises
        ------
        AgentNotFoundError
            If the requested agent is not registered.
        """
        try:
            return self._agents[name]
        except KeyError as exc:
            raise AgentNotFoundError(
                f"Agent '{name}' is not registered."
            ) from exc

    def has_agent(self, name: str) -> bool:
        """
        Check whether an agent exists.

        Parameters
        ----------
        name : str
            Agent name.

        Returns
        -------
        bool
        """
        return name in self._agents

    def list_agents(self) -> list[str]:
        """
        Return a list of registered agent names.
        """
        return sorted(self._agents.keys())

    def clear(self) -> None:
        """
        Remove all registered agents.
        """
        self._agents.clear()

    def __len__(self) -> int:
        """
        Return the number of registered agents.
        """
        return len(self._agents)
