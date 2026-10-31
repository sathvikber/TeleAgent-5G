import pytest

from teleagent.coordinator.agent_registry import AgentRegistry
from teleagent.exceptions import AgentNotFoundError


class DummyAgent:
    pass


@pytest.fixture
def registry():
    return AgentRegistry()


def test_register_agent(registry):
    agent = DummyAgent()

    registry.register("network", agent)

    assert registry.has_agent("network")
    assert len(registry) == 1


def test_get_registered_agent(registry):
    agent = DummyAgent()

    registry.register("network", agent)

    returned = registry.get_agent("network")

    assert returned is agent


def test_unregister_agent(registry):
    registry.register("network", DummyAgent())

    registry.unregister("network")

    assert not registry.has_agent("network")
    assert len(registry) == 0


def test_list_agents(registry):
    registry.register("network", DummyAgent())
    registry.register("log", DummyAgent())

    agents = registry.list_agents()

    assert agents == ["log", "network"]


def test_clear_registry(registry):
    registry.register("network", DummyAgent())
    registry.register("log", DummyAgent())

    registry.clear()

    assert len(registry) == 0
    assert registry.list_agents() == []


def test_unknown_agent_raises_exception(registry):
    with pytest.raises(AgentNotFoundError):
        registry.get_agent("knowledge")
