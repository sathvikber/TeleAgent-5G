import pytest

from teleagent.coordinator.agent_registry import AgentRegistry
from teleagent.coordinator.dispatcher import Dispatcher
from teleagent.coordinator.task import Task
from teleagent.coordinator.task_result import TaskResult
from teleagent.exceptions import AgentNotFoundError


class DummyAgent:
    """Dummy agent used for testing."""

    def execute(self, request):
        return f"Processed: {request}"


@pytest.fixture
def dispatcher():

    registry = AgentRegistry()
    registry.register("network", DummyAgent())

    return Dispatcher(registry)


def test_dispatch_to_registered_agent(dispatcher):

    task = Task(
        task_id="001",
        intent="network",
        query="Why is UE attach failing?"
    )

    result = dispatcher.dispatch(task)

    assert isinstance(result, TaskResult)

    assert result.task == task
    assert result.agent == "DummyAgent"

    assert result.response == "Processed: Why is UE attach failing?"

    assert result.success is True

    assert result.metadata == {}


def test_dispatch_unknown_agent(dispatcher):

    task = Task(
        task_id="task_001",
        intent="unknown",
        query="Hello"
    )

    with pytest.raises(AgentNotFoundError):
        dispatcher.dispatch(task)
