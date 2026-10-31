from teleagent.coordinator.task import Task
from teleagent.coordinator.task_result import TaskResult


def test_task_result_creation():

    task = Task(
        task_id="task_001",
        intent="network",
        query="UE attach failed"
    )

    result = TaskResult(
        task=task,
        agent="NetworkAgent",
        response="UE registration successful."
    )

    assert result.task == task
    assert result.agent == "NetworkAgent"
    assert result.response == "UE registration successful."
    assert result.success is True
    assert result.metadata == {}


def test_task_result_failure():

    task = Task(
        task_id="task_001",
        intent="log",
        query="Show AMF logs"
    )

    result = TaskResult(
        task=task,
        agent="LogAgent",
        response="Log server unavailable.",
        success=False
    )

    assert result.success is False


def test_task_result_metadata():

    task = Task(
        task_id="task_001",
        intent="monitoring",
        query="Show CPU utilization"
    )

    result = TaskResult(
        task=task,
        agent="MonitoringAgent",
        response="CPU utilization is 35%",
        metadata={"latency": 0.25}
    )

    assert result.metadata["latency"] == 0.25


def test_task_result_is_immutable():

    task = Task(
        task_id="task_001",
        intent="network",
        query="Hello"
    )

    result = TaskResult(
        task=task,
        agent="NetworkAgent",
        response="Hello"
    )

    try:
        result.response = "Modified"
        assert False
    except AttributeError:
        assert True
