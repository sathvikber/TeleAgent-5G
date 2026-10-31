from teleagent.coordinator.task import Task


def test_task_creation():

    task = Task(
        task_id="task_001",
        intent="network",
        query="UE attach failed"
    )

    assert task.intent == "network"
    assert task.query == "UE attach failed"
    assert task.metadata == {}


def test_task_metadata():

    task = Task(
        task_id="task_001",
        intent="network",
        query="Check AMF",
        metadata={"priority": "high"}
    )

    assert task.metadata["priority"] == "high"


def test_task_is_immutable():

    task = Task(
        task_id="task_001",
        intent="network",
        query="Hello"
    )

    try:
        task.intent = "log"
        assert False
    except AttributeError:
        assert True
