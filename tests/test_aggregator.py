from teleagent.coordinator.aggregator import Aggregator
from teleagent.coordinator.task import Task
from teleagent.coordinator.task_result import TaskResult


def create_result(
    task_id: str,
    intent: str,
    response: str,
    success: bool = True,
):

    task = Task(
        task_id=task_id,
        intent=intent,
        query="Dummy query"
    )

    return TaskResult(
        task=task,
        agent="DummyAgent",
        response=response,
        success=success
    )


def test_single_result():

    aggregator = Aggregator()

    result = create_result(
        "task_001",
        "network",
        "Network issue detected."
    )

    final = aggregator.aggregate([result])

    assert final == "Network issue detected."


def test_multiple_results():

    aggregator = Aggregator()

    results = [
        create_result(
            "task_001",
            "network",
            "Network analysis complete."
        ),
        create_result(
            "task_002",
            "knowledge",
            "PFCP explanation."
        )
    ]

    final = aggregator.aggregate(results)

    assert "Network analysis complete." in final
    assert "PFCP explanation." in final


def test_failed_results_are_ignored():

    aggregator = Aggregator()

    results = [
        create_result(
            "task_001",
            "network",
            "Success."
        ),
        create_result(
            "task_002",
            "log",
            "Failure.",
            success=False
        )
    ]

    final = aggregator.aggregate(results)

    assert final == "Success."


def test_all_failed():

    aggregator = Aggregator()

    results = [
        create_result(
            "task_001",
            "network",
            "Failure 1",
            success=False
        ),
        create_result(
            "task_002",
            "log",
            "Failure 2",
            success=False
        )
    ]

    final = aggregator.aggregate(results)

    assert final == "All tasks failed."


def test_empty_results():

    aggregator = Aggregator()

    final = aggregator.aggregate([])

    assert final == "No results were produced."
