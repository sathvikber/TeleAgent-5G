from teleagent.coordinator.task_planner import TaskPlanner


def test_create_single_task():

    planner = TaskPlanner()

    tasks = planner.create_tasks(
        intent="network",
        query="Why is UE attach failing?"
    )

    assert len(tasks) == 1

    task = tasks[0]

    assert task.task_id == "task_001"
    assert task.intent == "network"
    assert task.query == "Why is UE attach failing?"


def test_task_ids_are_unique():

    planner = TaskPlanner()

    first = planner.create_tasks(
        "network",
        "First query"
    )[0]

    second = planner.create_tasks(
        "log",
        "Second query"
    )[0]

    assert first.task_id != second.task_id


def test_returns_task_list():

    planner = TaskPlanner()

    tasks = planner.create_tasks(
        "knowledge",
        "Explain PFCP"
    )

    assert isinstance(tasks, list)
    assert len(tasks) == 1
