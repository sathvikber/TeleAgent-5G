"""
coordinator_response.py

Final response returned by the Coordinator.
"""

from pydantic import BaseModel

from teleagent.coordinator.task import Task
from teleagent.coordinator.task_result import TaskResult


class CoordinatorResponse(BaseModel):
    """
    Response returned by the Coordinator.
    """

    intent: str

    tasks: list[Task]

    results: list[TaskResult]

    response: str
