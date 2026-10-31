"""
Task Result

Represents the output generated after executing a Task.
"""

from dataclasses import dataclass, field
from typing import Any

from teleagent.coordinator.task import Task
from teleagent.reasoning.schemas import ReasoningResponse


@dataclass(frozen=True, slots=True)
class TaskResult:
    """
    Result returned by an agent after executing a task.
    """

    task: Task

    agent: str

    response: ReasoningResponse

    success: bool = True

    metadata: dict[str, Any] = field(default_factory=dict)
