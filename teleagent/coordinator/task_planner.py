"""
task_planner.py

Task Planner

Creates one or more tasks from a user query.

M4 Enhancement:
- Supports multi-agent collaboration.
- Generates multiple tasks when a query contains
  evidence for multiple intents.

Author : TeleAgent-5G
Version : 0.2
"""

from teleagent.coordinator.task import Task


class TaskPlanner:
    """
    Creates executable tasks from a user query.
    """

    LOG_KEYWORDS = {
        "log",
        "logs",
        "trace",
        "debug",
        "warning",
        "error",
        "stack",
    }

    MONITORING_KEYWORDS = {
        "cpu",
        "memory",
        "disk",
        "latency",
        "throughput",
        "utilization",
        "usage",
        "health",
        "alarm",
        "kpi",
        "signal",
        "rssi",
        "snr",
    }

    NETWORK_KEYWORDS = {
        "registration",
        "handover",
        "attach",
        "session",
        "amf",
        "smf",
        "upf",
        "gnodeb",
        "rrc",
        "ngap",
        "pfcp",
        "ue",
        "network",
    }

    def create_tasks(
        self,
        intent: str,
        query: str,
    ) -> list[Task]:
        """
        Create one or more executable tasks.
        """

        tasks: list[Task] = []

        added: set[str] = set()

        task_counter = 1

        def add_task(task_intent: str) -> None:
            nonlocal task_counter

            if task_intent in added:
                return

            tasks.append(
                Task(
                    task_id=f"task-{task_counter}",
                    intent=task_intent,
                    query=query,
                    metadata={},
                )
            )

            added.add(task_intent)
            task_counter += 1

        # -------------------------------------------------
        # Primary task
        # -------------------------------------------------

        add_task(intent)

        query_lower = query.lower()

        # -------------------------------------------------
        # Additional tasks
        # -------------------------------------------------

        if any(word in query_lower for word in self.LOG_KEYWORDS):
            add_task("log")

        if any(word in query_lower for word in self.MONITORING_KEYWORDS):
            add_task("monitoring")

        if any(word in query_lower for word in self.NETWORK_KEYWORDS):
            add_task("network")

        return tasks
