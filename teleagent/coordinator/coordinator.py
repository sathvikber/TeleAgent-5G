"""
Coordinator

Main orchestration component of the TeleAgent platform.

Coordinates the complete request execution pipeline:
    IntentAnalyzer
        ↓
    TaskPlanner
        ↓
    Dispatcher
        ↓
    Aggregator
"""
from concurrent.futures import ThreadPoolExecutor
from teleagent.coordinator.aggregator import Aggregator
from teleagent.coordinator.dispatcher import Dispatcher
from teleagent.coordinator.intent_analyzer import IntentAnalyzer
from teleagent.coordinator.task_planner import TaskPlanner
from teleagent.coordinator.coordinator_response import (
    CoordinatorResponse,
)

class Coordinator:
    """
    Coordinates execution of user requests.
    """

    def __init__(
        self,
        intent_analyzer: IntentAnalyzer,
        task_planner: TaskPlanner,
        dispatcher: Dispatcher,
        aggregator: Aggregator,
    ) -> None:
        self._intent_analyzer = intent_analyzer
        self._task_planner = task_planner
        self._dispatcher = dispatcher
        self._aggregator = aggregator

    def execute(self, query: str) -> CoordinatorResponse:
        """
        Execute a user query.

        Parameters
        ----------
        query : str
            User request.

        Returns
        -------
        str
            Final response.
        """

        # Step 1
        intent = self._intent_analyzer.analyze(query)

        # Step 2
        tasks = self._task_planner.create_tasks(
            intent=intent,
            query=query,
        )

        # Step 3
        with ThreadPoolExecutor(
            max_workers=len(tasks)
        ) as executor:

            futures = [
                executor.submit(
                    self._dispatcher.dispatch,
                    task
                )
                for task in tasks
            ]

            results = [
                future.result()
                for future in futures
            ]

        # Step 4

        final_response = self._aggregator.aggregate(
            results
        )

        return CoordinatorResponse(
            intent=intent,
            tasks=tasks,
            results=results,
            response=final_response,
        )
