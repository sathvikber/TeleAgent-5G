"""
aggregator.py

Aggregator

Combines the outputs of one or more TaskResults into
a final response.

M4 Enhancement:
- Produces structured multi-agent responses.
- Supports ReasoningResponse objects.
"""

from teleagent.coordinator.task_result import TaskResult
from teleagent.reasoning.schemas import ReasoningResponse


class Aggregator:
    """
    Aggregates TaskResults into a single response.
    """

    def aggregate(
        self,
        results: list[TaskResult],
    ) -> str:
        """
        Aggregate multiple task results.

        Parameters
        ----------
        results : list[TaskResult]

        Returns
        -------
        str
        """

        if not results:
            return "No results were produced."

        successful = [
            result
            for result in results
            if result.success
        ]

        if not successful:
            return "All tasks failed."

        title_map = {
            "NetworkAgent": "Network Analysis",
            "KnowledgeAgent": "Knowledge Response",
            "MonitoringAgent": "Monitoring Analysis",
            "LogAgent": "Log Analysis",
        }

        sections = []

        for result in successful:

            title = title_map.get(
                result.agent,
                result.agent,
            )

            # -----------------------------------------
            # Extract text from ReasoningResponse
            # -----------------------------------------

            if isinstance(result.response, ReasoningResponse):

                response_text = result.response.response

            else:
                response_text = str(result.response)

            section = (
                "=" * 60
                + "\n"
                + title
                + "\n"
                + "=" * 60
                + "\n\n"
                + response_text.strip()
            )

            sections.append(section)

        return "\n\n".join(sections)
