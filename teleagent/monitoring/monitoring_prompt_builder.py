"""
Monitoring Prompt Builder

Builds a structured prompt for the Telecom SLM using the
results of the monitoring analysis pipeline.
"""

from teleagent.monitoring.models import MonitoringAnalysisResult


class MonitoringPromptBuilder:
    """
    Builds prompts for monitoring analysis.
    """

    def build(
        self,
        user_query: str,
        analysis: MonitoringAnalysisResult,
    ) -> str:
        """
        Build a structured prompt.

        Parameters
        ----------
        user_query : str

        analysis : MonitoringAnalysisResult

        Returns
        -------
        str
        """

        prompt = []

        # --------------------------------------------------
        # System Role
        # --------------------------------------------------

        prompt.append(
            "You are an expert 5G Network Monitoring Assistant."
        )

        prompt.append(
            "Analyze the monitoring metrics and explain the "
            "health of the 5G network."
        )

        prompt.append(
            "Use ONLY the provided monitoring information. "
            "Do not invent metrics, alarms, or network events."
        )

        prompt.append("")

        # --------------------------------------------------
        # User Request
        # --------------------------------------------------

        prompt.append("User Request:")
        prompt.append(user_query)
        prompt.append("")

        # --------------------------------------------------
        # Monitoring Summary
        # --------------------------------------------------

        prompt.append("Monitoring Summary:")

        prompt.append(
            f"- Total Metrics   : {analysis.summary.total_metrics}"
        )

        prompt.append(
            f"- Normal Metrics  : {analysis.summary.normal_metrics}"
        )

        prompt.append(
            f"- Warning Metrics : {analysis.summary.warning_metrics}"
        )

        prompt.append(
            f"- Critical Metrics: {analysis.summary.critical_metrics}"
        )

        prompt.append("")

        # --------------------------------------------------
        # Affected Components
        # --------------------------------------------------

        if analysis.affected_components:

            prompt.append("Affected Components:")

            for component in sorted(set(analysis.affected_components)):

                prompt.append(f"- {component}")

            prompt.append("")

        # --------------------------------------------------
        # Detected Anomalies
        # --------------------------------------------------

        if analysis.anomalies:

            prompt.append("Detected Anomalies:")

            for anomaly in analysis.anomalies:

                prompt.append(f"- {anomaly}")

            prompt.append("")

        # --------------------------------------------------
        # Alerts
        # --------------------------------------------------

        if analysis.alerts:

            prompt.append("Alerts:")

            for alert in analysis.alerts:

                prompt.append(
                    f"- [{alert.severity}] "
                    f"{alert.component} "
                    f"{alert.metric} = "
                    f"{alert.value}"
                )

                prompt.append(
                    f"  Threshold : {alert.threshold}"
                )

                prompt.append(
                    f"  Message   : {alert.message}"
                )

            prompt.append("")

        # --------------------------------------------------
        # Raw Metrics
        # --------------------------------------------------

        if analysis.metrics:

            prompt.append("Observed Metrics:")

            for metric in analysis.metrics:

                prompt.append(
                    f"- {metric.component} "
                    f"{metric.name} = "
                    f"{metric.value}{metric.unit}"
                )

                # Optional live-source evidence. Existing text metrics do not
                # populate these fields, so their prompt format is unchanged.
                if metric.source:
                    prompt.append(f"  Source    : {metric.source}")

                if metric.timestamp:
                    prompt.append(f"  Timestamp : {metric.timestamp}")

                if metric.gnb_id:
                    prompt.append(f"  gNB ID    : {metric.gnb_id}")

                if metric.imsi:
                    prompt.append(f"  IMSI      : {metric.imsi}")

                if metric.description:
                    prompt.append(f"  Context   : {metric.description}")

            prompt.append("")

        # --------------------------------------------------
        # Instructions
        # --------------------------------------------------

        prompt.append("Tasks:")

        prompt.append(
            "1. Assess the overall health of the monitored system."
        )

        prompt.append(
            "2. Identify the most probable cause of any abnormal metrics."
        )

        prompt.append(
            "3. Explain the possible impact on the 5G network."
        )

        prompt.append(
            "4. Recommend practical troubleshooting steps."
        )

        prompt.append(
            "5. Mention any critical metrics that require immediate attention."
        )

        prompt.append(
            "6. If insufficient monitoring data is available, clearly state what additional metrics are required."
        )

        prompt.append(
            "7. Base your answer only on the provided monitoring information."
        )

        return "\n".join(prompt)
