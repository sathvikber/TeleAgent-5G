"""
Monitoring Analysis Service

Coordinates the complete monitoring analysis pipeline.
"""

from teleagent.monitoring.metric_parser import MetricParser
from teleagent.monitoring.threshold_checker import ThresholdChecker
from teleagent.monitoring.anomaly_detector import AnomalyDetector
from teleagent.monitoring.monitoring_prompt_builder import (
    MonitoringPromptBuilder,
)
from teleagent.monitoring.models import (
    MonitoringData,
    MonitoringAnalysisResult,
    MonitoringSummary,
)


class MonitoringAnalysisService:
    """
    Orchestrates the monitoring analysis pipeline.
    """

    def __init__(
        self,
        metric_parser: MetricParser | None = None,
        threshold_checker: ThresholdChecker | None = None,
        anomaly_detector: AnomalyDetector | None = None,
        prompt_builder: MonitoringPromptBuilder | None = None,
    ):

        self._metric_parser = (
            metric_parser or MetricParser()
        )

        self._threshold_checker = (
            threshold_checker or ThresholdChecker()
        )

        self._anomaly_detector = (
            anomaly_detector or AnomalyDetector()
        )

        self._prompt_builder = (
            prompt_builder or MonitoringPromptBuilder()
        )

    # ---------------------------------------------------------
    # Analysis
    # ---------------------------------------------------------

    def analyze(
        self,
        raw_metrics: str | MonitoringData,
    ) -> MonitoringAnalysisResult:
        """
        Analyze raw monitoring metrics or normalized live monitoring data.
        """

        if isinstance(raw_metrics, MonitoringData):
            metrics = raw_metrics.metrics
            source_payloads = raw_metrics.source_payloads
        else:
            # Preserve the existing text parsing path.
            metrics = self._metric_parser.parse(raw_metrics)
            source_payloads = []

        # Threshold checking
        alerts = self._threshold_checker.check(metrics)

        # Detect anomalies
        anomalies = self._anomaly_detector.detect(alerts)

        # Summary
        summary = self._build_summary(
            metrics,
            alerts,
        )

        # Components affected
        affected_components = sorted(
            {
                alert.component
                for alert in alerts
            }
        )

        return MonitoringAnalysisResult(
            summary=summary,
            metrics=metrics,
            alerts=alerts,
            anomalies=anomalies,
            affected_components=affected_components,
            source_payloads=source_payloads,
        )

    # ---------------------------------------------------------
    # Prompt
    # ---------------------------------------------------------

    def build_prompt(
        self,
        user_query: str,
        analysis: MonitoringAnalysisResult,
    ) -> str:
        """
        Build prompt for the Telecom SLM.
        """

        return self._prompt_builder.build(
            user_query=user_query,
            analysis=analysis,
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def _build_summary(
        self,
        metrics,
        alerts,
    ) -> MonitoringSummary:
        """
        Build monitoring summary statistics.
        """

        warning_count = sum(
            1
            for alert in alerts
            if alert.severity == "WARNING"
        )

        critical_count = sum(
            1
            for alert in alerts
            if alert.severity == "CRITICAL"
        )

        return MonitoringSummary(
            total_metrics=len(metrics),
            normal_metrics=len(metrics) - len(alerts),
            warning_metrics=warning_count,
            critical_metrics=critical_count,
        )
