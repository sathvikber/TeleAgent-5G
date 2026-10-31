"""
Threshold Checker

Checks monitoring metrics against predefined thresholds and
generates alerts.
"""

from teleagent.monitoring.models import (
    Alert,
    Metric,
)


class ThresholdChecker:
    """
    Checks monitoring metrics against predefined thresholds.
    """

    DEFAULT_THRESHOLDS = {

        "CPU": {
            "warning": 80.0,
            "critical": 90.0,
        },

        "Memory": {
            "warning": 80.0,
            "critical": 90.0,
        },

        "Latency": {
            "warning": 100.0,
            "critical": 200.0,
        },

        "PacketLoss": {
            "warning": 2.0,
            "critical": 5.0,
        },

        "Throughput": {
            # Throughput is abnormal when it drops
            "warning": 100.0,
            "critical": 50.0,
        },
    }

    def __init__(
        self,
        thresholds: dict | None = None,
    ):
        self._thresholds = (
            thresholds
            if thresholds is not None
            else self.DEFAULT_THRESHOLDS
        )

    def check(
        self,
        metrics: list[Metric],
    ) -> list[Alert]:
        """
        Check all metrics against thresholds.

        Parameters
        ----------
        metrics : list[Metric]

        Returns
        -------
        list[Alert]
        """

        alerts = []

        for metric in metrics:

            alert = self._check_metric(metric)

            if alert is not None:
                alerts.append(alert)

        return alerts

    def _check_metric(
        self,
        metric: Metric,
    ) -> Alert | None:
        """
        Check a single metric.
        """

        is_throughput = metric.name.endswith("Throughput")
        if is_throughput and (metric.source or "").startswith("niral."):
            # Niral testbed throughput needs a learned/baseline threshold.
            # Keep it as evidence until that detector is introduced rather
            # than applying the generic absolute throughput thresholds.
            return None

        thresholds = self._thresholds.get(metric.name)
        if thresholds is None and is_throughput:
            thresholds = self._thresholds.get("Throughput")

        if thresholds is None:
            return None

        value = metric.value

        # --------------------------------------------------
        # Throughput (lower is worse)
        # --------------------------------------------------

        if is_throughput:

            if value <= thresholds["critical"]:

                severity = "CRITICAL"

            elif value <= thresholds["warning"]:

                severity = "WARNING"

            else:

                return None

        # --------------------------------------------------
        # All other metrics (higher is worse)
        # --------------------------------------------------

        else:

            if value >= thresholds["critical"]:

                severity = "CRITICAL"

            elif value >= thresholds["warning"]:

                severity = "WARNING"

            else:

                return None

        return Alert(
            metric=metric.name,
            component=metric.component,
            value=value,
            threshold=thresholds[
                severity.lower()
            ],
            severity=severity,
            message=(
                f"{metric.name} exceeded "
                f"{severity.lower()} threshold"
            ),
        )
