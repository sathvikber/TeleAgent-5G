"""
Anomaly Detector

Detects abnormal monitoring conditions from generated alerts.
"""

from teleagent.monitoring.models import Alert


class AnomalyDetector:
    """
    Detects anomalies from monitoring alerts.
    """

    def detect(
        self,
        alerts: list[Alert],
    ) -> list[str]:
        """
        Detect anomalies.

        Parameters
        ----------
        alerts : list[Alert]

        Returns
        -------
        list[str]
        """

        anomalies = []

        if not alerts:
            return anomalies

        # --------------------------------------------------
        # Critical alerts
        # --------------------------------------------------

        critical_alerts = [
            alert
            for alert in alerts
            if alert.severity == "CRITICAL"
        ]

        if critical_alerts:

            anomalies.append(
                f"{len(critical_alerts)} critical metric(s) detected."
            )

        # --------------------------------------------------
        # CPU
        # --------------------------------------------------

        cpu_alerts = [
            alert
            for alert in alerts
            if alert.metric == "CPU"
        ]

        if cpu_alerts:

            anomalies.append(
                "High CPU utilization detected."
            )

        # --------------------------------------------------
        # Memory
        # --------------------------------------------------

        memory_alerts = [
            alert
            for alert in alerts
            if alert.metric == "Memory"
        ]

        if memory_alerts:

            anomalies.append(
                "High memory utilization detected."
            )

        # --------------------------------------------------
        # Latency
        # --------------------------------------------------

        latency_alerts = [
            alert
            for alert in alerts
            if alert.metric == "Latency"
        ]

        if latency_alerts:

            anomalies.append(
                "Network latency is above threshold."
            )

        # --------------------------------------------------
        # Packet Loss
        # --------------------------------------------------

        packet_loss_alerts = [
            alert
            for alert in alerts
            if alert.metric == "PacketLoss"
        ]

        if packet_loss_alerts:

            anomalies.append(
                "Packet loss detected."
            )

        # --------------------------------------------------
        # Throughput
        # --------------------------------------------------

        throughput_alerts = [
            alert
            for alert in alerts
            if alert.metric.endswith("Throughput")
        ]

        if throughput_alerts:

            anomalies.append(
                "Throughput degradation detected."
            )

        return anomalies
