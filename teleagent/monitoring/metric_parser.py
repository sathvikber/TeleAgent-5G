"""
Metric Parser

Parses raw monitoring metrics into structured Metric objects.
"""

import re

from teleagent.monitoring.models import Metric


class MetricParser:
    """
    Parses monitoring metrics.

    Expected input examples

    CPU=95%
    Memory=78%
    Latency=120ms
    PacketLoss=3%
    Throughput=850Mbps

    AMF CPU=92%
    UPF Latency=150ms
    MEC Memory=85%
    """

    DEFAULT_COMPONENT = "System"

    METRIC_PATTERN = re.compile(
        r"^(?:(\w+)\s+)?([A-Za-z_]+)\s*=\s*([\d.]+)\s*([A-Za-z%]*)$"
    )

    def parse(
        self,
        raw_metrics: str,
    ) -> list[Metric]:
        """
        Parse raw monitoring text.

        Parameters
        ----------
        raw_metrics : str

        Returns
        -------
        list[Metric]
        """

        metrics = []

        for line in raw_metrics.splitlines():

            line = line.strip()

            if not line:
                continue

            match = self.METRIC_PATTERN.match(line)

            if not match:
                continue

            component, name, value, unit = match.groups()

            if component is None:
                component = self.DEFAULT_COMPONENT

            metrics.append(
                Metric(
                    name=name,
                    value=float(value),
                    unit=unit,
                    component=component,
                )
            )

        return metrics
