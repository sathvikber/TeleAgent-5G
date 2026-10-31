"""
Monitoring Data Models

Shared data models used by the MonitoringAgent pipeline.
"""

from dataclasses import dataclass, field
from typing import Any


# ------------------------------------------------------------------
# Individual Metric
# ------------------------------------------------------------------

@dataclass
class Metric:
    """
    Represents a single monitoring metric.
    """

    name: str
    value: float
    unit: str
    component: str
    timestamp: str | None = None
    source: str | None = None
    gnb_id: str | None = None
    imsi: str | None = None
    description: str | None = None


@dataclass
class MonitoringData:
    """Normalized monitoring data supplied by an optional live provider."""

    metrics: list[Metric] = field(default_factory=list)
    source_payloads: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class MonitoringDataRequest:
    """Monitoring information required to answer an already-routed request."""

    network_state: bool = False
    gnb_throughput: bool = False
    ue_throughput: bool = False


# ------------------------------------------------------------------
# Alert
# ------------------------------------------------------------------

@dataclass
class Alert:
    """
    Represents an abnormal monitoring event.
    """

    metric: str
    component: str
    value: float
    threshold: float
    severity: str
    message: str


# ------------------------------------------------------------------
# Monitoring Summary
# ------------------------------------------------------------------

@dataclass
class MonitoringSummary:
    """
    Overall monitoring statistics.
    """

    total_metrics: int = 0
    normal_metrics: int = 0
    warning_metrics: int = 0
    critical_metrics: int = 0


# ------------------------------------------------------------------
# Monitoring Analysis Result
# ------------------------------------------------------------------

@dataclass
class MonitoringAnalysisResult:
    """
    Output of the Monitoring Analysis Service.
    """

    summary: MonitoringSummary

    metrics: list[Metric] = field(default_factory=list)

    alerts: list[Alert] = field(default_factory=list)

    affected_components: list[str] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    source_payloads: list[Any] = field(default_factory=list)
