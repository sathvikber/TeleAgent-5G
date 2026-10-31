from teleagent.monitoring.models import Metric, MonitoringData
from teleagent.monitoring.monitoring_analysis_service import MonitoringAnalysisService


def test_normalized_live_metrics_use_existing_threshold_and_anomaly_pipeline():
    payload = {"raw": "kept for programmatic consumers"}
    analysis = MonitoringAnalysisService().analyze(
        MonitoringData(
            metrics=[
                Metric(
                    name="CPU",
                    value=95.0,
                    unit="%",
                    component="gNB",
                    source="test",
                )
            ],
            source_payloads=[payload],
        )
    )

    assert analysis.summary.total_metrics == 1
    assert analysis.summary.critical_metrics == 1
    assert analysis.anomalies == [
        "1 critical metric(s) detected.",
        "High CPU utilization detected.",
    ]
    assert analysis.source_payloads == [payload]


def test_text_metrics_keep_existing_parser_behavior():
    analysis = MonitoringAnalysisService().analyze("AMF CPU=92%")

    assert analysis.metrics[0].component == "AMF"
    assert analysis.metrics[0].name == "CPU"
    assert analysis.summary.critical_metrics == 1
    assert analysis.source_payloads == []


def test_directional_throughput_uses_existing_deterministic_thresholds():
    analysis = MonitoringAnalysisService().analyze(
        MonitoringData(
            metrics=[
                Metric(
                    name="UplinkThroughput",
                    value=40.0,
                    unit="Mbps",
                    component="gNB",
                )
            ]
        )
    )

    assert analysis.summary.critical_metrics == 1
    assert analysis.anomalies == [
        "1 critical metric(s) detected.",
        "Throughput degradation detected.",
    ]


def test_niral_throughput_is_preserved_without_absolute_threshold_alerts():
    analysis = MonitoringAnalysisService().analyze(
        MonitoringData(
            metrics=[
                Metric(
                    name="UplinkThroughput",
                    value=14.5,
                    unit="Mbps",
                    component="gNB",
                    source="niral.gnb_throughput",
                ),
                Metric(
                    name="DownlinkThroughput",
                    value=0.0,
                    unit="Mbps",
                    component="UE",
                    source="niral.ue_throughput",
                ),
            ]
        )
    )

    assert analysis.summary.total_metrics == 2
    assert analysis.summary.normal_metrics == 2
    assert analysis.alerts == []
    assert analysis.anomalies == []


def test_prompt_marks_total_network_state_values_as_not_active_counts():
    service = MonitoringAnalysisService()
    analysis = service.analyze(
        MonitoringData(
            metrics=[
                Metric(
                    name="TotalUe",
                    value=100.0,
                    unit="",
                    component="5G Core",
                    source="niral.network_state",
                    description="Reported total UE count; not an active-UE count.",
                )
            ]
        )
    )

    prompt = service.build_prompt("What is the current network status?", analysis)

    assert "TotalUe = 100.0" in prompt
    assert "Context   : Reported total UE count; not an active-UE count." in prompt
