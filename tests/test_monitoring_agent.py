import teleagent.config as config

from teleagent.agents.monitoring_agent import MonitoringAgent
from teleagent.coordinator.task import Task
from teleagent.monitoring.models import Metric, MonitoringData, MonitoringDataRequest


class MockReasoningService:

    def __init__(self):
        self.request = None

    def generate(self, request):
        self.request = request
        return "Monitoring Response"


class MockMonitoringDataProvider:

    def __init__(self):
        self.request = None

    def get_monitoring_data(self, request: MonitoringDataRequest) -> MonitoringData:
        self.request = request
        return MonitoringData(
            metrics=[
                Metric(
                    name="UplinkThroughput",
                    value=40.0,
                    unit="Mbps",
                    component="gNB",
                    timestamp="2026-09-03T10:00:00Z",
                    source="niral.gnb_throughput",
                    gnb_id="gnb-1",
                    description="Uplink throughput; distinct from downlink and total throughput.",
                )
            ]
        )


def test_monitoring_agent():

    reasoning = MockReasoningService()
    agent = MonitoringAgent(reasoning)

    response = agent.execute("Show CPU utilization")

    assert response == "Monitoring Response"
    assert "Total Metrics   : 0" in reasoning.request.query


def test_monitoring_agent_uses_optional_live_provider():

    reasoning = MockReasoningService()
    provider = MockMonitoringDataProvider()
    agent = MonitoringAgent(
        reasoning_service=reasoning,
        monitoring_data_provider=provider,
    )

    response = agent.execute("Monitor gNB throughput")

    assert response == "Monitoring Response"
    assert provider.request == MonitoringDataRequest(gnb_throughput=True)
    assert "UplinkThroughput = 40.0Mbps" in reasoning.request.query
    assert "Source    : niral.gnb_throughput" in reasoning.request.query
    assert "gNB ID    : gnb-1" in reasoning.request.query
    assert "Context   : Uplink throughput; distinct from downlink" in reasoning.request.query


def test_monitoring_agent_prints_latest_normalized_throughput_when_debug_enabled(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(config, "MONITORING_DEBUG", True)
    agent = MonitoringAgent(
        reasoning_service=MockReasoningService(),
        monitoring_data_provider=MockMonitoringDataProvider(),
    )

    agent.execute("Monitor gNB throughput")

    output = capsys.readouterr().out
    assert "=== Live gNB Throughput ===" in output
    assert "gNB ID        : gnb-1" in output
    assert "Timestamp     : 2026-09-03T10:00:00Z" in output
    assert "Uplink        : 40.0" in output
    assert "Downlink      : " in output
    assert "Total         : " in output
    assert "source=" not in output


def test_monitoring_agent_prints_latest_ue_throughput_per_imsi_when_debug_enabled(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(config, "MONITORING_DEBUG", True)
    data = MonitoringData(
        metrics=[
            Metric(
                name="TotalThroughput",
                value=60.0,
                unit="",
                component="UE",
                timestamp="2026-09-03T10:02:00Z",
                source="niral.ue_throughput",
                gnb_id="gnb-42",
                imsi="001010000000001",
            ),
            Metric(
                name="TotalThroughput",
                value=70.0,
                unit="",
                component="UE",
                timestamp="2026-09-03T10:03:00Z",
                source="niral.ue_throughput",
                gnb_id="gnb-43",
                imsi="001010000000002",
            ),
        ]
    )

    MonitoringAgent._print_live_monitoring_evidence(data)

    output = capsys.readouterr().out
    assert output.count("=== Live UE Throughput ===") == 2
    assert "IMSI          : 001010000000001" in output
    assert "IMSI          : 001010000000002" in output
    assert "gNB ID        : gnb-42" in output
    assert "gNB ID        : gnb-43" in output
    assert "Timestamp     : 2026-09-03T10:02:00Z" in output
    assert "Timestamp     : 2026-09-03T10:03:00Z" in output
    assert "Total         : 60.0" in output
    assert "Total         : 70.0" in output


def test_monitoring_agent_uses_query_field_from_structured_request():
    reasoning = MockReasoningService()
    provider = MockMonitoringDataProvider()
    agent = MonitoringAgent(reasoning, monitoring_data_provider=provider)

    agent.execute(Task(task_id="task-1", intent="monitoring", query="Monitor UE throughput"))

    assert provider.request == MonitoringDataRequest(ue_throughput=True)
    assert "User Request:\nMonitor UE throughput" in reasoning.request.query


def test_monitoring_data_request_selects_multiple_resources_when_needed():
    request = MonitoringAgent._monitoring_data_request("Monitor gNB and UE throughput")
    assert request == MonitoringDataRequest(
        gnb_throughput=True,
        ue_throughput=True,
    )

    request = MonitoringAgent._monitoring_data_request("Monitor throughput")
    assert request == MonitoringDataRequest(
        gnb_throughput=True,
        ue_throughput=True,
    )

    request = MonitoringAgent._monitoring_data_request("What is the current network status?")
    assert request == MonitoringDataRequest(network_state=True)
