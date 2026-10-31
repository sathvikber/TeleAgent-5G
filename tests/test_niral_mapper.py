from teleagent.monitoring.models import MonitoringDataRequest
from teleagent.monitoring.niral_mapper import (
    NiralMapper,
    NiralMonitoringDataProvider,
)


class FakeNiralClient:

    def get_network_state(self):
        return {
            "timestamp": "2026-09-03T10:00:00Z",
            "gnb_id": "gnb-42",
            "cpu_percent": 91,
        }

    def get_gnb_throughput(self):
        return {
            "gnb_id": "gnb-42",
            "timestamp": "2026-09-03T10:01:00Z",
            "uplink_throughput": 40,
        }

    def get_ue_throughput(self):
        return {
            "imsi": "001010000000001",
            "timestamp": "2026-09-03T10:02:00Z",
            "total_throughput": 60,
        }


def test_mapper_preserves_metric_evidence_context():
    metrics = NiralMapper().map_payload(
        {
            "timestamp": "2026-09-03T10:00:00Z",
            "gnb_id": "gnb-42",
            "packet_loss": 3.5,
        },
        "niral.network_state",
    )

    assert len(metrics) == 1
    assert metrics[0].name == "PacketLoss"
    assert metrics[0].value == 3.5
    assert metrics[0].source == "niral.network_state"
    assert metrics[0].timestamp == "2026-09-03T10:00:00Z"
    assert metrics[0].gnb_id == "gnb-42"
    assert metrics[0].component == "gNB"


def test_provider_maps_only_requested_gnb_payload_and_preserves_raw_json():
    provider = NiralMonitoringDataProvider(FakeNiralClient())

    data = provider.get_monitoring_data(
        MonitoringDataRequest(gnb_throughput=True)
    )

    assert len(data.metrics) == 1
    assert data.metrics[0].name == "UplinkThroughput"
    assert data.metrics[0].component == "gNB"
    assert data.source_payloads == [
        {
            "source": "niral.gnb_throughput",
            "payload": {
                "gnb_id": "gnb-42",
                "timestamp": "2026-09-03T10:01:00Z",
                "uplink_throughput": 40,
            },
        }
    ]


def test_provider_maps_only_requested_ue_payload():
    provider = NiralMonitoringDataProvider(FakeNiralClient())

    data = provider.get_monitoring_data(
        MonitoringDataRequest(ue_throughput=True)
    )

    assert len(data.metrics) == 1
    assert data.metrics[0].name == "TotalThroughput"
    assert data.metrics[0].component == "UE"
    assert data.metrics[0].imsi == "001010000000001"


def test_provider_maps_only_latest_gnb_throughput_record_and_preserves_history():
    payload = [
        {
            "gnb_id": 3584,
            "uplink_throughput": 100,
            "downlink_throughput": 200,
            "total_throughput": 300,
            "current_time": "2026-09-25T11:14:50",
        },
        {
            "gnb_id": 3584,
            "uplink_throughput": 6301,
            "downlink_throughput": 119029,
            "total_throughput": 125331,
            "current_time": "2026-09-25T11:15:50",
        },
    ]

    class ThroughputClient:
        def get_gnb_throughput(self):
            return payload

    data = NiralMonitoringDataProvider(ThroughputClient()).get_monitoring_data(
        MonitoringDataRequest(gnb_throughput=True)
    )

    assert {metric.name for metric in data.metrics} == {
        "UplinkThroughput",
        "DownlinkThroughput",
        "TotalThroughput",
    }
    assert {metric.value for metric in data.metrics} == {6301.0, 119029.0, 125331.0}
    assert all(metric.gnb_id == "3584" for metric in data.metrics)
    assert all(metric.timestamp == "2026-09-25T11:15:50" for metric in data.metrics)
    assert all(metric.unit == "" for metric in data.metrics)
    assert data.source_payloads == [
        {"source": "niral.gnb_throughput", "payload": payload}
    ]


def test_provider_maps_latest_ue_throughput_per_imsi():
    payload = [
        {
            "imsi": "001", "gnb_id": 1, "total_throughput": 10,
            "current_time": "2026-09-25T11:14:00",
        },
        {
            "imsi": "002", "gnb_id": 1, "total_throughput": 20,
            "current_time": "2026-09-25T11:15:00",
        },
        {
            "imsi": "001", "gnb_id": 2, "total_throughput": 30,
            "current_time": "2026-09-25T11:16:00",
        },
    ]

    class ThroughputClient:
        def get_ue_throughput(self):
            return payload

    data = NiralMonitoringDataProvider(ThroughputClient()).get_monitoring_data(
        MonitoringDataRequest(ue_throughput=True)
    )

    assert {(metric.imsi, metric.value) for metric in data.metrics} == {
        ("001", 30.0),
        ("002", 20.0),
    }
    assert {metric.gnb_id for metric in data.metrics} == {"1", "2"}
    assert {metric.timestamp for metric in data.metrics} == {
        "2026-09-25T11:15:00",
        "2026-09-25T11:16:00",
    }


def test_mapper_preserves_throughput_and_network_state_semantics():
    metrics = NiralMapper().map_payload(
        {
            "uplink_throughput": 10,
            "downlink_throughput": 20,
            "total_throughput": 30,
            "totalUe": 100,
            "totalUeSession": 200,
            "highestRegisterUeCount": 50,
        },
        "niral.network_state",
    )

    by_name = {metric.name: metric for metric in metrics}
    assert set(by_name) == {
        "UplinkThroughput",
        "DownlinkThroughput",
        "TotalThroughput",
        "TotalUe",
        "TotalUeSession",
        "HighestRegisterUeCount",
    }
    assert "not an active-UE count" in by_name["TotalUe"].description
    assert "not a current active count" in by_name["HighestRegisterUeCount"].description


def test_mapper_maps_numeric_strings_without_mapping_context_or_text_fields():
    metrics = NiralMapper().map_payload(
        {
            "timestamp": "2026-09-03T10:00:00Z",
            "gnbId": "gnb-42",
            "imsi": "001010000000001",
            "nfName": "amf-1",
            "activeUeCount": "0",
            "registerUeCount": "0",
            "totalUe": "4",
            "activeGnb": "1",
            "totalGnb": "1",
            "state": "healthy",
            "enabled": True,
        },
        "niral.network_state",
    )

    by_name = {metric.name: metric for metric in metrics}
    assert {metric.value for metric in metrics} == {0.0, 1.0, 4.0}
    assert len(metrics) == 5
    assert by_name["TotalUe"].value == 4.0
    assert all(metric.timestamp == "2026-09-03T10:00:00Z" for metric in metrics)
    assert all(metric.gnb_id == "gnb-42" for metric in metrics)
    assert all(metric.imsi == "001010000000001" for metric in metrics)


def test_provider_maps_only_live_network_state_and_preserves_history():
    payload = {
        "liveDataModel": {
            "localDateTime": "2026-09-03T10:00:00Z",
            "activeAmf": "1",
            "activeGnb": "2",
            "activeUeCount": "3",
            "totalUeSession": "4",
        },
        "combineGraphData": [
            {
                "localDateTime": "2026-09-03T09:00:00Z",
                "activeAmf": "99",
            }
        ],
    }

    class NetworkStateClient:
        def get_network_state(self):
            return payload

    provider = NiralMonitoringDataProvider(NetworkStateClient())
    data = provider.get_monitoring_data(MonitoringDataRequest(network_state=True))

    assert len(data.metrics) == 4
    assert {metric.value for metric in data.metrics} == {1.0, 2.0, 3.0, 4.0}
    assert {metric.component for metric in data.metrics} == {"AMF", "gNB", "UE", "5G Core"}
    assert all(metric.timestamp == "2026-09-03T10:00:00Z" for metric in data.metrics)
    assert data.source_payloads == [
        {"source": "niral.network_state", "payload": payload}
    ]
