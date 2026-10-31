import socket
from urllib.error import HTTPError
from unittest.mock import patch

import pytest

from teleagent.monitoring.niral_client import (
    NiralClient,
    NiralHTTPError,
    NiralResponseError,
    NiralTimeoutError,
)


class FakeResponse:

    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def test_get_network_state_uses_known_endpoint():
    client = NiralClient("https://niral.example", retries=0)

    with patch(
        "teleagent.monitoring.niral_client.urlopen",
        return_value=FakeResponse(b'{"state": "healthy"}'),
    ) as urlopen:
        assert client.get_network_state() == {"state": "healthy"}

    request = urlopen.call_args.args[0]
    assert request.full_url.endswith(
        "/v2/5gcore/overview/cards_core_gnb_ue_graph/range=hour"
    )
    assert request.get_header("Authorization") is None


def test_get_gnb_throughput_uses_known_endpoint():
    client = NiralClient("https://niral.example", retries=0)

    with patch(
        "teleagent.monitoring.niral_client.urlopen",
        return_value=FakeResponse(b'{"max": 100}'),
    ) as urlopen:
        assert client.get_gnb_throughput() == {"max": 100}

    assert urlopen.call_args.args[0].full_url.endswith(
        "/v2/5gcore/throughput/max/gnb"
    )


def test_get_ue_throughput_uses_known_endpoint_without_query_parameters():
    client = NiralClient("https://niral.example", retries=0)

    with patch(
        "teleagent.monitoring.niral_client.urlopen",
        return_value=FakeResponse(b'{"max": 50}'),
    ) as urlopen:
        assert client.get_ue_throughput() == {"max": 50}

    assert urlopen.call_args.args[0].full_url == (
        "https://niral.example/v2/5gcore/throughput/max/imsi"
    )


def test_http_failure_is_reported_clearly():
    client = NiralClient("https://niral.example", retries=0)
    error = HTTPError("https://niral.example", 401, "Unauthorized", {}, None)

    with patch("teleagent.monitoring.niral_client.urlopen", side_effect=error):
        with pytest.raises(NiralHTTPError, match="HTTP 401"):
            client.get_network_state()


def test_timeout_is_reported_clearly():
    client = NiralClient("https://niral.example", retries=0)

    with patch(
        "teleagent.monitoring.niral_client.urlopen",
        side_effect=socket.timeout(),
    ):
        with pytest.raises(NiralTimeoutError):
            client.get_network_state()


def test_malformed_json_is_reported_clearly():
    client = NiralClient("https://niral.example", retries=0)

    with patch(
        "teleagent.monitoring.niral_client.urlopen",
        return_value=FakeResponse(b"not json"),
    ):
        with pytest.raises(NiralResponseError, match="malformed JSON"):
            client.get_network_state()
