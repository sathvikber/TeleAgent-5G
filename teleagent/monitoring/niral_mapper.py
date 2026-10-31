from __future__ import annotations

from typing import Any

from teleagent.monitoring.models import (
    Metric,
    MonitoringData,
    MonitoringDataRequest,
)
from teleagent.monitoring.niral_client import NiralClient


class NiralMapper:
    """Converts Niral JSON payloads to the vendor-neutral monitoring model."""

    _TIMESTAMP_KEYS = {
        "timestamp",
        "time",
        "datetime",
        "localdatetime",
        "current_time",
        "date",
        "collected_at",
    }

    _GNB_ID_KEYS = {"gnb_id", "gnbid", "gnb"}
    _IMSI_KEYS = {"imsi", "ue_imsi", "subscriber_imsi"}

    _ID_KEYS = _GNB_ID_KEYS | _IMSI_KEYS | {
        "id",
        "cell_id",
        "ue_id",
        "nfname",
        "nf_name",
    }

    def map_payload(self, payload: Any, source: str) -> list[Metric]:
        """Map numeric observations in a Niral payload to Metric objects."""

        if source == "niral.network_state" and isinstance(payload, dict):
            payload = payload.get("liveDataModel", payload)

        # Existing gNB and UE throughput endpoints.
        if source in {
            "niral.gnb_throughput",
            "niral.ue_throughput",
        }:
            return [
                metric
                for record in self._current_throughput_records(payload, source)
                for metric in self._map_value(record, source, {})
            ]

        # New endpoint:
        # /v2/5gcore/uestatistics/imsi_list/pageNumber=1?filtter=1
        #
        # This endpoint returns current records for multiple IMSIs.
        if source == "niral.ue_throughput_all":
            return [
                metric
                for record in self._ue_throughput_all_records(payload)
                for metric in self._map_value(record, source, {})
            ]

        return list(self._map_value(payload, source, {}))

    def _current_throughput_records(
        self,
        payload: Any,
        source: str,
    ) -> list[dict[str, Any]]:
        """Return the current Niral throughput record(s).

        gNB throughput is a single current view.

        Existing UE throughput is current per IMSI, since its endpoint
        can return history for more than one subscriber.
        """

        records = self._throughput_records(payload)

        if not records:
            return []

        if source == "niral.gnb_throughput":
            return [self._latest_record(records)]

        current_by_imsi: dict[str | None, dict[str, Any]] = {}

        for record in records:
            imsi = self._string_value(record, self._IMSI_KEYS)

            previous = current_by_imsi.get(imsi)

            if previous is None or self._is_newer(record, previous):
                current_by_imsi[imsi] = record

        return list(current_by_imsi.values())

    @staticmethod
    def _ue_throughput_all_records(
        payload: Any,
    ) -> list[dict[str, Any]]:
        """Normalize the new Niral all-UE response.

        The endpoint may return:
        - a list of UE records, or
        - a dictionary whose values are UE records.

        Each record contains:
            key                 -> IMSI
            gnbId               -> gNB ID
            uplink_throughput
            downlink_throughput
            total_throughput
        """

        records: list[dict[str, Any]] = []

        if isinstance(payload, list):
            source_records = payload

        elif isinstance(payload, dict):
            source_records = list(payload.values())

        else:
            source_records = []

        for record in source_records:
            if not isinstance(record, dict):
                continue

            # The new endpoint uses "key" for the IMSI.
            # Convert it to the standard "imsi" field expected
            # by the generic mapper.
            normalized_record = dict(record)

            if "key" in normalized_record:
                normalized_record["imsi"] = normalized_record["key"]

            records.append(normalized_record)

        return records

    @staticmethod
    def _throughput_records(
        payload: Any,
    ) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [
                item
                for item in payload
                if isinstance(item, dict)
            ]

        if isinstance(payload, dict):
            # Accommodate the common API-envelope forms without treating
            # other nested values as throughput history.
            for key in ("data", "records", "items", "result"):
                value = payload.get(key)

                if isinstance(value, list):
                    return [
                        item
                        for item in value
                        if isinstance(item, dict)
                    ]

            return [payload]

        return []

    def _latest_record(
        self,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return max(
            enumerate(records),
            key=lambda item: self._record_sort_key(*item),
        )[1]

    def _is_newer(
        self,
        candidate: dict[str, Any],
        current: dict[str, Any],
    ) -> bool:
        return self._record_sort_key(
            1,
            candidate,
        ) >= self._record_sort_key(
            0,
            current,
        )

    def _record_sort_key(
        self,
        index: int,
        record: dict[str, Any],
    ) -> tuple[bool, str, int]:
        # Niral timestamps are ISO-like strings, so lexical ordering
        # preserves chronological ordering. The array position breaks
        # ties and provides a sensible fallback when an endpoint
        # omits a timestamp.

        timestamp = self._string_value(
            record,
            self._TIMESTAMP_KEYS,
        )

        return (
            timestamp is not None,
            timestamp or "",
            index,
        )

    def _map_value(
        self,
        value: Any,
        source: str,
        context: dict[str, str],
        field_name: str | None = None,
        unit: str = "",
    ) -> list[Metric]:

        if isinstance(value, dict):
            next_context = self._context_from(
                value,
                context,
            )

            record_unit = (
                self._string_value(
                    value,
                    {"unit", "units"},
                )
                or unit
            )

            metrics: list[Metric] = []

            for key, child in value.items():
                normalized_key = key.lower()

                if normalized_key in (
                    self._TIMESTAMP_KEYS
                    | self._ID_KEYS
                    | {"unit", "units"}
                ):
                    continue

                metrics.extend(
                    self._map_value(
                        child,
                        source,
                        next_context,
                        field_name=key,
                        unit=record_unit,
                    )
                )

            return metrics

        if isinstance(value, list):
            metrics: list[Metric] = []

            for item in value:
                metrics.extend(
                    self._map_value(
                        item,
                        source,
                        context,
                        field_name,
                        unit,
                    )
                )

            return metrics

        numeric_value = self._numeric_value(value)

        if numeric_value is None:
            return []

        if not field_name:
            return []

        return [
            Metric(
                name=self._metric_name(
                    field_name,
                    source,
                ),
                value=numeric_value,
                unit=self._unit_for(
                    field_name,
                    unit,
                ),
                component=self._component_for(
                    source,
                    context,
                    field_name,
                ),
                timestamp=context.get("timestamp"),
                source=source,
                gnb_id=context.get("gnb_id"),
                imsi=context.get("imsi"),
                description=self._metric_description(
                    field_name,
                    source,
                ),
            )
        ]

    @staticmethod
    def _numeric_value(
        value: Any,
    ) -> float | None:
        """Return numeric JSON values, including numeric strings, as floats."""

        if isinstance(value, bool):
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None

        return None

    def _context_from(
        self,
        value: dict[str, Any],
        parent: dict[str, str],
    ) -> dict[str, str]:

        context = dict(parent)

        timestamp = self._string_value(
            value,
            self._TIMESTAMP_KEYS,
        )

        gnb_id = self._string_value(
            value,
            self._GNB_ID_KEYS,
        )

        imsi = self._string_value(
            value,
            self._IMSI_KEYS,
        )

        if timestamp is not None:
            context["timestamp"] = timestamp

        if gnb_id is not None:
            context["gnb_id"] = gnb_id

        if imsi is not None:
            context["imsi"] = imsi

        return context

    @staticmethod
    def _string_value(
        value: dict[str, Any],
        keys: set[str],
    ) -> str | None:

        for key, candidate in value.items():
            if key.lower() in keys and candidate is not None:
                return str(candidate)

        return None

    @staticmethod
    def _metric_name(
        field_name: str,
        source: str,
    ) -> str:

        normalized = field_name.lower().replace(
            "_",
            "",
        )

        names = {
            "cpu": "CPU",
            "memory": "Memory",
            "latency": "Latency",
            "packetloss": "PacketLoss",
            "uplinkthroughput": "UplinkThroughput",
            "downlinkthroughput": "DownlinkThroughput",
            "totalthroughput": "TotalThroughput",
            "totalue": "TotalUe",
            "totaluesession": "TotalUeSession",
            "highestregisteruecount": "HighestRegisterUeCount",
        }

        if normalized in names:
            return names[normalized]

        for key, name in names.items():
            if key in normalized:
                return name

        if source.endswith(
            (
                "gnb_throughput",
                "ue_throughput",
            )
        ) and normalized in {
            "value",
            "max",
            "maximum",
        }:
            return "MaximumThroughput"

        return field_name.replace(
            "_",
            " ",
        ).title()

    @staticmethod
    def _metric_description(
        field_name: str,
        source: str,
    ) -> str | None:

        normalized = field_name.lower().replace(
            "_",
            "",
        )

        descriptions = {
            "uplinkthroughput": (
                "Uplink throughput; distinct from downlink "
                "and total throughput."
            ),
            "downlinkthroughput": (
                "Downlink throughput; distinct from uplink "
                "and total throughput."
            ),
            "totalthroughput": (
                "Total throughput; do not interpret it as "
                "uplink or downlink alone."
            ),
            "totalue": (
                "Reported total UE count; not an active-UE count."
            ),
            "totaluesession": (
                "Reported total UE session count; not an "
                "active-session count."
            ),
            "highestregisteruecount": (
                "Reported highest registered-UE count; not "
                "a current active count."
            ),
        }

        if normalized in descriptions:
            return descriptions[normalized]

        if source.endswith(
            (
                "gnb_throughput",
                "ue_throughput",
            )
        ) and normalized in {
            "value",
            "max",
            "maximum",
        }:
            return (
                "Maximum reported throughput; direction "
                "is not specified."
            )

        return None

    @staticmethod
    def _unit_for(
        field_name: str,
        supplied_unit: str,
    ) -> str:

        if supplied_unit:
            return supplied_unit

        normalized = field_name.lower()

        if (
            "percent" in normalized
            or normalized.endswith("pct")
        ):
            return "%"

        return ""

    @staticmethod
    def _component_for(
        source: str,
        context: dict[str, str],
        field_name: str,
    ) -> str:

        if source == "niral.network_state":
            normalized = field_name.lower().replace(
                "_",
                "",
            )

            if "amf" in normalized:
                return "AMF"

            if "gnb" in normalized:
                return "gNB"

            if "session" in normalized or "core" in normalized:
                return "5G Core"

            if "ue" in normalized:
                return "UE"

        if context.get("gnb_id"):
            return "gNB"

        if context.get("imsi"):
            return "UE"

        if source.endswith("gnb_throughput"):
            return "gNB"

        if source.endswith("ue_throughput"):
            return "UE"

        return "5G Core"


class NiralMonitoringDataProvider:
    """Supplies requested monitoring data without exposing Niral to the agent."""

    def __init__(
        self,
        client: NiralClient,
        mapper: NiralMapper | None = None,
    ) -> None:
        self._client = client
        self._mapper = mapper or NiralMapper()

    def get_monitoring_data(
        self,
        request: MonitoringDataRequest,
    ) -> MonitoringData:

        metrics: list[Metric] = []
        source_payloads: list[dict[str, Any]] = []

        for source, fetch in self._requested_sources(request):
            payload = fetch()

            source_payloads.append(
                {
                    "source": source,
                    "payload": payload,
                }
            )

            metrics.extend(
                self._mapper.map_payload(
                    payload,
                    source,
                )
            )

        return MonitoringData(
            metrics=metrics,
            source_payloads=source_payloads,
        )

    def _requested_sources(
        self,
        request: MonitoringDataRequest,
    ):
        sources = []

        if request.network_state:
            sources.append(
                (
                    "niral.network_state",
                    self._client.get_network_state,
                )
            )

        if request.gnb_throughput:
            sources.append(
                (
                    "niral.gnb_throughput",
                    self._client.get_gnb_throughput,
                )
            )

        if request.ue_throughput:
            sources.append(
                (
                    "niral.ue_throughput_all",
                    self._client.get_ue_throughput_all,
                )
            )

        return sources
