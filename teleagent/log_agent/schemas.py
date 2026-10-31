from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LogRecord:
    source_file: str
    line_number: Optional[int] = None

    timestamp: Optional[str] = None
    component: str = "UNKNOWN"
    log_level: str = "UNKNOWN"
    protocol: str = "UNKNOWN"

    ip_addresses: List[str] = field(default_factory=list)

    error_code: Optional[str] = None
    event_type: str = "LOG_EVENT"

    message: str = ""
    raw_log: Any = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogSummary:
    total_events: int

    components: Dict[str, int]
    log_levels: Dict[str, int]
    protocols: Dict[str, int]

    errors: List[Dict[str, Any]]


@dataclass
class IncidentContext:
    task: str
    source_files: List[str]

    summary: Dict[str, Any]

    critical_events: List[Dict[str, Any]]
    event_timeline: List[Dict[str, Any]]

    relevant_components: List[str]

    structured_logs: List[Dict[str, Any]]