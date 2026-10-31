import re


# ---------------------------------------------------------
# Valid log event line
# ---------------------------------------------------------

LOG_LINE_PATTERN = re.compile(
    r"^\s*"
    r"(?P<date>\d{4}-\d{2}-\d{2}\s+)?"
    r"(?P<timestamp>\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
    r"\s+"
    r"\[(?P<level>[A-Za-z]+)\s*\]"
    r"\s+"
    r"(?P<component>\S+)"
    r"\s*"
    r"(?P<message>.*)"
    r"$"
)


# ---------------------------------------------------------
# Supported log levels
# ---------------------------------------------------------

LEVEL_ALIASES = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARN": "WARN",
    "WARNING": "WARN",
    "ERROR": "ERROR",
    "CRIT": "CRIT",
    "CRITICAL": "CRIT",
    "FATAL": "FATAL",
    "ALERT": "ALERT",
    "EMERG": "EMERG",
}


# ---------------------------------------------------------
# High severity levels
#
# These levels must be included in the SLM's
# important/error events.
#
# IMPORTANT:
# This is also imported by the context builder so that
# the same severity definition is used everywhere.
# ---------------------------------------------------------

HIGH_SEVERITY_LEVELS = {
    "ERROR",
    "CRIT",
    "FATAL",
    "ALERT",
    "EMERG",
}


# ---------------------------------------------------------
# Extract one specific key=value field
# ---------------------------------------------------------

def extract_key_value(text, key):

    pattern = rf"\b{re.escape(key)}=([^\s]+)"

    match = re.search(
        pattern,
        text
    )

    if match:
        return match.group(1)

    return None


# ---------------------------------------------------------
# Extract all key=value fields
#
# Examples:
#
# cause=RADIO_LINK_FAILURE
# latency_ms=812
# error_code=PFCP-0007
# target=UDM
# ---------------------------------------------------------

def extract_key_values(text):

    pattern = (
        r"\b"
        r"([A-Za-z_][A-Za-z0-9_]*)"
        r"="
        r"([^\s]+)"
    )

    matches = re.findall(
        pattern,
        text
    )

    return {
        key: value
        for key, value in matches
    }


# ---------------------------------------------------------
# Extract IP addresses
# ---------------------------------------------------------

def extract_ip_addresses(text):

    pattern = (
        r"\b"
        r"(?:\d{1,3}\.){3}"
        r"\d{1,3}"
        r"\b"
    )

    return re.findall(
        pattern,
        text
    )


# ---------------------------------------------------------
# Extract protocol
# ---------------------------------------------------------

def extract_protocol(text):

    protocols = [
        "GTP-U",
        "HTTPS",
        "PFCP",
        "NGAP",
        "GTP",
        "HTTP",
        "SCTP",
        "TCP",
        "UDP",
        "NAS",
        "RRC",
        "N1",
        "N2",
        "N3",
        "N4",
        "N6",
        "N11",
        "N12",
        "N13",
        "N14",
        "N15",
        "N20",
        "N21",
    ]

    upper_text = text.upper()

    for protocol in protocols:

        if protocol in upper_text:
            return protocol

    return None


# ---------------------------------------------------------
# Parse one log line
# ---------------------------------------------------------

def parse_log_line(line):

    if not line:
        return None

    raw_line = line.rstrip(
        "\r\n"
    )

    match = LOG_LINE_PATTERN.match(
        raw_line
    )

    # -----------------------------------------------------
    # Header / footer / non-log line
    # -----------------------------------------------------

    if not match:
        return None

    # -----------------------------------------------------
    # Timestamp
    #
    # Preserve full date + time.
    #
    # Example:
    #
    # 2026-09-17 09:14:50.900
    # -----------------------------------------------------

    date = match.group(
        "date"
    )

    time = match.group(
        "timestamp"
    )

    if date:

        timestamp = (
            date.strip()
            + " "
            + time
        )

    else:

        timestamp = time

    # -----------------------------------------------------
    # Log level
    # -----------------------------------------------------

    raw_level = match.group(
        "level"
    ).upper()

    log_level = LEVEL_ALIASES.get(
        raw_level
    )

    if log_level is None:
        return None

    # -----------------------------------------------------
    # Component
    #
    # The component is the token immediately after
    # the log level.
    #
    # Examples:
    #
    # AMF
    # SMF
    # NRF
    # UDM
    # gNB-256
    # -----------------------------------------------------

    component = match.group(
        "component"
    )

    # -----------------------------------------------------
    # Message
    # -----------------------------------------------------

    message = match.group(
        "message"
    ).strip()

    # -----------------------------------------------------
    # Extract all key=value fields
    # -----------------------------------------------------

    fields = extract_key_values(
        message
    )

    # -----------------------------------------------------
    # Explicit important fields
    # -----------------------------------------------------

    error_code = fields.get(
        "error_code"
    )

    target = fields.get(
        "target"
    )

    msg_name = fields.get(
        "msg"
    )

    action = fields.get(
        "action"
    )

    # -----------------------------------------------------
    # Component identifier
    #
    # Example:
    #
    # gNB-256
    #
    # becomes:
    #
    # component      = gNB-256
    # component_type = gNB
    # component_id   = 256
    # -----------------------------------------------------

    component_type = component

    component_id = None

    component_match = re.match(
        r"^([A-Za-z]+)-(\d+)$",
        component
    )

    if component_match:

        component_type = (
            component_match.group(1)
        )

        component_id = (
            component_match.group(2)
        )

    # -----------------------------------------------------
    # Protocol
    # -----------------------------------------------------

    protocol = extract_protocol(
        message
    )

    # -----------------------------------------------------
    # IP addresses
    # -----------------------------------------------------

    ip_addresses = extract_ip_addresses(
        message
    )

    # -----------------------------------------------------
    # Return parsed structure
    # -----------------------------------------------------

    return {

        # Full timestamp
        "timestamp": timestamp,

        # Normalized log level
        "log_level": log_level,

        # Component
        "component": component,

        "component_type": component_type,

        "component_id": component_id,

        # Protocol
        "protocol": protocol,

        # IP addresses
        "ip_addresses": ip_addresses,

        # Explicit fields
        "error_code": error_code,

        "target": target,

        "msg": msg_name,

        "action": action,

        # All key=value fields
        "fields": fields,

        # Message without timestamp/level/component
        "message": message,

        # Original complete log line
        "raw_log": raw_line,

        "raw_line": raw_line
    }


# ---------------------------------------------------------
# Parse multiple lines
# ---------------------------------------------------------

def parse_lines(lines):

    records = []

    skipped_lines = 0

    for line in lines:

        parsed = parse_log_line(
            line
        )

        if parsed is None:

            skipped_lines += 1

            continue

        records.append(
            parsed
        )

    return records, skipped_lines