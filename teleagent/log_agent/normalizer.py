from .parser import parse_log_line
from .schemas import LogRecord


def normalize_txt(text, source_file):

    records = []

    lines = text.splitlines()

    print(
        f"\n[NORMALIZER] {source_file}"
    )

    print(
        "Input lines:",
        len(lines)
    )

    for line_number, line in enumerate(
        lines,
        start=1
    ):

        original_line = line

        line = line.strip()

        if not line:
            continue

        parsed = parse_log_line(
            line
        )

        # Debug first 10 non-empty lines
        if line_number <= 10:

            print(
                f"[LINE {line_number}]",
                repr(line)
            )

            print(
                "  PARSED:",
                parsed
            )

        if parsed is None:
            continue

        record = LogRecord(
            source_file=source_file,
            line_number=line_number,

            timestamp=parsed["timestamp"],
            component=parsed["component"],
            log_level=parsed["log_level"],
            protocol=parsed["protocol"],

            ip_addresses=parsed["ip_addresses"],

            error_code=parsed["error_code"],

            event_type="LOG_EVENT",

            message=parsed["message"],

            raw_log=parsed["raw_log"]
        )

        records.append(
            record
        )

    print(
        "[NORMALIZER] Records:",
        len(records)
    )

    return records


def normalize_csv(data, source_file):

    records = []

    for index, row in enumerate(
        data["records"],
        start=1
    ):

        text = " | ".join(
            f"{key}={value}"
            for key, value in row.items()
        )

        parsed = parse_log_line(text)

        # Skip rows that cannot be interpreted
        # as log events.
        if parsed is None:
            continue

        record = LogRecord(
            source_file=source_file,
            line_number=index,

            timestamp=parsed["timestamp"],
            component=parsed["component"],
            log_level=parsed["log_level"],
            protocol=parsed["protocol"],

            ip_addresses=parsed["ip_addresses"],

            error_code=parsed["error_code"],

            event_type="CSV_EVENT",

            message=text,

            raw_log=row
        )

        records.append(record)

    return records