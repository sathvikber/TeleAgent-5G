from collections import Counter


# ---------------------------------------------------------
# High-severity log levels
#
# These are treated as failure/critical events for
# troubleshooting summaries.
#
# The parser normalizes:
# CRITICAL -> CRIT
# ---------------------------------------------------------

HIGH_SEVERITY_LEVELS = {
    "ERROR",
    "CRIT",
    "FATAL",
    "ALERT",
    "EMERG",
}


# ---------------------------------------------------------
# Convert LogRecord to dictionary
# ---------------------------------------------------------

def record_to_dict(record):

    if hasattr(record, "__dict__"):

        return record.__dict__

    return record


# ---------------------------------------------------------
# Generate summary
# ---------------------------------------------------------

def generate_summary(records):

    component_count = Counter()

    level_count = Counter()

    protocol_count = Counter()

    errors = []

    # -----------------------------------------------------
    # Process every record
    # -----------------------------------------------------

    for record in records:

        data = record_to_dict(
            record
        )

        # -------------------------------------------------
        # Component count
        # -------------------------------------------------

        component = data.get(
            "component"
        )

        if component:

            component_count[
                component
            ] += 1

        # -------------------------------------------------
        # Log-level count
        # -------------------------------------------------

        level = data.get(
            "log_level"
        )

        if level:

            level_count[
                level
            ] += 1

        # -------------------------------------------------
        # Protocol count
        # -------------------------------------------------

        protocol = data.get(
            "protocol"
        )

        if protocol:

            protocol_count[
                protocol
            ] += 1

        # -------------------------------------------------
        # High-severity events
        #
        # Previously this only checked:
        #
        # ERROR / CRITICAL
        #
        # But the parser normalizes CRITICAL -> CRIT.
        #
        # Now we include:
        #
        # ERROR
        # CRIT
        # FATAL
        # ALERT
        # EMERG
        # -------------------------------------------------

        if level in HIGH_SEVERITY_LEVELS:

            errors.append({

                "timestamp":
                    data.get("timestamp"),

                "component":
                    component,

                "level":
                    level,

                "protocol":
                    protocol,

                "error_code":
                    data.get("error_code"),

                "message":
                    data.get("message"),

                "source_file":
                    data.get("source_file"),
            })

    # -----------------------------------------------------
    # Return summary
    # -----------------------------------------------------

    return {

        "total_events":
            len(records),

        "components":
            dict(component_count),

        "log_levels":
            dict(level_count),

        "protocols":
            dict(protocol_count),

        # High-severity events
        "errors":
            errors
    }


# ---------------------------------------------------------
# Create chronological event timeline
# ---------------------------------------------------------

def create_event_timeline(records):

    timeline = []

    for record in records:

        data = record_to_dict(
            record
        )

        timeline.append({

            "timestamp":
                data.get("timestamp"),

            "component":
                data.get("component"),

            "level":
                data.get("log_level"),

            "protocol":
                data.get("protocol"),

            "event":
                data.get("message"),

            "source_file":
                data.get("source_file")
        })

    return timeline