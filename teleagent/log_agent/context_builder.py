from .correlator import record_to_dict


# ---------------------------------------------------------
# High-severity levels
#
# These events are treated as important failures/events
# for the SLM.
#
# The parser normalizes:
# CRITICAL -> CRIT
# WARNING  -> WARN
# ---------------------------------------------------------

IMPORTANT_LEVELS = {
    "ERROR",
    "CRIT",
    "FATAL",
    "ALERT",
    "EMERG"
}


# ---------------------------------------------------------
# Warning levels
#
# WARN is kept separate from high-severity errors.
# ---------------------------------------------------------

WARNING_LEVELS = {
    "WARN"
}


# ---------------------------------------------------------
# Build compact context for SLM
# ---------------------------------------------------------

def build_compact_context(
    records,
    summary,
    max_events=40
):

    important = []

    warnings = []

    # -----------------------------------------------------
    # Process all parsed records
    # -----------------------------------------------------

    for record in records:

        data = record_to_dict(
            record
        )

        level = data.get(
            "log_level",
            "UNKNOWN"
        )

        # -------------------------------------------------
        # High-severity events
        # -------------------------------------------------

        if level in IMPORTANT_LEVELS:

            important.append({

                "timestamp":
                    data.get("timestamp"),

                "component":
                    data.get("component"),

                "level":
                    level,

                "protocol":
                    data.get("protocol"),

                "error_code":
                    data.get("error_code"),

                "message":
                    data.get("message"),

                "source_file":
                    data.get("source_file")
            })

        # -------------------------------------------------
        # Warning events
        # -------------------------------------------------

        elif level in WARNING_LEVELS:

            warnings.append({

                "timestamp":
                    data.get("timestamp"),

                "component":
                    data.get("component"),

                "level":
                    level,

                "protocol":
                    data.get("protocol"),

                "error_code":
                    data.get("error_code"),

                "message":
                    data.get("message"),

                "source_file":
                    data.get("source_file")
            })

    # -----------------------------------------------------
    # Limit context size
    # -----------------------------------------------------

    important = important[:max_events]

    warnings = warnings[:max_events]

    # -----------------------------------------------------
    # Components
    # -----------------------------------------------------

    components = list(
        summary["components"].keys()
    )

    # -----------------------------------------------------
    # Return compact SLM context
    # -----------------------------------------------------

    return {

        "task":
            "5G network troubleshooting",

        "total_events":
            summary["total_events"],

        "affected_components":
            components,

        # IMPORTANT:
        # Count high-severity events directly from records.
        #
        # For current PDF:
        # ERROR = 7
        # CRIT  = 1
        # TOTAL = 8
        #
        "error_count":
            len(important),

        # -------------------------------------------------
        # High-severity events
        # -------------------------------------------------

        "important_events":
            important,

        # -------------------------------------------------
        # Backward-compatible error summary
        #
        # Previously this used summary["errors"], which
        # contained only ERROR records.
        #
        # Now it contains ERROR + CRIT + FATAL + ALERT
        # + EMERG events.
        # -------------------------------------------------

        "error_summary":
            important,

        # -------------------------------------------------
        # Warnings
        # -------------------------------------------------

        "warnings":
            warnings
    }