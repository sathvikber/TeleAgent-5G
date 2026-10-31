import json
import os

from .correlator import (
    generate_summary,
    create_event_timeline
)

from .context_builder import (
    build_compact_context
)


def create_slm_context(
    records,
    source_files
):

    summary = generate_summary(records)

    timeline = create_event_timeline(records)

    compact_context = build_compact_context(
        records,
        summary
    )

    return {
        "task":
            "5G network log analysis",

        "source_files":
            source_files,

        "summary":
            summary,

        "event_timeline":
            timeline,

        "compact_context":
            compact_context,

        "structured_logs":
            [
                record.__dict__
                if hasattr(record, "__dict__")
                else record
                for record in records
            ]
    }


def create_prompt(context):

    compact = context[
        "compact_context"
    ]

    return f"""
You are an expert 5G network
troubleshooting assistant.

Analyze the following network evidence.

SOURCE FILES:
{json.dumps(
    context["source_files"],
    indent=2
)}

TOTAL EVENTS:
{compact["total_events"]}

AFFECTED COMPONENTS:
{json.dumps(
    compact["affected_components"],
    indent=2
)}

ERROR COUNT:
{compact["error_count"]}

IMPORTANT EVENTS:
{json.dumps(
    compact["important_events"],
    indent=2
)}

ERROR SUMMARY:
{json.dumps(
    compact["error_summary"],
    indent=2
)}

Provide the analysis in this structure:

1. Root Cause
2. Affected Component
3. Evidence
4. Impact
5. Recommended Action

Only use evidence present in the
provided logs.

Do not invent missing information.
"""


def save_json(data, path):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            default=str
        )