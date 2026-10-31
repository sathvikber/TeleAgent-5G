from teleagent.logs.models import LogEntry
from teleagent.logs.error_extractor import ErrorExtractor


def test_extract_errors():

    extractor = ErrorExtractor()

    entries = [

        LogEntry(
            severity="INFO"
        ),

        LogEntry(
            severity="ERROR"
        ),

        LogEntry(
            severity="WARNING"
        )

    ]

    important = extractor.extract(entries)

    assert len(important) == 2


def test_empty():

    extractor = ErrorExtractor()

    assert extractor.extract([]) == []
