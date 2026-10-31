from teleagent.logs.models import LogEntry
from teleagent.logs.pattern_detector import PatternDetector


def test_registration_pattern():

    detector = PatternDetector()

    entries = [

        LogEntry(
            message="Registration Failed"
        ),

        LogEntry(
            message="Registration Failed"
        )

    ]

    patterns = detector.detect(entries)

    assert len(patterns) == 1

    assert patterns[0].name == "Registration Failure"


def test_timeout_pattern():

    detector = PatternDetector()

    entries = [

        LogEntry(
            message="Authentication Timeout"
        )

    ]

    patterns = detector.detect(entries)

    assert any(
        p.name == "Timeout"
        for p in patterns
    )


def test_no_pattern():

    detector = PatternDetector()

    patterns = detector.detect([

        LogEntry(
            message="Everything OK"
        )

    ])

    assert patterns == []
