from teleagent.logs.models import LogEntry
from teleagent.logs.event_classifier import EventClassifier


def test_registration_classification():

    classifier = EventClassifier()

    entries = [

        LogEntry(
            message="UE Registration Failed"
        )

    ]

    categories = classifier.classify(entries)

    assert "Registration" in categories


def test_authentication_classification():

    classifier = EventClassifier()

    entries = [

        LogEntry(
            message="Authentication Timeout"
        )

    ]

    categories = classifier.classify(entries)

    assert "Authentication" in categories


def test_unknown_classification():

    classifier = EventClassifier()

    entries = [

        LogEntry(
            message="abcdefg"
        )

    ]

    categories = classifier.classify(entries)

    assert "Unknown" in categories
