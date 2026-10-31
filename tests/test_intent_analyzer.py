"""
Tests for IntentAnalyzer.
"""

from teleagent.coordinator.intent_analyzer import IntentAnalyzer
from teleagent.coordinator.intent_classifier import IntentClassifier


class FakeClassifier(IntentClassifier):

    def classify(self, query: str) -> str:
        return "monitoring"


# ---------------------------------------------------------
# Rule-based tests
# ---------------------------------------------------------

def test_network_rule():

    analyzer = IntentAnalyzer()

    assert (
        analyzer.analyze(
            "AMF registration and UPF session failure"
        )
        == "network"
    )


def test_log_rule():

    analyzer = IntentAnalyzer()

    assert (
        analyzer.analyze(
            "Show debug log and error trace"
        )
        == "log"
    )


def test_knowledge_rule():

    analyzer = IntentAnalyzer()

    assert (
        analyzer.analyze(
            "What is the NGAP protocol?"
        )
        == "knowledge"
    )


def test_monitoring_rule():

    analyzer = IntentAnalyzer()

    assert (
        analyzer.analyze(
            "CPU memory utilization"
        )
        == "monitoring"
    )


# ---------------------------------------------------------
# Hybrid behaviour
# ---------------------------------------------------------

def test_slm_used_when_score_below_threshold():

    analyzer = IntentAnalyzer(
        classifier=FakeClassifier()
    )

    #
    # Only one keyword ("AMF")
    #
    assert (
        analyzer.analyze(
            "AMF issue"
        )
        == "monitoring"
    )


def test_rule_overrides_slm():

    analyzer = IntentAnalyzer(
        classifier=FakeClassifier()
    )

    #
    # Multiple network keywords
    #
    assert (
        analyzer.analyze(
            "AMF registration UPF session"
        )
        == "network"
    )


def test_unknown_without_classifier():

    analyzer = IntentAnalyzer()

    assert (
        analyzer.analyze(
            "Something strange happened"
        )
        == "knowledge"
    )


def test_unknown_with_classifier():

    analyzer = IntentAnalyzer(
        classifier=FakeClassifier()
    )

    assert (
        analyzer.analyze(
            "Something strange happened"
        )
        == "monitoring"
    )


def test_custom_threshold():

    analyzer = IntentAnalyzer(
        confidence_threshold=1,
    )

    #
    # One keyword is enough
    #
    assert (
        analyzer.analyze(
            "AMF issue"
        )
        == "network"
    )
