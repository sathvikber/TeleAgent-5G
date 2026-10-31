"""
Tests for intent_classifier.py
"""

import pytest

from teleagent.coordinator.intent_classifier import IntentClassifier


class DummyClassifier(IntentClassifier):

    def classify(self, query: str) -> str:
        return "network"


class InvalidClassifier(IntentClassifier):
    pass


# ---------------------------------------------------------
# Interface
# ---------------------------------------------------------

def test_dummy_classifier_returns_intent():

    classifier = DummyClassifier()

    assert classifier.classify("Show AMF status") == "network"


def test_cannot_instantiate_abstract_classifier():

    with pytest.raises(TypeError):
        IntentClassifier()


def test_subclass_without_classify_cannot_be_instantiated():

    with pytest.raises(TypeError):
        InvalidClassifier()
