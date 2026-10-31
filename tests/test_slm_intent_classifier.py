"""
Tests for SLMIntentClassifier.
"""

from unittest.mock import Mock

from teleagent.coordinator.slm_intent_classifier import (
    SLMIntentClassifier,
)

from teleagent.reasoning.schemas import (
    ModelMetadata,
    ReasoningResponse,
)


def create_response(text: str) -> ReasoningResponse:

    metadata = ModelMetadata(
        model_name="Qwen",
        model_type="lora",
        tokenizer_name="Qwen",
        lora_enabled=True,
        device="cpu",
    )

    return ReasoningResponse(
        success=True,
        response=text,
        metadata=metadata,
    )


def test_network_intent():

    reasoning_service = Mock()

    reasoning_service.generate.return_value = create_response(
        '{"intent":"network"}'
    )

    classifier = SLMIntentClassifier(
        reasoning_service
    )

    assert (
        classifier.classify(
            "AMF registration failure"
        )
        == "network"
    )


def test_monitoring_intent():

    reasoning_service = Mock()

    reasoning_service.generate.return_value = create_response(
        '{"intent":"monitoring"}'
    )

    classifier = SLMIntentClassifier(
        reasoning_service
    )

    assert (
        classifier.classify(
            "CPU utilization"
        )
        == "monitoring"
    )


def test_invalid_json_returns_knowledge():

    reasoning_service = Mock()

    reasoning_service.generate.return_value = create_response(
        "this is not json"
    )

    classifier = SLMIntentClassifier(
        reasoning_service
    )

    assert (
        classifier.classify(
            "random query"
        )
        == "knowledge"
    )


def test_unknown_intent_returns_knowledge():

    reasoning_service = Mock()

    reasoning_service.generate.return_value = create_response(
        '{"intent":"security"}'
    )

    classifier = SLMIntentClassifier(
        reasoning_service
    )

    assert (
        classifier.classify(
            "firewall issue"
        )
        == "knowledge"
    )


def test_missing_intent_returns_knowledge():

    reasoning_service = Mock()

    reasoning_service.generate.return_value = create_response(
        '{"category":"network"}'
    )

    classifier = SLMIntentClassifier(
        reasoning_service
    )

    assert (
        classifier.classify(
            "AMF failure"
        )
        == "knowledge"
    )


def test_prompt_contains_query():

    prompt = SLMIntentClassifier._build_prompt(
        "AMF registration failed"
    )

    assert "AMF registration failed" in prompt
    assert "network" in prompt
    assert "monitoring" in prompt
    assert "knowledge" in prompt

def test_intent_is_case_insensitive():

    reasoning_service = Mock()

    reasoning_service.generate.return_value = create_response(
        '{"intent":"Network"}'
    )

    classifier = SLMIntentClassifier(
        reasoning_service
    )

    assert (
        classifier.classify(
            "AMF failure"
        )
        == "network"
    )


def test_intent_whitespace():

    reasoning_service = Mock()

    reasoning_service.generate.return_value = create_response(
        '{ "intent" : "  monitoring  " }'
    )

    classifier = SLMIntentClassifier(
        reasoning_service
    )

    assert (
        classifier.classify(
            "CPU utilization"
        )
        == "monitoring"
    )
