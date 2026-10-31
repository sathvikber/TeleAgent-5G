from teleagent.reasoning.schemas import (
    ReasoningRequest,
    GenerationConfig,
)


def test_reasoning_request_creation():
    request = ReasoningRequest(
        task="log_analysis",
        query="AMF Registration Failure"
    )

    assert request.task == "log_analysis"
    assert request.query == "AMF Registration Failure"

    assert request.system_prompt is None
    assert request.context is None
    assert request.conversation == []

    assert isinstance(
        request.generation_config,
        GenerationConfig
    )
