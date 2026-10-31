import pytest
from pydantic import ValidationError

from teleagent.reasoning.schemas import (
    ReasoningRequest,
    GenerationConfig,
)


def test_empty_query_validation():
    """An empty query should fail validation."""

    with pytest.raises(ValidationError):
        ReasoningRequest(
            task="log_analysis",
            query=""
        )


def test_default_generation_config():
    """Default generation parameters should be assigned."""

    request = ReasoningRequest(
        task="chat",
        query="Hello"
    )

    config = request.generation_config

    assert isinstance(config, GenerationConfig)
    assert config.temperature == 0.2
    assert config.top_p == 0.9
    assert config.top_k == 50
    assert config.max_new_tokens == 512
    assert config.do_sample is True
    assert config.repetition_penalty == 1.1
