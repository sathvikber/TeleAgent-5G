from teleagent.reasoning.prompt_builder import PromptBuilder
from teleagent.reasoning.schemas import (
    Message,
    ReasoningRequest,
)


def test_prompt_builder_default_system_prompt():

    builder = PromptBuilder()

    request = ReasoningRequest(
        task="chat",
        query="Hello"
    )

    messages = builder.build(request)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "TeleAgent" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello"


def test_prompt_builder_custom_system_prompt():

    builder = PromptBuilder()

    request = ReasoningRequest(
        task="chat",
        query="Hello",
        system_prompt="Custom Prompt"
    )

    messages = builder.build(request)

    assert messages[0]["content"] == "Custom Prompt"


def test_prompt_builder_conversation():

    builder = PromptBuilder()

    request = ReasoningRequest(
        task="chat",
        query="How are you?",
        conversation=[
            Message(
                role="user",
                content="Hello"
            ),
            Message(
                role="assistant",
                content="Hi!"
            )
        ]
    )

    messages = builder.build(request)

    assert len(messages) == 4
    assert messages[1]["content"] == "Hello"
    assert messages[2]["content"] == "Hi!"
    assert messages[3]["content"] == "How are you?"


def test_prompt_builder_context():

    builder = PromptBuilder()

    request = ReasoningRequest(
        task="interpret_log",
        query="Interpret this log",
        context="AMF Registration Failure"
    )

    messages = builder.build(request)

    assert "Additional Context" in messages[1]["content"]
    assert "AMF Registration Failure" in messages[1]["content"]
    assert messages[-1]["content"] == "Interpret this log"
