import teleagent.config as config

from unittest.mock import MagicMock

from teleagent.agents.knowledge_agent import KnowledgeAgent
from teleagent.reasoning.schemas import (
    ModelMetadata,
    ReasoningResponse,
)


def mock_response(answer: str) -> ReasoningResponse:

    return ReasoningResponse(
        success=True,
        response=answer,
        metadata=ModelMetadata(
            model_name="Qwen2.5-1.5B",
            model_type="lora",
            tokenizer_name="Qwen",
            lora_enabled=True,
            device="cuda",
        ),
    )


def test_execute():

    reasoning = MagicMock()
    rag = MagicMock()

    rag.retrieve_text.return_value = (
        "AMF handles registration.\n"
        "AMF manages mobility."
    )

    reasoning.generate.return_value = mock_response(
        "AMF manages UE registration and mobility."
    )

    agent = KnowledgeAgent(
        reasoning_service=reasoning,
        rag_service=rag,
    )

    response = agent.execute(
        "What is AMF?"
    )

    rag.retrieve_text.assert_called_once_with(
        query="What is AMF?",
        top_k=config.RAG_TOP_K,
    )

    assert response.response == \
        "AMF manages UE registration and mobility."


def test_empty_context():

    reasoning = MagicMock()
    rag = MagicMock()

    rag.retrieve_text.return_value = ""

    reasoning.generate.return_value = mock_response(
        "The information is not available in the knowledge base."
    )

    agent = KnowledgeAgent(
        reasoning_service=reasoning,
        rag_service=rag,
    )

    response = agent.execute(
        "Unknown telecom topic"
    )

    assert response.response == \
        "The information is not available in the knowledge base."


def test_prompt_contains_context():

    reasoning = MagicMock()
    rag = MagicMock()

    rag.retrieve_text.return_value = \
        "AMF authenticates UE."

    reasoning.generate.return_value = mock_response(
        "AMF authenticates the UE."
    )

    agent = KnowledgeAgent(
        reasoning_service=reasoning,
        rag_service=rag,
    )

    agent.execute(
        "What is AMF?"
    )

    request = reasoning.generate.call_args[0][0]

    assert request.task == "knowledge"

    assert "AMF authenticates UE." in request.query

    assert "What is AMF?" in request.query

    assert "Retrieved Knowledge" in request.query

    assert "Question" in request.query


def test_top_k():

    reasoning = MagicMock()
    rag = MagicMock()

    rag.retrieve_text.return_value = "Context"

    reasoning.generate.return_value = mock_response(
        "Answer"
    )

    agent = KnowledgeAgent(
        reasoning_service=reasoning,
        rag_service=rag,
    )

    agent.execute(
        "Explain AMF"
    )

    rag.retrieve_text.assert_called_once_with(
        query="Explain AMF",
        top_k=config.RAG_TOP_K,
    )
