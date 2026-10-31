from teleagent.agents.knowledge_agent import KnowledgeAgent
from teleagent.agents.log_agent import LogAgent
from teleagent.agents.monitoring_agent import MonitoringAgent
from teleagent.agents.network_agent import NetworkAgent

from teleagent.coordinator.agent_registry import AgentRegistry
from teleagent.coordinator.aggregator import Aggregator
from teleagent.coordinator.coordinator import Coordinator
from teleagent.coordinator.dispatcher import Dispatcher
from teleagent.coordinator.intent_analyzer import IntentAnalyzer
from teleagent.coordinator.task_planner import TaskPlanner


class DummyReasoningService:
    """
    Fake reasoning service used for unit testing.
    """

    def generate(self, prompt: str) -> str:
        return f"Dummy response:\n{prompt}"


def build_coordinator():

    reasoning = DummyReasoningService()

    registry = AgentRegistry()

    registry.register(
        "network",
        NetworkAgent(reasoning)
    )

    registry.register(
        "log",
        LogAgent(reasoning)
    )

    registry.register(
        "knowledge",
        KnowledgeAgent(reasoning)
    )

    registry.register(
        "monitoring",
        MonitoringAgent(reasoning)
    )

    dispatcher = Dispatcher(registry)

    return Coordinator(
        intent_analyzer=IntentAnalyzer(),
        task_planner=TaskPlanner(),
        dispatcher=dispatcher,
        aggregator=Aggregator(),
    )


def test_network_query():

    coordinator = build_coordinator()

    response = coordinator.execute(
        "Why is UE registration failing?"
    )

    assert "Dummy response" in response


def test_log_query():

    coordinator = build_coordinator()

    response = coordinator.execute(
        "Show AMF logs."
    )

    assert "Dummy response" in response


def test_monitoring_query():

    coordinator = build_coordinator()

    response = coordinator.execute(
        "Monitor CPU utilization."
    )

    assert "Dummy response" in response


def test_knowledge_query():

    coordinator = build_coordinator()

    response = coordinator.execute(
        "Explain PFCP Session Establishment."
    )

    assert "Dummy response" in response


def test_unknown_query_defaults_to_knowledge():

    coordinator = build_coordinator()

    response = coordinator.execute(
        "What is 5G?"
    )

    assert "Dummy response" in response
