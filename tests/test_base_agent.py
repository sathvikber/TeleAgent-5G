import pytest

from teleagent.agents.base_agent import BaseAgent


class DummyAgent(BaseAgent):

    def execute(self, request):
        return f"Received: {request}"


def test_base_agent_execution():

    agent = DummyAgent()

    response = agent.execute("Hello")

    assert response == "Received: Hello"


def test_base_agent_cannot_be_instantiated():

    with pytest.raises(TypeError):
        BaseAgent()
