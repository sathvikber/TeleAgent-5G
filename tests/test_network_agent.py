from teleagent.agents.network_agent import NetworkAgent


class MockReasoningService:

    def generate(self, prompt):
        return "Network Response"


def test_network_agent():

    agent = NetworkAgent(MockReasoningService())

    response = agent.execute("UE attach failed")

    assert response == "Network Response"
