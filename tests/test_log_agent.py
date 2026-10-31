from teleagent.agents.log_agent import LogAgent


class MockReasoningService:

    def generate(self, prompt):
        return "Log Response"


def test_log_agent():

    agent = LogAgent(MockReasoningService())

    response = agent.execute("Show AMF logs")

    assert response == "Log Response"
