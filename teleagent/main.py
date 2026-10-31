"""
main.py

Entry point for TeleAgent.

Run using:
python -m teleagent.main

Author : Sapna Jha
Version : 0.3
"""

from teleagent.config import (
    MODEL_NAME,
    ADAPTER_PATHS,
    DEFAULT_ADAPTER,
    KNOWLEDGE_BASE_PATH,
)

from teleagent.reasoning.model_manager import ModelManager
from teleagent.reasoning.reasoning_service import ReasoningService

from teleagent.coordinator.coordinator import Coordinator
from teleagent.coordinator.intent_analyzer import IntentAnalyzer
from teleagent.coordinator.slm_intent_classifier import (
    SLMIntentClassifier,
)
from teleagent.coordinator.task_planner import TaskPlanner
from teleagent.coordinator.dispatcher import Dispatcher
from teleagent.coordinator.aggregator import Aggregator
from teleagent.coordinator.agent_registry import AgentRegistry

from teleagent.agents.network_agent import NetworkAgent
from teleagent.agents.log_agent import LogAgent
from teleagent.agents.monitoring_agent import MonitoringAgent
from teleagent.agents.knowledge_agent import KnowledgeAgent

from teleagent.rag.rag_service import RAGService
from teleagent.retrieval.intelligent_retrieval_service import (
    IntelligentRetrievalService,
)

from teleagent.language.translator import BoundaryTranslator


# ---------------------------------------------------------
# Banner
# ---------------------------------------------------------

def print_banner():
    print("=" * 70)
    print("TeleAgent")
    print("Private 5G Multi-Agent AI Platform")
    print("=" * 70)


# ---------------------------------------------------------
# Initialize Reasoning Service
# ---------------------------------------------------------

def initialize_reasoning_service() -> ReasoningService:
    """Load the Telecom SLM and its LoRA adapters."""

    print("\nLoading Telecom SLM...\n")

    manager = ModelManager(
        model_name=MODEL_NAME,
        adapter_paths=ADAPTER_PATHS,
        default_adapter=DEFAULT_ADAPTER,
    )

    manager.load()

    print("\n✓ Model Loaded\n")
    manager.print_model_info()

    return ReasoningService(manager)


# ---------------------------------------------------------
# Initialize Coordinator
# ---------------------------------------------------------

def initialize_coordinator(
    reasoning_service: ReasoningService,
) -> Coordinator:
    """Initialize the TeleAgent coordinator."""

    # Intent classification
    classifier = SLMIntentClassifier(reasoning_service)

    intent_analyzer = IntentAnalyzer(
        classifier=classifier
    )

    # Specialized agents
    network_agent = NetworkAgent(reasoning_service)
    log_agent = LogAgent(reasoning_service)
    monitoring_agent = MonitoringAgent(reasoning_service)

    rag_service = RAGService(
        knowledge_base_directory=KNOWLEDGE_BASE_PATH
    )

    retrieval_service = IntelligentRetrievalService(
        rag_service=rag_service
    )

    knowledge_agent = KnowledgeAgent(
        reasoning_service=reasoning_service,
        retrieval_service=retrieval_service,
    )

    # Agent registry
    registry = AgentRegistry()

    registry.register("network", network_agent)
    registry.register("log", log_agent)
    registry.register("monitoring", monitoring_agent)
    registry.register("knowledge", knowledge_agent)

    # Dispatcher
    dispatcher = Dispatcher(registry)

    # Coordinator
    coordinator = Coordinator(
        intent_analyzer=intent_analyzer,
        task_planner=TaskPlanner(),
        dispatcher=dispatcher,
        aggregator=Aggregator(),
    )

    return coordinator


# ---------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------

def interactive_chat(
    coordinator: Coordinator,
    translator: BoundaryTranslator,
):
    """Interactive CLI with Hindi-to-English translation."""

    print("\nType 'exit' to quit.\n")

    while True:
        query = input("User > ").strip()

        if query.lower() in ["exit", "quit"]:
            print("\nGoodbye.\n")
            break

        if not query:
            continue

        # -------------------------------------------------
        # Translate input to English when necessary
        # -------------------------------------------------

        try:
            english_query, reply_in_hindi = (
                translator.prepare_input(query)
            )

            print(
                f"\n[Debug] Hindi response requested: "
                f"{reply_in_hindi}"
            )

            if english_query != query:
                print(
                    f"[Debug] English query: {english_query}"
                )

        except Exception as error:
            print(f"\nInput translation failed: {error}")
            print("Please try again.\n")
            continue

        # -------------------------------------------------
        # Execute query using the existing coordinator
        # -------------------------------------------------

        try:
            response = coordinator.execute(english_query)

        except Exception as error:
            print(f"\nTeleAgent execution failed: {error}\n")
            continue

        # -------------------------------------------------
        # Read and validate the coordinator response
        # -------------------------------------------------

        original_response = getattr(
            response,
            "response",
            "",
        )

        if not isinstance(original_response, str):
            original_response = str(original_response or "")

        original_response = original_response.strip()

        print(
            f"[Debug] Coordinator response: "
            f"{original_response!r}"
        )

        if not original_response:
            print(
                "\nTeleAgent returned an empty response. "
                "The issue is upstream of output translation."
            )
            print("Check the agent and aggregator response.\n")
            continue

        # -------------------------------------------------
        # Translate final response when necessary
        # -------------------------------------------------

        final_response = original_response

        if reply_in_hindi:
            try:
                print("\nTranslating response into Hindi...")

                # Remove decorative headings and separator lines
                # from the aggregated response before translation.
                lines = original_response.splitlines()

                cleaned_lines = [
                    line.strip()
                    for line in lines
                    if line.strip()
                    and not all(
                        char == "="
                        for char in line.strip()
                    )
                    and line.strip() not in {
                        "Knowledge Response",
                        "Network Analysis",
                        "Monitoring Analysis",
                        "Log Analysis",
                    }
                ]

                answer_text = "\n".join(cleaned_lines).strip()

                print(
                    "[Debug] Text sent for translation:",
                    repr(answer_text),
                )

                translated_response = translator.translate_output(
                    answer_text,
                    reply_in_hindi=True,
                )

                print(
                    "[Debug] Translated response:",
                    repr(translated_response),
                )

                # Accept only non-empty output containing letters.
                # Otherwise, display the English answer.
                if (
                    isinstance(translated_response, str)
                    and translated_response.strip()
                    and any(
                        char.isalpha()
                        for char in translated_response
                    )
                ):
                    final_response = translated_response.strip()
                else:
                    print(
                        "[Warning] Translation produced unusable "
                        "output. Displaying English instead."
                    )
                    final_response = answer_text

            except Exception as error:
                print(
                    f"\nOutput translation failed: {error}"
                )
                print(
                    "Displaying the original English response instead."
                )
                final_response = original_response

        # -------------------------------------------------
        # Final response
        # -------------------------------------------------

        print("\nTeleAgent >\n")

        if (
            isinstance(final_response, str)
            and final_response.strip()
        ):
            print(final_response.strip())
        else:
            print(
                "[Warning] Final response is empty. "
                "Displaying the original response."
            )
            print(original_response)

        # -------------------------------------------------
        # Metadata
        # -------------------------------------------------

        print("\n---------------- Metadata ----------------")

        print(f"Intent : {response.intent}")
        print(f"Tasks  : {len(response.tasks)}")

        print("\nExecuted Agents:")

        for result in response.results:
            metadata = result.response.metadata

            print(f"\n• {result.agent}")

            print(
                f"  Task ID         : "
                f"{result.task.task_id}"
            )

            print(
                f"  Inference Time  : "
                f"{metadata.inference_time_ms:.2f} ms"
            )

            print(
                f"  Tokens          : "
                f"{metadata.tokens_generated}"
            )

            print(
                f"  Model           : "
                f"{metadata.model_name}"
            )

            print(
                f"  LoRA Enabled    : "
                f"{metadata.lora_enabled}"
            )

            print(
                f"  Adapter         : "
                f"{metadata.adapter_name or 'None'}"
            )

        print("------------------------------------------\n")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    print_banner()

    reasoning_service = initialize_reasoning_service()

    coordinator = initialize_coordinator(
        reasoning_service
    )

    # Load the translator once at startup.
    print("\nLoading Translator...\n")

    translator = BoundaryTranslator()

    print("\n✓ Translator Loaded\n")

    interactive_chat(
        coordinator,
        translator,
    )


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
