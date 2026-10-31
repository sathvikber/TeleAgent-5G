"""
Network Prompt Builder

Builds a structured prompt for the Telecom SLM using the
results of the network analysis pipeline.
"""

from teleagent.network.models import NetworkAnalysisResult


class NetworkPromptBuilder:
    """
    Builds prompts for telecom network troubleshooting.
    """

    def build(
        self,
        user_query: str,
        analysis: NetworkAnalysisResult,
    ) -> str:
        """
        Build a structured prompt.
        """

        prompt = []

        # --------------------------------------------------
        # System Role
        # --------------------------------------------------

        prompt.append(
            "You are an expert 5G Core Network Troubleshooting Assistant."
        )

        prompt.append(
            "Analyze the structured network information provided below."
        )

        prompt.append(
            "Use ONLY the provided information. "
            "Do not invent network functions, protocols, alarms, "
            "or failures."
        )

        prompt.append("")

        # --------------------------------------------------
        # User Query
        # --------------------------------------------------

        prompt.append("User Request:")
        prompt.append(user_query)
        prompt.append("")

        # --------------------------------------------------
        # Real-Time Topology State (Included when present)
        # --------------------------------------------------

        if analysis.topology_context:

            prompt.append("Real-Time Network Topology State:")

            topo = analysis.topology_context

            prompt.append(f"- Active Nodes : {', '.join(topo.get('active_nodes', []))}")

            if topo.get("degraded_links"):

                prompt.append("- Degraded/Down Links:")

                for link in topo.get("degraded_links"):

                    prompt.append(f"  * {link}")

            else:

                prompt.append("- Degraded/Down Links: None")

            if topo.get("recent_state_changes"):

                prompt.append("- Recent Connection Updates/Changes:")

                for event in topo.get("recent_state_changes"):

                    prompt.append(
                        f"  * [{event.get('timestamp')}] Link {event.get('link_id')} "
                        f"({event.get('interface')}): {event.get('old_status')} -> {event.get('new_status')} "
                        f"(Reason: {event.get('reason', 'N/A')})"
                    )

            prompt.append("")

        # --------------------------------------------------
        # Diagnostic Evidence (Included when present)
        # --------------------------------------------------

        if analysis.diagnostic_evidence:

            prompt.append("Live Diagnostic Verification:")

            for ip, status in analysis.diagnostic_evidence.items():

                prompt.append(f"- Node IP {ip} : {status.get('status')}")

            prompt.append("")

        # --------------------------------------------------
        # Issue Classification
        # --------------------------------------------------

        prompt.append("Issue Classification:")

        prompt.append(
            f"- Category   : {analysis.classification.category}"
        )

        prompt.append(
            f"- Confidence : {analysis.classification.confidence:.2f}"
        )

        prompt.append("")

        # --------------------------------------------------
        # Affected Network Functions
        # --------------------------------------------------

        if analysis.analysis.affected_network_functions:

            prompt.append("Affected Network Functions:")

            for nf in analysis.analysis.affected_network_functions:

                prompt.append(f"- {nf}")

            prompt.append("")

        # --------------------------------------------------
        # Interfaces
        # --------------------------------------------------

        if analysis.analysis.interfaces:

            prompt.append("Relevant Interfaces:")

            for interface in analysis.analysis.interfaces:

                prompt.append(f"- {interface}")

            prompt.append("")

        # --------------------------------------------------
        # Protocols
        # --------------------------------------------------

        if analysis.analysis.protocols:

            prompt.append("Relevant Protocols:")

            for protocol in analysis.analysis.protocols:

                prompt.append(f"- {protocol}")

            prompt.append("")

        # --------------------------------------------------
        # Possible Causes
        # --------------------------------------------------

        if analysis.analysis.possible_causes:

            prompt.append("Possible Causes:")

            for cause in analysis.analysis.possible_causes:

                prompt.append(f"- {cause}")

            prompt.append("")

        # --------------------------------------------------
        # Troubleshooting Steps
        # --------------------------------------------------

        if analysis.troubleshooting_steps:

            prompt.append("Recommended Troubleshooting Steps:")

            for step in analysis.troubleshooting_steps:

                prompt.append(
                    f"{step.step_number}. {step.description}"
                )

            prompt.append("")

        # --------------------------------------------------
        # Instructions
        # --------------------------------------------------

        prompt.append("Tasks:")

        prompt.append(
            "1. Identify the most probable root cause."
        )

        prompt.append(
            "2. Explain why the identified network functions are relevant."
        )

        prompt.append(
            "3. Describe the likely impact on the 5G network."
        )

        prompt.append(
            "4. Recommend the next troubleshooting actions."
        )

        prompt.append(
            "5. If the available information is insufficient, "
            "clearly state what additional information is required."
        )

        prompt.append(
            "6. Do not make assumptions beyond the provided evidence."
        )

        return "\n".join(prompt)