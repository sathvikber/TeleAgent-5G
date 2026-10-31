"""
Network Recommendation Engine

Generates deterministic troubleshooting recommendations
based on the analyzed network issue.
"""

from teleagent.network.models import (
    NetworkAnalysis,
    TroubleshootingStep,
)


class RecommendationEngine:
    """
    Generates troubleshooting steps for common
    5G network issues.
    """

    def __init__(self) -> None:

        self._recommendations = {

            "AMF": [
                "Verify that the AMF service is running.",
                "Check AMF logs for registration failures.",
                "Verify NGAP connectivity with the gNodeB.",
            ],

            "AUSF": [
                "Verify authentication requests are reaching the AUSF.",
                "Check AUSF availability and response times.",
            ],

            "UDM": [
                "Verify subscriber data is available in the UDM.",
                "Check connectivity between AUSF and UDM.",
            ],

            "SMF": [
                "Verify SMF is reachable.",
                "Check PDU Session Establishment procedures.",
            ],

            "UPF": [
                "Verify GTP-U tunnels are established.",
                "Check PFCP association between SMF and UPF.",
            ],

            "gNodeB": [
                "Verify the gNodeB is connected to the AMF.",
                "Check radio link quality and NG Setup status.",
            ],

            "NSSF": [
                "Verify slice selection configuration.",
                "Check S-NSSAI availability.",
            ],
        }

    def generate(
        self,
        analysis: NetworkAnalysis,
    ) -> list[TroubleshootingStep]:
        """
        Generate troubleshooting recommendations.

        Parameters
        ----------
        analysis : NetworkAnalysis

        Returns
        -------
        list[TroubleshootingStep]
        """

        steps = []
        step_number = 1

        for nf in analysis.affected_network_functions:

            recommendations = self._recommendations.get(nf, [])

            for recommendation in recommendations:

                steps.append(
                    TroubleshootingStep(
                        step_number=step_number,
                        description=recommendation,
                    )
                )

                step_number += 1

        if not steps:

            steps.append(
                TroubleshootingStep(
                    step_number=1,
                    description=(
                        "Collect additional information before "
                        "performing troubleshooting."
                    ),
                )
            )

        return steps
