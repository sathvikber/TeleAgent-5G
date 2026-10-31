"""
Network Analyzer

Analyzes a classified telecom issue and extracts the
most relevant network functions, interfaces, protocols,
and possible causes.
"""

from teleagent.network.models import (
    IssueClassification,
    NetworkAnalysis,
)


class NetworkAnalyzer:
    """
    Performs deterministic network analysis.
    """

    def __init__(self) -> None:

        self._analysis_rules = {

            "registration": {
                "network_functions": [
                    "AMF",
                    "AUSF",
                    "UDM",
                ],
                "interfaces": [
                    "N1",
                    "N2",
                ],
                "protocols": [
                    "NAS",
                    "NGAP",
                ],
                "possible_causes": [
                    "Registration request rejected",
                    "Authentication failure",
                    "AMF configuration issue",
                    "NGAP signalling problem",
                ],
            },

            "authentication": {
                "network_functions": [
                    "AUSF",
                    "UDM",
                    "AMF",
                ],
                "interfaces": [
                    "N1",
                ],
                "protocols": [
                    "NAS",
                ],
                "possible_causes": [
                    "Authentication timeout",
                    "Invalid subscriber credentials",
                    "UDM unreachable",
                ],
            },

            "pdu_session": {
                "network_functions": [
                    "SMF",
                    "UPF",
                    "AMF",
                ],
                "interfaces": [
                    "N4",
                    "N11",
                ],
                "protocols": [
                    "PFCP",
                    "GTP-U",
                ],
                "possible_causes": [
                    "PDU Session Reject",
                    "UPF unavailable",
                    "SMF configuration error",
                ],
            },

            "mobility": {
                "network_functions": [
                    "AMF",
                    "gNodeB",
                ],
                "interfaces": [
                    "N2",
                ],
                "protocols": [
                    "NGAP",
                    "RRC",
                ],
                "possible_causes": [
                    "Handover failure",
                    "Tracking area update failure",
                    "Radio link degradation",
                ],
            },

            "ngap": {
                "network_functions": [
                    "AMF",
                    "gNodeB",
                ],
                "interfaces": [
                    "N2",
                ],
                "protocols": [
                    "NGAP",
                ],
                "possible_causes": [
                    "NG Setup failure",
                    "NGAP timeout",
                ],
            },

            "nas": {
                "network_functions": [
                    "AMF",
                    "UE",
                ],
                "interfaces": [
                    "N1",
                ],
                "protocols": [
                    "NAS",
                ],
                "possible_causes": [
                    "NAS message rejection",
                    "Security context mismatch",
                ],
            },

            "gtp": {
                "network_functions": [
                    "UPF",
                    "SMF",
                ],
                "interfaces": [
                    "N3",
                    "N9",
                ],
                "protocols": [
                    "GTP-U",
                ],
                "possible_causes": [
                    "Missing GTP tunnel",
                    "Tunnel endpoint failure",
                ],
            },

            "radio": {
                "network_functions": [
                    "gNodeB",
                ],
                "interfaces": [
                    "Uu",
                ],
                "protocols": [
                    "RRC",
                ],
                "possible_causes": [
                    "Poor radio coverage",
                    "Interference",
                    "Weak signal strength",
                ],
            },

            "slice": {
                "network_functions": [
                    "NSSF",
                    "AMF",
                ],
                "interfaces": [
                    "N22",
                ],
                "protocols": [
                    "HTTP/2",
                ],
                "possible_causes": [
                    "Slice selection failure",
                    "Requested S-NSSAI unavailable",
                ],
            },

            "qos": {
                "network_functions": [
                    "SMF",
                    "UPF",
                ],
                "interfaces": [
                    "N4",
                    "N3",
                ],
                "protocols": [
                    "GTP-U",
                ],
                "possible_causes": [
                    "High latency",
                    "Packet loss",
                    "Incorrect QoS policy",
                ],
            },

            "general": {
                "network_functions": [],
                "interfaces": [],
                "protocols": [],
                "possible_causes": [
                    "Insufficient information",
                ],
            },
        }

    def analyze(
        self,
        classification: IssueClassification,
    ) -> NetworkAnalysis:
        """
        Analyze a classified network issue.

        Parameters
        ----------
        classification : IssueClassification

        Returns
        -------
        NetworkAnalysis
        """

        rule = self._analysis_rules.get(
            classification.category,
            self._analysis_rules["general"],
        )

        return NetworkAnalysis(
            affected_network_functions=rule["network_functions"],
            interfaces=rule["interfaces"],
            protocols=rule["protocols"],
            possible_causes=rule["possible_causes"],
        )
