"""
Network Issue Classifier

Classifies telecom troubleshooting queries into predefined
network issue categories.
"""

from teleagent.network.models import IssueClassification


class IssueClassifier:
    """
    Rule-based classifier for telecom network issues.
    """

    def __init__(self) -> None:

        self._rules = {
            "registration": [
                "registration",
                "register",
                "attach",
                "initial registration",
            ],
            "authentication": [
                "authentication",
                "authenticate",
                "ausf",
                "udm",
                "identity",
            ],
            "pdu_session": [
                "pdu session",
                "session",
                "smf",
                "establishment",
                "release",
            ],
            "mobility": [
                "handover",
                "mobility",
                "tracking area",
                "tau",
            ],
            "ngap": [
                "ngap",
                "n2",
            ],
            "nas": [
                "nas",
                "n1",
            ],
            "gtp": [
                "gtp",
                "gtp-u",
                "teid",
                "tunnel",
                "upf",
            ],
            "radio": [
                "gnb",
                "cell",
                "rrc",
                "radio",
                "signal",
                "coverage",
            ],
            "slice": [
                "slice",
                "snssai",
                "nssf",
            ],
            "qos": [
                "qos",
                "5qi",
                "latency",
                "throughput",
                "packet loss",
                "jitter",
            ],
        }

    def classify(
        self,
        query: str,
    ) -> IssueClassification:
        """
        Classify a telecom troubleshooting query.

        Parameters
        ----------
        query : str

        Returns
        -------
        IssueClassification
        """

        text = query.lower()

        for category, keywords in self._rules.items():

            for keyword in keywords:

                if keyword in text:

                    return IssueClassification(
                        category=category,
                        confidence=1.0,
                    )

        return IssueClassification(
            category="general",
            confidence=0.5,
        )
