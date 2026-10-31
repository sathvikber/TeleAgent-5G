"""
intent_classifier.py

Abstract interface for intent classification.

This module defines the interface that all intent
classifiers must implement.

Implementations may use:
    - Fine-tuned Telecom SLM
    - External LLM
    - ML classifier
    - Rule-based classifier

Author : TeleAgent-5G
Version: 0.1
"""

from abc import ABC, abstractmethod


class IntentClassifier(ABC):
    """
    Abstract base class for intent classifiers.
    """

    @abstractmethod
    def classify(self, query: str) -> str:
        """
        Classify a user query into an intent.

        Parameters
        ----------
        query : str

        Returns
        -------
        str
            Detected intent.
        """
        raise NotImplementedError
