"""
exceptions.py

Custom exceptions used throughout the TeleAgent reasoning module.
"""


class TeleAgentError(Exception):
    """
    Base class for all TeleAgent exceptions.
    """
    pass


class ModelNotLoadedError(TeleAgentError):
    """
    Raised when inference is attempted before the model is loaded.
    """
    pass


class TokenizerNotLoadedError(TeleAgentError):
    """
    Raised when the tokenizer is unavailable.
    """
    pass


class InvalidPromptError(TeleAgentError):
    """
    Raised when the prompt cannot be constructed.
    """
    pass


class InferenceError(TeleAgentError):
    """
    Raised when model inference fails.
    """
    pass


class ConfigurationError(TeleAgentError):
    """
    Raised when the application configuration is invalid.
    """
    pass


class AgentNotFoundError(TeleAgentError):
    """
    Raised when an agent requested from the registry
    has not been registered.
    """
    pass
