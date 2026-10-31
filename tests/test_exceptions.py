import pytest

from teleagent.exceptions import (
    TeleAgentError,
    ModelNotLoadedError,
)


def test_model_not_loaded_error():
    with pytest.raises(ModelNotLoadedError):
        raise ModelNotLoadedError("Model not loaded")


def test_base_exception():
    with pytest.raises(TeleAgentError):
        raise TeleAgentError("Base TeleAgent exception")
