import pytest

from teleagent.reasoning.reasoning_service import ReasoningService
from teleagent.reasoning.model_manager import ModelManager
from teleagent.exceptions import ModelNotLoadedError


def test_service_requires_loaded_model():

    manager = ModelManager(
        model_name="Qwen/Qwen2.5-1.5B-Instruct"
    )

    with pytest.raises(ModelNotLoadedError):

        ReasoningService(manager)
