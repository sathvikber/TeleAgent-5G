import pytest

from teleagent.reasoning.inference_engine import InferenceEngine
from teleagent.reasoning.model_manager import ModelManager
from teleagent.exceptions import ModelNotLoadedError


def test_engine_requires_loaded_model():

    manager = ModelManager(
        model_name="Qwen/Qwen2.5-1.5B-Instruct"
    )

    with pytest.raises(ModelNotLoadedError):

        InferenceEngine(manager)
