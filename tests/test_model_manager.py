import pytest

from teleagent.reasoning.model_manager import ModelManager
from teleagent.exceptions import ModelNotLoadedError


def test_manager_initialization():

    manager = ModelManager(
        model_name="Qwen/Qwen2.5-1.5B-Instruct"
    )

    assert manager.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert manager.model is None
    assert manager.tokenizer is None
    assert manager.is_loaded is False


def test_get_model_before_loading():

    manager = ModelManager(
        model_name="Qwen/Qwen2.5-1.5B-Instruct"
    )

    with pytest.raises(ModelNotLoadedError):
        manager.get_model()


def test_get_tokenizer_before_loading():

    manager = ModelManager(
        model_name="Qwen/Qwen2.5-1.5B-Instruct"
    )

    with pytest.raises(ModelNotLoadedError):
        manager.get_tokenizer()
