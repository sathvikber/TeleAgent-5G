from teleagent.reasoning.metadata import REASONING_METADATA


def test_metadata_values():

    assert REASONING_METADATA.service_name == "TeleAgent Core Reasoning Service"

    assert REASONING_METADATA.lora_enabled is True

    assert "interpret_log" in REASONING_METADATA.supported_tasks

    assert REASONING_METADATA.base_model == "Qwen2.5-1.5B-Instruct"
