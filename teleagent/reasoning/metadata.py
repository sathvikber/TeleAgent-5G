"""
metadata.py

Capability descriptor for the TeleAgent Core Reasoning Service.

This module exposes metadata about the currently deployed reasoning
engine. The Coordinator Agent and other services can query this
information to determine the capabilities of the reasoning engine.

Author : TeleAgent-5G
Version: 0.1
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ReasoningMetadata:
    """
    Static metadata describing the reasoning engine.
    """

    service_name: str

    version: str

    model_name: str

    base_model: str

    lora_enabled: bool

    deployment_target: str

    max_context_length: int

    supported_languages: List[str]

    supported_tasks: List[str]

    supported_input_types: List[str]


REASONING_METADATA = ReasoningMetadata(

    service_name="TeleAgent Core Reasoning Service",

    version="0.1.0",

    model_name="TeleAgent-SLM",

    base_model="Qwen2.5-1.5B-Instruct",

    lora_enabled=True,

    deployment_target="MEC",

    max_context_length=32768,

    supported_languages=[
        "English"
    ],

    supported_tasks=[
        "chat",
        "interpret_log",
        "troubleshoot",
        "question_answering",
        "summarization"
    ],

    supported_input_types=[
        "text",
        "network_logs",
        "telecom_events"
    ]
)
