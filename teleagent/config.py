"""
TeleAgent Configuration

Global configuration for the TeleAgent runtime.

Author : Sapna Jha
"""

from pathlib import Path

# ==================================================
# Model
# ==================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Root directory containing all seven LoRA adapters
ADAPTERS_ROOT = Path(
    r"/workspace/slm-lora"
    r"/outputs/individual_adapters"
)

ADAPTER_NAMES = [
    "adapter_5g",
    "adapter_5g_core",
    "adapter_5g_slm",
    "adapter_beam",
    "adapter_kpi2",
    "adapter_trouble",
    "adapter_wave",
]

# Adapter name -> adapter directory
ADAPTER_PATHS = {
    name: str(ADAPTERS_ROOT / name)
    for name in ADAPTER_NAMES
}

# Default adapter for general-purpose requests
DEFAULT_ADAPTER = "adapter_5g_slm"

# Initial agent-to-adapter mappings.
# These mappings can be refined after validating the training datasets.
AGENT_ADAPTER_MAP = {
    "knowledge": "adapter_5g",
    "log": "adapter_trouble",
    "network": "adapter_5g_core",
    "monitoring": "adapter_kpi2",
}

# ==================================================
# Generation Parameters
# ==================================================

MAX_NEW_TOKENS = 120

TEMPERATURE = 0.7

TOP_P = 0.9

TOP_K = 50

DO_SAMPLE = True

REPETITION_PENALTY = 1.1

# ==================================================
# Device
# ==================================================

DEVICE_MAP = "cuda"

DTYPE = "float16"

# ==================================================
# Conversation
# ==================================================

SYSTEM_PROMPT = (
    "You are TeleAgent, an AI assistant specialized in "
    "private 5G networks. "
    "You assist in telecom troubleshooting, "
    "log interpretation, protocol analysis, "
    "network optimization, and edge computing."
)

MAX_HISTORY = 10

# ==================================================
# Logging
# ==================================================

LOG_DIR = "logs"

LOG_LEVEL = "INFO"

# ==================================================
# Application
# ==================================================

PROJECT_NAME = "TeleAgent"

VERSION = "0.1"

DEBUG = False

KNOWLEDGE_BASE_PATH = (
    r"/workspace/teleagent/data/knowledge_base"
)

RAG_TOP_K = 5
