"""
schemas.py

Pydantic data models for the TeleAgent Core Reasoning Service (CRS).

These schemas define the data contract between the different
modules of TeleAgent.

Author : TeleAgent-5G
Version: 0.2
"""

from datetime import datetime
from typing import List, Optional, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------
# Generation Configuration
# ---------------------------------------------------------------------

class GenerationConfig(BaseModel):
    """
    Text generation parameters used during inference.
    """

    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    top_p: float = Field(default=0.9, ge=0.0, le=1.0)

    top_k: int = Field(default=50, ge=1)

    max_new_tokens: int = Field(default=512, ge=1)

    do_sample: bool = True

    repetition_penalty: float = Field(default=1.1, ge=1.0)


# ---------------------------------------------------------------------
# Conversation Message
# ---------------------------------------------------------------------

class Message(BaseModel):
    """
    Represents a single message in a conversation.
    """

    role: Literal["system", "user", "assistant"] = Field(
        description="Role of the message sender."
    )

    content: str = Field(
        min_length=1,
        description="Message content."
    )


# ---------------------------------------------------------------------
# Reasoning Request
# ---------------------------------------------------------------------

class ReasoningRequest(BaseModel):
    """
    Input request to the Core Reasoning Service.
    """

    task: str = Field(
        ...,
        description="Reasoning task (e.g., troubleshoot, interpret_log, summarize)."
    )

    query: str = Field(
        ...,
        min_length=1,
        description="Primary input text or network log."
    )

    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt."
    )

    conversation: List[Message] = Field(
        default_factory=list,
        description="Conversation history."
    )

    context: Optional[str] = Field(
        default=None,
        description="Additional context."
    )

    generation_config: GenerationConfig = Field(
        default_factory=GenerationConfig
    )

    # Name of the adapter selected for this request.
    # None means the ModelManager's default adapter is used.
    adapter_name: Optional[str] = Field(
        default=None,
        description="LoRA adapter to use for this reasoning request."
    )


# ---------------------------------------------------------------------
# Model Metadata
# ---------------------------------------------------------------------

class ModelMetadata(BaseModel):
    """
    Metadata returned after inference.
    """

    model_name: str

    model_type: Literal["base", "lora"]

    tokenizer_name: str

    lora_enabled: bool

    device: str

    # Adapter actually used for this response, if any.
    adapter_name: Optional[str] = Field(
        default=None,
        description="Name of the LoRA adapter used during inference."
    )

    inference_time_ms: float = Field(
        default=0.0,
        ge=0
    )

    tokens_generated: int = Field(
        default=0,
        ge=0
    )

    finish_reason: str = "completed"


# ---------------------------------------------------------------------
# Reasoning Response
# ---------------------------------------------------------------------

class ReasoningResponse(BaseModel):
    """
    Response returned by the Core Reasoning Service.
    """

    success: bool = True

    response: str

    metadata: ModelMetadata

    timestamp: datetime = Field(
        default_factory=datetime.utcnow
    )
