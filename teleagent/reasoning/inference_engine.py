"""
inference_engine.py

Runs inference using the Telecom SLM.

Responsibilities
----------------
- Convert chat messages to model inputs
- Execute model.generate()
- Decode the output
- Return the generated response
- Activate the requested LoRA adapter safely

Author : TeleAgent-5G
Version : 0.2
"""

import time
from typing import List, Dict

import torch

from teleagent.reasoning.schemas import (
    ReasoningRequest,
    ReasoningResponse,
    ModelMetadata,
)
from teleagent.reasoning.model_manager import ModelManager


class InferenceEngine:
    """
    Executes inference using the loaded language model.
    """

    def __init__(self, model_manager: ModelManager):

        self.model_manager = model_manager

        self.model = model_manager.get_model()
        self.tokenizer = model_manager.get_tokenizer()

    # ---------------------------------------------------------
    # Generate Response
    # ---------------------------------------------------------

    def generate(
        self,
        request: ReasoningRequest,
        messages: List[Dict[str, str]]
    ) -> ReasoningResponse:

        start_time = time.perf_counter()

        # ---------------------------------------------
        # Apply Qwen chat template
        # ---------------------------------------------

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # ---------------------------------------------
        # Tokenize
        # ---------------------------------------------

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(self.model.device)
            for k, v in inputs.items()
        }

        config = request.generation_config

        # ---------------------------------------------
        # Activate adapter and generate
        # ---------------------------------------------
        # All agents share one model. The same lock must
        # protect adapter selection AND generation.

        with self.model_manager.adapter_lock:

            used_adapter = self.model_manager.activate_adapter(
                request.adapter_name
            )

            with torch.no_grad():

                outputs = self.model.generate(
                    **inputs,

                    max_new_tokens=config.max_new_tokens,

                    temperature=config.temperature,

                    top_p=config.top_p,

                    top_k=config.top_k,

                    do_sample=config.do_sample,

                    repetition_penalty=config.repetition_penalty,
                )

        # ---------------------------------------------
        # Decode only newly generated tokens
        # ---------------------------------------------

        generated_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        response_text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        print("\n[DEBUG InferenceEngine]")
        print("Task:", request.task)
        print("Adapter:", used_adapter)
        print("Generated token count:", len(generated_tokens))
        print("Decoded response:", repr(response_text))
        print("Response empty:", not response_text.strip())

        end_time = time.perf_counter()

        # ---------------------------------------------
        # Metadata
        # ---------------------------------------------

        metadata = ModelMetadata(

            model_name=self.model_manager.model_name,

            model_type="lora" if used_adapter else "base",

            tokenizer_name=self.model_manager.model_name,

            lora_enabled=used_adapter is not None,

            adapter_name=used_adapter,

            device=str(self.model.device),

            inference_time_ms=(end_time - start_time) * 1000,

            tokens_generated=len(generated_tokens),

            finish_reason="completed",
        )

        # ---------------------------------------------
        # Return response
        # ---------------------------------------------

        return ReasoningResponse(

            success=True,

            response=response_text.strip(),

            metadata=metadata,
        )
