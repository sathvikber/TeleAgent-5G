"""
model_manager.py

Loads and manages the lifecycle of the language model and tokenizer.

Responsibilities
----------------
- Load tokenizer
- Load base model
- Load multiple LoRA adapters into one base model
- Switch adapters for individual reasoning requests
- Provide access to loaded objects

Author : TeleAgent-5G
Version: 0.2
"""

from pathlib import Path
from threading import RLock
from typing import Dict, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from peft import PeftModel

from teleagent.exceptions import ModelNotLoadedError


class ModelManager:
    """
    Loads and manages the Telecom SLM and its LoRA adapters.
    """

    def __init__(
        self,
        model_name: str,
        adapter_paths: Optional[Dict[str, str]] = None,
        default_adapter: Optional[str] = None,
        device: Optional[str] = None,
    ):

        self.model_name = model_name
        self.adapter_paths = adapter_paths or {}
        self.default_adapter = default_adapter

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = None
        self.tokenizer = None
        self.active_adapter = None

        # Shared lock protects adapter switching and model generation.
        self.adapter_lock = RLock()

        if (
            self.default_adapter is not None
            and self.default_adapter not in self.adapter_paths
        ):
            raise ValueError(
                f"Default adapter '{self.default_adapter}' "
                "is not present in adapter_paths."
            )

    # -------------------------------------------------------------
    # Load model and adapters
    # -------------------------------------------------------------

    def load(self):

        print(f"Loading tokenizer: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        print(f"Loading base model: {self.model_name}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=(
                torch.float16
                if self.device == "cuda"
                else torch.float32
            ),
            device_map="auto" if self.device == "cuda" else None,
        )

        if self.adapter_paths:

            # Validate all adapter directories before loading any.
            for name, adapter_path in self.adapter_paths.items():

                path = Path(adapter_path)

                if not path.is_dir():
                    raise FileNotFoundError(
                        f"Adapter directory not found: {path}"
                    )

                if not (path / "adapter_config.json").is_file():
                    raise FileNotFoundError(
                        f"Missing adapter_config.json: {path}"
                    )

                if not (
                    (path / "adapter_model.safetensors").is_file()
                    or (path / "adapter_model.bin").is_file()
                ):
                    raise FileNotFoundError(
                        f"Adapter weights not found in: {path}"
                    )

            # Use the first adapter to wrap the base model.
            adapter_items = iter(self.adapter_paths.items())
            first_name, first_path = next(adapter_items)

            print(f"Loading LoRA adapter: {first_name}")

            self.model = PeftModel.from_pretrained(
                self.model,
                first_path,
                adapter_name=first_name,
                is_trainable=False,
            )

            # Load the remaining adapters into the same model.
            for name, adapter_path in adapter_items:

                print(f"Loading LoRA adapter: {name}")

                self.model.load_adapter(
                    adapter_path,
                    adapter_name=name,
                    is_trainable=False,
                )

            # Activate the configured default adapter.
            selected_adapter = (
                self.default_adapter or first_name
            )

            self.model.set_adapter(selected_adapter)
            self.active_adapter = selected_adapter

        self.model.eval()

        print("Model successfully loaded.")

        self.print_model_info()

    # -------------------------------------------------------------
    # Adapter Management
    # -------------------------------------------------------------

    def activate_adapter(
        self,
        adapter_name: Optional[str] = None,
    ):
        """
        Activate a loaded adapter.

        Call this method while holding adapter_lock when generation
        follows, so another agent cannot switch the adapter mid-request.
        """

        if self.model is None:
            raise ModelNotLoadedError(
                "Model has not been loaded."
            )

        selected_adapter = (
            adapter_name or self.default_adapter
        )

        # Support base-model-only operation if no adapters were supplied.
        if not self.adapter_paths:
            if selected_adapter is not None:
                raise ValueError(
                    "An adapter was requested, but no adapters are loaded."
                )
            return None

        if selected_adapter not in self.adapter_paths:
            raise ValueError(
                f"Unknown adapter '{selected_adapter}'. "
                f"Available adapters: {list(self.adapter_paths.keys())}"
            )

        with self.adapter_lock:

            if self.active_adapter != selected_adapter:
                self.model.set_adapter(selected_adapter)
                self.active_adapter = selected_adapter

        return selected_adapter

    # -------------------------------------------------------------
    # Model Information
    # -------------------------------------------------------------

    def print_model_info(self):

        print("=" * 60)
        print("Model Information")
        print("=" * 60)

        print(f"Model Name       : {self.model_name}")
        print(f"Device           : {self.device}")
        print(f"Adapters         : {list(self.adapter_paths.keys())}")
        print(f"Default Adapter  : {self.default_adapter}")
        print(f"Active Adapter   : {self.active_adapter}")

        if self.adapter_paths:
            print("LoRA             : ENABLED")
        else:
            print("LoRA             : DISABLED")

        print(
            f"Class            : "
            f"{type(self.model).__name__ if self.model else 'Not loaded'}"
        )

        print("=" * 60)

    # -------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------

    def get_model(self):

        if self.model is None:
            raise ModelNotLoadedError(
                "Model has not been loaded."
            )

        return self.model

    def get_tokenizer(self):

        if self.tokenizer is None:
            raise ModelNotLoadedError(
                "Tokenizer has not been loaded."
            )

        return self.tokenizer

    # -------------------------------------------------------------
    # Status
    # -------------------------------------------------------------

    @property
    def is_loaded(self):

        return (
            self.model is not None
            and self.tokenizer is not None
        )
