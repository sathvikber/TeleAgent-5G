import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel

from .agent import LogAgent


# =========================================================
# CONFIGURATION
# =========================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ADAPTER_PATH = "/workspace/slm-lora/outputs"

OUTPUT_DIR = "teleagent/log_agent/output"


# =========================================================
# LOAD SLM
# =========================================================

def load_slm():

    print("\n")
    print("=" * 60)
    print("LOADING QWEN SLM")
    print("=" * 60)

    # -----------------------------------------------------
    # Load tokenizer
    # -----------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    # -----------------------------------------------------
    # Load base model
    # -----------------------------------------------------

    print("Loading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32
    )

    # -----------------------------------------------------
    # Load LoRA adapter
    # -----------------------------------------------------

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        model,
        ADAPTER_PATH
    )

    model.eval()

    print("SLM loaded successfully.")

    return tokenizer, model


# =========================================================
# RUN SLM
# =========================================================

def run_slm(
    tokenizer,
    model,
    prompt
):

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert telecom network "
                "optimization and troubleshooting assistant. "
                "Analyze the provided 5G network logs and "
                "identify the problems, affected components, "
                "likely root causes, and recommended "
                "troubleshooting actions."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    # -----------------------------------------------------
    # Format prompt using Qwen chat template
    # -----------------------------------------------------

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # -----------------------------------------------------
    # Tokenize
    # -----------------------------------------------------

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    )

    # -----------------------------------------------------
    # Move inputs to model device
    # -----------------------------------------------------

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    print("\nGenerating SLM response...")

    # -----------------------------------------------------
    # Generate response
    # -----------------------------------------------------

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )

    # -----------------------------------------------------
    # Remove input tokens
    # -----------------------------------------------------

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    # -----------------------------------------------------
    # Decode response
    # -----------------------------------------------------

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return response.strip()


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("5G LOG AGENT + SLM TEST")
    print("=" * 60)

    # =====================================================
    # STEP 1 — RUN LOG AGENT
    # =====================================================

    print("\n")
    print("=" * 60)
    print("STEP 1: RUNNING LOG AGENT")
    print("=" * 60)

    agent = LogAgent(
        output_dir=OUTPUT_DIR
    )

    # Docker has no graphical display,
    # so use direct file path input.

    log_file = input(
        "\nEnter PDF/log file path: "
    ).strip()

    if not log_file:

        print("\nNo file provided. Exiting.")

        return

    if not os.path.exists(log_file):

        print(
            f"\nFile not found: {log_file}"
        )

        return

    print("\nProcessing file...")

    result = agent.run_from_files(
        [log_file]
    )

    # =====================================================
    # STEP 1 RESULT
    # =====================================================

    print("\nLog Agent completed.")

    print(
        "Events:",
        len(result.get("records", []))
    )

    print(
        "Source files:",
        result.get("source_files", [])
    )

    print(
        "Structured logs:",
        result.get("structured_path")
    )

    print(
        "SLM context:",
        result.get("context_path")
    )

    print(
        "Generated prompt:",
        result.get("prompt_path")
    )

    # -----------------------------------------------------
    # Check processing errors
    # -----------------------------------------------------

    if result.get("errors"):

        print("\nProcessing errors:")

        for error in result["errors"]:

            print(
                " -",
                error
            )

    # =====================================================
    # STEP 2 — GET GENERATED PROMPT
    # =====================================================

    print("\n")
    print("=" * 60)
    print("STEP 2: LOADING GENERATED SLM PROMPT")
    print("=" * 60)

    prompt_path = result.get(
        "prompt_path"
    )

    if not prompt_path:

        print(
            "\nERROR: SLM prompt path was not returned."
        )

        return

    if not os.path.exists(prompt_path):

        print(
            "\nERROR: SLM prompt file does not exist:"
        )

        print(prompt_path)

        return

    # -----------------------------------------------------
    # Read prompt
    # -----------------------------------------------------

    with open(
        prompt_path,
        "r",
        encoding="utf-8"
    ) as file:

        prompt = file.read()

    if not prompt.strip():

        print(
            "\nERROR: Generated SLM prompt is empty."
        )

        return

    print(
        "\nPrompt loaded:",
        len(prompt),
        "characters"
    )

    # =====================================================
    # STEP 3 — LOAD SLM
    # =====================================================

    print("\n")
    print("=" * 60)
    print("STEP 3: LOADING FINE-TUNED SLM")
    print("=" * 60)

    # -----------------------------------------------------
    # Check adapter path before loading
    # -----------------------------------------------------

    if not os.path.exists(ADAPTER_PATH):

        print(
            "\nERROR: LoRA adapter path does not exist:"
        )

        print(ADAPTER_PATH)

        print(
            "\nThe adapter needs to be accessible "
            "inside the Docker container."
        )

        return

    tokenizer, model = load_slm()

    # =====================================================
    # STEP 4 — RUN SLM
    # =====================================================

    print("\n")
    print("=" * 60)
    print("STEP 4: GENERATING TROUBLESHOOTING RESPONSE")
    print("=" * 60)

    response = run_slm(
        tokenizer,
        model,
        prompt
    )

    # =====================================================
    # STEP 5 — DISPLAY RESPONSE
    # =====================================================

    print("\n")
    print("=" * 60)
    print("SLM TROUBLESHOOTING RESPONSE")
    print("=" * 60)

    print()

    if response:

        print(response)

    else:

        print(
            "SLM returned an empty response."
        )

    print("\n")
    print("=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()
