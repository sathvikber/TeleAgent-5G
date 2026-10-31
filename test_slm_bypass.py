"""
Standalone Compact-RAG + SLM test.

This script bypasses:
- main.py
- Coordinator
- IntentAnalyzer
- TaskPlanner
- Dispatcher
- LogAgent
- MonitoringAgent
- NetworkAgent

It tests only:

Query
    ↓
RAGService
    ↓
IntelligentRetrievalService
    ↓
Compact Evidence
    ↓
RAGPromptBuilder
    ↓
Qwen2.5-1.5B + LoRA
    ↓
Answer
"""

import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel

from teleagent.config import (
    MODEL_NAME,
    LORA_PATH,
    USE_LORA,
    KNOWLEDGE_BASE_PATH,
)

from teleagent.rag.rag_service import RAGService
from teleagent.retrieval.intelligent_retrieval_service import (
    IntelligentRetrievalService,
)


# ============================================================
# Query
# ============================================================

QUERY = (
    "According to TS 33.501, "
    "what is UDM?"
)


# ============================================================
# Build RAG prompt
# ============================================================

print("=" * 80)
print("STEP 1: Loading Knowledge Base")
print("=" * 80)

rag_service = RAGService(
    knowledge_base_directory=KNOWLEDGE_BASE_PATH
)

retrieval_service = IntelligentRetrievalService(
    rag_service=rag_service
)


print()
print("=" * 80)
print("STEP 2: Building Compact RAG Prompt")
print("=" * 80)

rag_prompt = retrieval_service.build_prompt(
    query=QUERY,
    top_k=5,
)


print()
print("=" * 80)
print("FINAL RAG PROMPT")
print("=" * 80)
print(rag_prompt)
print("=" * 80)


# ============================================================
# Load tokenizer
# ============================================================

print()
print("=" * 80)
print("STEP 3: Loading SLM")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# Load base model
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)


# ============================================================
# Load LoRA adapter
# ============================================================

if USE_LORA:

    print(
        f"\nLoading LoRA adapter from: {LORA_PATH}"
    )

    model = PeftModel.from_pretrained(
        model,
        LORA_PATH,
    )


model.to("cpu")

model.eval()


print()
print("Model loaded successfully.")
print(f"Base Model : {MODEL_NAME}")
print(f"LoRA       : {USE_LORA}")

if USE_LORA:
    print(f"LoRA Path  : {LORA_PATH}")


# ============================================================
# Qwen chat format
# ============================================================

messages = [
    {
        "role": "user",
        "content": rag_prompt,
    }
]


formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)


# ============================================================
# Tokenize
# ============================================================

inputs = tokenizer(
    formatted_prompt,
    return_tensors="pt",
)

inputs = {
    key: value.to("cpu")
    for key, value in inputs.items()
}


input_token_count = inputs[
    "input_ids"
].shape[-1]


print()
print("=" * 80)
print("STEP 4: SLM INFERENCE")
print("=" * 80)

print(
    f"Input Tokens : {input_token_count}"
)


# ============================================================
# Generate
# ============================================================

start_time = time.perf_counter()

with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=180,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )


elapsed = (
    time.perf_counter()
    - start_time
)


# ============================================================
# Remove input tokens
# ============================================================

generated_tokens = outputs[
    0,
    input_token_count:
]


answer = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True,
).strip()


# ============================================================
# Output
# ============================================================

print()
print("=" * 80)
print("SLM ANSWER")
print("=" * 80)

print(answer)

print()
print("=" * 80)
print("INFERENCE METADATA")
print("=" * 80)

print(
    f"Inference Time : "
    f"{elapsed:.2f} seconds"
)

print(
    f"Generated Tokens : "
    f"{len(generated_tokens)}"
)

if elapsed > 0:

    print(
        f"Tokens / second : "
        f"{len(generated_tokens) / elapsed:.2f}"
    )

print("=" * 80)
