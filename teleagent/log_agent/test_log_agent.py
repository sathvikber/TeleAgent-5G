from teleagent.log_agent import LogAgent


def main():
    print("=" * 60)
    print("5G LOG AGENT + SLM TEST")
    print("=" * 60)

    # --------------------------------------------------
    # STEP 1: RUN LOG AGENT
    # --------------------------------------------------
    print("\n")
    print("=" * 60)
    print("STEP 1: RUNNING LOG AGENT")
    print("=" * 60)

    agent = LogAgent(
        output_dir="teleagent/log_agent/output"
    )

    # Docker has no graphical display, so use a file path
    log_file = input("\nEnter PDF/log file path: ").strip()

    if not log_file:
        print("No file provided. Exiting.")
        return

    if not os.path.exists(log_file):
        print(f"File not found: {log_file}")
        return

    result = agent.run_from_files([log_file])

    print("\nLog Agent completed.")
    print("Status:", result.get("status", "success"))
    print("Files:", result.get("source_files", []))
    print("Events:", len(result.get("records", [])))

    if result.get("errors"):
        print("\nProcessing errors:")
        for error in result["errors"]:
            print(error)

    print("\nGenerated files:")
    print("Structured logs:", result.get("structured_path"))
    print("SLM context:", result.get("context_path"))
    print("SLM prompt:", result.get("prompt_path"))

    # --------------------------------------------------
    # STEP 2: CHECK GENERATED PROMPT
    # --------------------------------------------------
    prompt_path = result.get("prompt_path")

    if not prompt_path or not os.path.exists(prompt_path):
        print("\nERROR: SLM prompt was not generated.")
        return

    with open(prompt_path, "r", encoding="utf-8") as f:
        slm_prompt = f.read().strip()

    if not slm_prompt:
        print("\nERROR: Generated SLM prompt is empty.")
        return

    print("\n")
    print("=" * 60)
    print("STEP 2: GENERATED SLM PROMPT")
    print("=" * 60)
    print(slm_prompt)

    # --------------------------------------------------
    # STEP 3: LOAD SLM
    # --------------------------------------------------
    print("\n")
    print("=" * 60)
    print("STEP 3: LOADING FINE-TUNED SLM")
    print("=" * 60)

    model, tokenizer = load_slm()

    # --------------------------------------------------
    # STEP 4: RUN SLM
    # --------------------------------------------------
    print("\n")
    print("=" * 60)
    print("STEP 4: SLM TROUBLESHOOTING RESPONSE")
    print("=" * 60)

    response = run_slm(
        model,
        tokenizer,
        slm_prompt
    )

    print("\n")
    print(response)

    print("\n")
    print("=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
