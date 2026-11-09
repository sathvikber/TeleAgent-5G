# TeleAgent: Autonomous Multi-Agent Framework for 5G Core Diagnostics

**A multi-agent AI assistant for monitoring and troubleshooting Private 5G networks at the edge.**

TeleAgent combines specialised agents with a fine-tuned Telecom Small Language Model (SLM). Each agent does the deterministic work first (parsing, threshold checks, retrieval, classification), and the SLM then reasons only over the evidence it is given. This keeps answers grounded, reduces unnecessary inference, and lets the system run on CPU-only MEC hardware.

## Features

- **Multi-agent orchestration**: a coordinator classifies each query, plans tasks, and runs the relevant agents in parallel.
- **Four specialised agents**
  - **Knowledge**: telecom Q&A and standards search using hybrid retrieval (FAISS + BM25, fused with Reciprocal Rank Fusion).
  - **Log**: log parsing, event classification and pattern detection.
  - **Monitoring**: KPI and resource analysis, threshold checks and anomaly detection, with live data from the Niral Networks API.
  - **Network**: registration, authentication, mobility, PDU session, NGAP/NAS and QoS troubleshooting.
- **Telecom SLM**: Qwen2.5-1.5B-Instruct fine-tuned with LoRA on telecom data.
- **Multilingual**: answers in Hindi when asked in Hindi.
- **Edge-ready**: Dockerised, with CPU inference.

## Architecture

```
User query
    │
Coordinator ── Intent Analyzer → Task Planner → Dispatcher
    │
    ├── Knowledge Agent ── Hybrid RAG
    ├── Log Agent ──────── Log analysis pipeline
    ├── Monitoring Agent ─ Metric analysis pipeline
    └── Network Agent ──── Issue analysis pipeline
    │
Telecom SLM (Qwen2.5-1.5B + LoRA)
    │
Aggregator → Response
```

Vendor-specific integrations such as Niral live in client and adapter modules, not inside the agents.

## Knowledge Base

The Knowledge Agent retrieves from a curated corpus, stored in `knowledge_base/` and chunked for the hybrid RAG pipeline. It covers two areas:

- **Indian AI policy and governance:** national AI guidelines and strategy (MeitY, NITI Aayog), TRAI recommendations on AI in telecom, the Digital Personal Data Protection Act and Rules, and responsible-AI, robustness and fairness standards (TEC).
- **Telecom AI standards and agentic networks:** ITU-T recommendations on ML architecture and AI-enhanced operations (Y.3172, M.3080), and ETSI ENI, NGMN and TM Forum publications on multi-agent frameworks, agent interfaces, MLOps and agentic security.

## Fine-tuning Datasets

The Telecom SLM (`Qwen/Qwen2.5-1.5B-Instruct`) was fine-tuned on the 5G datasets in the `datasets/` folder: 5G network data, 5G network KPIs, 5G wave channel prediction, and 5G adaptive beamforming with SNR.

Each dataset is used independently to train a specialized LoRA adapter. The appropriate adapter is activated alongside the shared base model depending on the agent or task.

These datasets are not part of the knowledge base, since the SLM is already trained on them.

## Getting Started

### Prerequisites

- Docker
- The TeleAgent repository and the LoRA model directory (`slm-lora/`)

### Installation

```bash
git clone https://github.com/sathvikber/TeleAgent-5G.git
cd TeleAgent-5G
docker build -t teleagent:cpu .
```

The image installs every package listed in `requirements.txt`.

### Run

```bash
docker run --rm -it \
  -v "$PWD":/workspace \
  -v "$PWD"/slm-lora:/workspace/slm-lora \
  -v "$PWD"/knowledge_base:/workspace/knowledge_base \
  -w /workspace \
  teleagent:cpu

# inside the container
python -m teleagent.main
```

This loads the SLM and opens an interactive prompt. Ask in English or Hindi, for example:

```
> UE registration failed and CPU utilisation is above 95%
> What is the role of the AMF in 5G core?
```

### Live monitoring (optional)

The Monitoring Agent can analyse live data from a **Niral Networks** Private 5G testbed. Niral is an external system, not part of TeleAgent, so nothing from Niral is installed inside the TeleAgent container. To use live monitoring:

1. The Niral dashboard/backend must be **running and reachable** from the machine running Docker (for example `http://<host>:8082`).
2. You need a valid Niral API token.
3. Create a `.env.niral` file and pass it to Docker with `--env-file .env.niral`:

```
NIRAL_API_BASE_URL=http://<host>:8082
NIRAL_API_TOKEN=<token>
```

Optional settings: `NIRAL_API_TIMEOUT` (seconds) and `NIRAL_API_RETRIES`.

If `NIRAL_API_BASE_URL` is not set, TeleAgent does not contact Niral and the Monitoring Agent works in offline mode on metrics you provide as text. Never commit the token.

## Testing

Run the tests inside the container:

```bash
python -m pytest -q
```

## Status and Roadmap

The core platform is complete: the SLM, coordinator, four specialised agents, hybrid RAG, and live Niral monitoring.

| Next | Focus |
|---|---|
| Intent and task planning | Avoid unnecessary multi-agent tasks |
| Multi-agent collaboration | Share evidence between agents before reasoning |
| Memory | Session and historical incident context |
| Real-time telemetry | Prometheus, Grafana and streaming log integration |
| Topology-aware RCA | Network-function topology and root cause analysis |
| Benchmarking | Latency, accuracy and production deployment |

## Known Limitations

- gNB and UE throughput data still needs validation under active traffic.
- The task planner can create more agent tasks than a query needs.
- The knowledge base currently holds test documents and needs real telecom documentation.

