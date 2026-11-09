# Datasets

TeleAgent uses publicly available 5G telecom datasets to fine-tune the **Qwen/Qwen2.5-1.5B-Instruct** model using Low-Rank Adaptation (LoRA).

Each dataset is used independently to train a specialized LoRA adapter. The appropriate adapter is activated alongside the shared base model depending on the agent or task.

## Dataset Sources

| Dataset | Description | Source |
|---|---|---|
| 5G/4G/2G RAN Performance Management Dataset | Real-world telecom performance counters covering traffic volume, resource utilization, connected users, and energy consumption. | [Zenodo – Dataset 03](https://zenodo.org/records/17815388) |
| 5G Network Data | 5G network-related information for network analysis. | [Kaggle](https://www.kaggle.com/datasets/vinothkannaece/5g-network-data) |
| 5G Wave Channel Prediction | Wireless channel characteristics and prediction. | [Kaggle](https://www.kaggle.com/datasets/zoya77/5g-wave-channel-prediction-dataset) |
| 5G Adaptive Beamforming | Beamforming and SNR-related data for wireless optimization. | [Kaggle](https://www.kaggle.com/datasets/zoya77/5g-adaptive-beamforming-with-snr-dataset) |
| 5G Network KPI | Network performance indicators for monitoring and optimization. | [Kaggle](https://www.kaggle.com/datasets/srikumarnayak/5g-network-kpi-dataset) |

**Base Model:** [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

**Fine-Tuning:** Dataset-specific LoRA adapters using PEFT.

*Note: Original datasets are available from their respective sources and are subject to their licensing terms.*
