# Direct Prompt Injection Guardrail Framework

## Project Description
This repository contains an end-to-end local machine learning pipeline designed to implement and benchmark distinct NLP text-classification strategies for direct prompt injection detection. The primary objective is to evaluate a custom fine-tuned DistilBERT model against off-the-shelf baselines (such as ProtectAI's DeBERTa) and alternative paradigms (Vector Space Embeddings, local LLM-as-a-Judge) using a strictly frozen, stratified test dataset.

## Folder Structure
The workspace is decoupled to separate raw data, processing scripts, and analytical outputs:
* `data/` — Contains the strictly frozen, stratified data splits (Train, Validation, Test) in CSV format. (Ignored from remote version control cache to prevent data leakage).
* `notebooks/` — Contains Jupyter Notebooks used for Exploratory Data Analysis (EDA) and attack-type heuristic tagging.
* `results/` — Contains output artifacts including generated distribution graphs and future baseline JSON metrics.
* `src/` — Contains the core Python scripts for data pipeline ingestion and model evaluation.

## Dataset Sources
This project aggregates an isolated corpus of 5,517 rows using the following open-source Hugging Face datasets:
1. **[Deepset Prompt Injections](https://huggingface.co/datasets/deepset/prompt-injections):** High-quality, structurally diverse prompt injections and standard user queries (requires Hugging Face authentication).
2. **[Rogue Security Prompt Injections Benchmark](https://huggingface.co/datasets/rogue-security/prompt-injections-benchmark):** A dense adversarial dataset spanning multiple attack taxonomies (requires Hugging Face authentication).

## Environment Setup
This framework is built for local execution (Apple Silicon compatible). To set up the environment, clone the repository and initialize the virtual environment:

```bash
# 1. Initialize and activate the virtual environment
python3 -m venv env
source env/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Authenticate with Hugging Face (Required for the both dataset, HF access token needed)
hf auth login