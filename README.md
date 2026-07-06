# Direct Prompt Injection Guardrail Framework

## Project Description
This repository contains an end-to-end local machine learning pipeline designed to implement and benchmark distinct NLP text-classification strategies for direct prompt injection detection. The primary objective is to evaluate a custom fine-tuned DistilBERT model against off-the-shelf baselines (such as ProtectAI's DeBERTa) and alternative paradigms (Vector Space Embeddings, local LLM-as-a-Judge) using a strictly frozen, stratified test dataset.

## Folder Structure
The workspace is decoupled to separate raw data, processing scripts, and analytical outputs:
* `data/` — Contains the strictly frozen, stratified data splits (Train, Validation, Test) in CSV format. (Ignored from remote version control cache to prevent data leakage).
* `notebooks/` — Contains Jupyter Notebooks used for Exploratory Data Analysis (EDA) and attack-type heuristic tagging.
* `results/` — Contains output artifacts including generated distribution graphs, per-model JSON evaluation metrics, and the unified `master_benchmark_report.json` leaderboard.
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

# 2. Install all dependencies
# Covers the full pipeline: data ingestion, DistilBERT training/evaluation,
# Paradigm A (Vector DB), and Paradigm B (LLM-as-a-Judge). Key packages:
# transformers, torch, datasets, pandas, scikit-learn, sentence-transformers,
# and llama-cpp-python.
pip install -r requirements.txt

# 3. Authenticate with Hugging Face (Required for both datasets and to
#    download the Phi-3 GGUF weights; an HF access token is needed)
hf auth login

> **Apple Silicon acceleration (optional):** the wheel installed above runs
> `llama-cpp-python` on CPU. For GPU acceleration on a Mac, recompile it with the
> Metal (MPS) backend enabled:
>
> ```bash
> CMAKE_ARGS="-DGGML_METAL=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
> ```

## Milestone 3: Production Model & Test Metrics

The primary DistilBERT model has been successfully trained, fine-tuned, and validated. Due to file size constraints, the final trained weights and tokenizer configurations are securely hosted on the Hugging Face Hub.

📦 **Hugging Face Model Repository:** [thienyu/prompt-injection-guardrail](https://huggingface.co/thienyu/prompt-injection-guardrail)

### Final Production Test Results (Blind 20% Dataset)
- **Accuracy:** 87.95%
- **Precision:** 92.48%
- **Recall:** 75.80%
- **F1-Score:** 83.31%
- **False Positive Rate (FPR):** 4.05% *(Satisfies strict < 5% constraint)*

### Local Inference Verification
To verify the basic inference loop locally, ensure your environment is activated and execute:
```bash
python src/evaluate_custom_model.py
```

## Milestone 4: Alternative Paradigm Benchmarking

Milestone 4 benchmarks the custom fine-tuned DistilBERT model against two alternative machine learning paradigms, evaluating each against the frozen 20% Test Set (1,104 prompts). The goal is to mathematically quantify the trade-offs between inference latency, detection accuracy, and adherence to the strict contract constraint (False Positive Rate < 5%).

### Paradigm A: Embedding Vector Database (Semantic Search)
Instead of a neural network forward pass, this paradigm classifies text by geometric distance. The `all-MiniLM-L6-v2` sentence transformer maps the entire training set into a 384-dimensional vector space, and incoming test prompts are classified using K-Nearest Neighbors (KNN, k=5) based on cosine similarity.

- **Script:** `src/vector_db_classifier.py`
- **F1-Score:** 79.84% · **FPR:** 21.92% · **Recall:** 88.58% · **Precision:** 72.66% · **Latency:** 12.96 ms/prompt
- **Analysis:** Extremely fast, but **FAILED** the FPR constraint. Benign queries with complex instructions (e.g. *"Ignore my previous typo and write a story"*) share heavy vocabulary with malicious jailbreaks, so their coordinates sit too close in vector space. A KNN distance metric cannot be asymmetrically weighted to protect the user experience the way DistilBERT's weighted loss function can.

### Paradigm B: Local LLM-as-a-Judge (Zero-Shot Generative AI)
This paradigm deploys Microsoft's `Phi-3-mini-4k-instruct` (Q4 Quantized) locally on Apple Silicon via `llama.cpp`. The model is given a strict system prompt to act as a cybersecurity guardrail and output a binary `SAFE` / `ATTACK` classification in a zero-shot manner.

- **Script:** `src/llm_judge_classifier.py`
- **F1-Score:** 68.44% · **FPR:** 15.92% · **Recall:** 64.61% · **Precision:** 72.75% · **Latency:** 539.10 ms/prompt
- **DoS Mitigation:** A 2,254-token prompt (#407) originally overflowed the context window (simulating a Denial-of-Service memory overload). This was fixed by (1) expanding `n_ctx` to the maximum 4,096 tokens and (2) applying a left-truncation slicing strategy that keeps only the final 2,000 words of oversized prompts, preserving the payload while preventing crashes.
- **Analysis:** **FAILED** on both performance and latency (~35x slower than DistilBERT). Because LLMs are designed to follow instructions, they are susceptible to being manipulated by the very injections they are meant to classify — DistilBERT avoids this because it mathematically classifies text rather than "reading" it to obey.

### Automated Benchmarking Aggregator
`src/automated_benchmark.py` sweeps the `results/` directory, extracts the isolated JSON metrics from all four models (ProtectAI, DistilBERT, Vector DB, and Phi-3), compiles a unified `results/master_benchmark_report.json`, and prints a Pandas-formatted leaderboard with an automatic contract verdict.

```bash
# Run individual paradigm evaluations
python src/vector_db_classifier.py
python src/llm_judge_classifier.py

# Aggregate all metrics into the master leaderboard
python src/automated_benchmark.py
```

### Final Architectural Standings (Frozen 20% Test Set)

| Model Architecture | F1-Score | FPR (Constraint < 5%) | Recall | Precision | Latency/Prompt | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| ProtectAI Baseline (DeBERTa) | 66.05% | 21.32% | 65.30% | 66.82% | — | FAILED |
| **Custom DistilBERT** | **83.31%** | **4.05%** | 75.80% | 92.48% | 10.90 ms | **PASSED** |
| Vector DB (KNN) | 79.84% | 21.92% | 88.58% | 72.66% | 12.96 ms | FAILED |
| LLM-as-a-Judge (Phi-3) | 68.44% | 15.92% | 64.61% | 72.75% | 539.10 ms | FAILED |

### Final Verdict
The Custom DistilBERT model is the **only** architecture that satisfies the contract constraint (FPR < 5%) while maintaining a high F1-Score and low latency. The failures of both semantic search and generative reasoning mathematically validate the necessity of an asymmetric loss function and dedicated classification heads for robust LLM security.

> **Note:** DistilBERT's F1-Score (83.31%) still falls ~1.6% short of the ≥85% target. Milestone 5 will address this by tuning hyperparameters, epoch count, and learning rate to reach the 85% F1 goal.

