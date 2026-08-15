import pandas as pd
import json
import torch
import os
import time
import statistics
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

# Match DistilBERT eval protocol (evaluate_custom_model.py):
# MPS, truncation=True, max_length=512, batch_size=1, wall-clock mean ms/prompt.
# Plus professor refinements: discard warm-up passes; report median alongside mean.


WARMUP_N = 5  # discard first-call MPS graph compilation before timing


def run_protectai_baseline():
    print("Loading frozen test set...")
    df_test = pd.read_csv("data/frozen_test_set.csv")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using hardware acceleration: {device}")

    print("Downloading and loading ProtectAI DeBERTa model...")
    classifier = pipeline(
        "text-classification",
        model="protectai/deberta-v3-base-prompt-injection-v2",
        device=device,
        truncation=True,
        max_length=512,
    )

    texts = df_test["text"].astype(str).tolist()
    true_labels = df_test["label"].tolist()
    n = len(texts)
    print(f"Evaluating {n} test prompts (batch_size=1, sequential)...")

    # Warm-up: compile MPS graph / caches without counting toward latency
    warmup_n = min(WARMUP_N, n)
    print(f"Warming up on {warmup_n} prompts (not timed)...")
    for i in range(warmup_n):
        _ = classifier(texts[i], batch_size=1)

    predictions = []
    latencies_ms = []

    for text in tqdm(texts, total=n):
        t0 = time.perf_counter()
        out = classifier(text, batch_size=1)[0]
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

        # ProtectAI outputs 'SAFE' or 'INJECTION'
        pred_label = 1 if out["label"] == "INJECTION" else 0
        predictions.append(pred_label)

    avg_latency_ms = statistics.mean(latencies_ms)
    median_latency_ms = statistics.median(latencies_ms)

    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("\n Calculating performance metrics...")
    metrics = {
        "model": "protectai/deberta-v3-base-prompt-injection-v2",
        "dataset_size": int(n),
        "accuracy": float(round(accuracy_score(true_labels, predictions), 4)),
        "precision": float(round(precision_score(true_labels, predictions), 4)),
        "recall": float(round(recall_score(true_labels, predictions), 4)),
        "f1_score": float(round(f1_score(true_labels, predictions), 4)),
        "false_positive_rate": float(round(fpr, 4)),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "avg_latency_ms": float(round(avg_latency_ms, 2)),
        "median_latency_ms": float(round(median_latency_ms, 2)),
        "warmup_prompts_discarded": int(warmup_n),
        "device": str(device),
    }

    for key, value in metrics.items():
        if key == "false_positive_rate":
            print(f"   -> {key.replace('_', ' ').title()}: {value} ({value * 100:.2f}%)")
        elif key in ("avg_latency_ms", "median_latency_ms"):
            print(f"   -> {key.replace('_', ' ').title()}: {value} ms/prompt")
        else:
            print(f"   -> {key.replace('_', ' ').title()}: {value}")

    os.makedirs("results", exist_ok=True)
    out_path = "results/baseline_protectai_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n Milestone 2b Baseline Success: Metrics locked in {out_path}")


if __name__ == "__main__":
    run_protectai_baseline()
