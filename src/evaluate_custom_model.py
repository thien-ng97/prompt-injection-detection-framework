import pandas as pd
import json
import torch
import os
import time
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

def evaluate_custom_model():
    print("Loading strictly frozen test set.")
    df_test = pd.read_csv("data/frozen_test_set.csv")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using hardware acceleration: {device}")

    hf_model_id = "thienyu/prompt-injection-guardrail"
    print(f" Loading custom DistilBERT model from {hf_model_id}.")
    print("   (If not cached locally, this will automatically download the weights from Hugging Face)")
    
    classifier = pipeline(
        "text-classification", 
        model=hf_model_id,         # Reproducibility: replace "models/custom_distilbert" with author's Hugging Face model ID
        tokenizer=hf_model_id,     # If you don't have the model cached locally, it will automatically download from Hugging Face
        device=device,             # If you ran the code from the beginning, replace model=hf_model_id with "models/custom_distilbert"
        truncation=True, 
        max_length=512
    )

    print(f"Running inference on {len(df_test)} blind test prompts.")
    texts = df_test['text'].astype(str).tolist()
    true_labels = df_test['label'].tolist()
    
    predictions = []

    # --- LATENCY TRACKING START ---
    start_time = time.time()

    # Force batch_size=1 to simulate real-world, sequential user requests
    for out in tqdm(classifier(texts, batch_size=1), total=len(texts)):
        # Hugging Face defaults to 'LABEL_0' (Safe) and 'LABEL_1' (Injection)
        # We extract the integer to match our true_labels
        pred_label = int(out['label'].split('_')[-1]) if 'LABEL' in out['label'] else int(out['label'])
        predictions.append(pred_label)

    end_time = time.time()
    # --- LATENCY TRACKING END ---

    # Calculate Latency Metrics
    total_inference_time = end_time - start_time
    avg_latency_ms = (total_inference_time / len(texts)) * 1000

    print("\n Calculating final production metrics.")
    
    # Extract native Python counts for JSON safety
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "model": "custom_distilbert_v1",
        "dataset_size": int(len(df_test)),
        "accuracy": float(round(accuracy_score(true_labels, predictions), 4)),
        "precision": float(round(precision_score(true_labels, predictions), 4)),
        "recall": float(round(recall_score(true_labels, predictions), 4)),
        "f1_score": float(round(f1_score(true_labels, predictions), 4)),
        "false_positive_rate": float(round(fpr, 4)),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "avg_latency_ms": float(round(avg_latency_ms, 2)) # <-- Add to JSON
    }

    # Print to console
    for key, value in metrics.items():
        if key == "false_positive_rate":
            print(f"   -> {key.replace('_', ' ').title()}: {value} ({value*100:.2f}%)")
        elif key == "avg_latency_ms":
             print(f"   -> {key.replace('_', ' ').title()}: {value} ms/prompt")
        else:
            print(f"   -> {key.replace('_', ' ').title()}: {value}")

    os.makedirs("results", exist_ok=True)
    with open("results/custom_distilbert_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("\n Milestone 3 Success: Final metrics locked in results/custom_distilbert_metrics.json")

if __name__ == "__main__":
    evaluate_custom_model()