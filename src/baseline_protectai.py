import pandas as pd
import json
import torch
import os
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

# transformers to load a pre-trained model and data tokenizer
# pytorch to define the optimization logic, loss function and backpropagation
# sklearn to calculate the performance metrics
# tqdm to show a progress bar


def run_protectai_baseline():
    print("Loading frozen test set...")
    # 1. Load the frozen 20% test partition
    df_test = pd.read_csv("data/frozen_test_set.csv")
    
    # 2. Hardware Acceleration
    # Automatically utilizes Apple Silicon (MPS) if available for massive speedups
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using hardware acceleration: {device}")

    # 3. Initialize the ProtectAI Pipeline
    print("Downloading and loading ProtectAI DeBERTa model...")
    classifier = pipeline(
        "text-classification", 
        model="protectai/deberta-v3-base-prompt-injection-v2", 
        device=device,
        truncation=True, # Crucial: Truncates extremely long prompts to fit model memory
        max_length=512
    )

    # 4. Run Predictions
    print(f"Evaluating {len(df_test)} test prompts. Please wait a minute...")
    
    # Extract just the text column as a list
    texts = df_test['text'].astype(str).tolist()
    true_labels = df_test['label'].tolist()
    
    # Run the pipeline over the texts with a progress bar
    predictions = []
    for out in tqdm(classifier(texts, batch_size=16), total=len(texts)):
        # ProtectAI outputs 'SAFE' or 'INJECTION'
        # We must map this to our binary 0 and 1 schema
        pred_label = 1 if out['label'] == 'INJECTION' else 0
        predictions.append(pred_label)

    # Extract True Negatives, False Positives, False Negatives, and True Positives
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    
    # Calculate False Positive Rate: FP / Total Actual Negatives
    fpr = fp / (fp + tn)
    # 5. Calculate ML Metrics
    print("\n Calculating performance metrics...")
    metrics = {
        "model": "protectai/deberta-v3-base-prompt-injection-v2",
        "dataset_size": len(df_test),
        "accuracy": round(accuracy_score(true_labels, predictions), 4),
        "precision": round(precision_score(true_labels, predictions), 4),
        "recall": round(recall_score(true_labels, predictions), 4),
        "f1_score": round(f1_score(true_labels, predictions), 4),
        "false_positive_rate": round(fpr, 4),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "true_negative": int(tn)
    }

    # Print to console
    for key, value in metrics.items():
        print(f"   -> {key.capitalize()}: {value}")

    # 6. Save results to the framework
    os.makedirs("results", exist_ok=True)
    with open("results/baseline_protectai_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("\n Milestone 2b Baseline Success: Metrics locked in results/baseline_protectai_metrics.json")

if __name__ == "__main__":
    run_protectai_baseline()