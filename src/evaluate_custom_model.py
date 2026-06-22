import pandas as pd
import json
import torch
import os
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

def evaluate_custom_model():
    print("Loading strictly frozen test set.")
    df_test = pd.read_csv("data/frozen_test_set.csv")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using hardware acceleration: {device}")

    print(" Loading custom DistilBERT model from local storage.")
    # Pointing to the local directory where the custom model is saved
    classifier = pipeline(
        "text-classification", 
        model="models/custom_distilbert", 
        tokenizer="models/custom_distilbert",
        device=device,
        truncation=True, 
        max_length=512
    )

    print(f"Running inference on {len(df_test)} blind test prompts.")
    texts = df_test['text'].astype(str).tolist()
    true_labels = df_test['label'].tolist()
    
    predictions = []
    for out in tqdm(classifier(texts, batch_size=16), total=len(texts)):
        # Hugging Face defaults to 'LABEL_0' (Safe) and 'LABEL_1' (Injection)
        # We extract the integer to match our true_labels
        pred_label = int(out['label'].split('_')[-1]) if 'LABEL' in out['label'] else int(out['label'])
        predictions.append(pred_label)

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
        "true_negatives": int(tn)
    }

    # Print to console
    for key, value in metrics.items():
        if key == "false_positive_rate":
            print(f"   -> {key.replace('_', ' ').title()}: {value} ({value*100:.2f}%)")
        else:
            print(f"   -> {key.replace('_', ' ').title()}: {value}")

    os.makedirs("results", exist_ok=True)
    with open("results/custom_distilbert_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("\n Milestone 3 Success: Final metrics locked in results/custom_distilbert_metrics.json")

if __name__ == "__main__":
    evaluate_custom_model()