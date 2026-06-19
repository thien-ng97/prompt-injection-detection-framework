import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from datasets import Dataset

# STRATEGY 2: Asymmetric Class Weights (Custom Trainer - Addressing FPR)
# ---------------------------------------------------------
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    # We override the default mathematical loss function to apply our 'w' coefficients
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            # Move weights to the exact same hardware device as the model (MPS/CPU)
            class_weights = self.class_weights.to(model.device)
            loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
        return (loss, outputs) if return_outputs else loss

# Grader for validation loop
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions)),
        "recall": float(recall_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions)),
        "fpr": float(fpr)  # Tracking this to ensure it drops below 5%
    }

def train_custom_model():
    print("Loading frozen Train and Validation datasets")
    train_df = pd.read_csv("data/frozen_train_set.csv")
    val_df = pd.read_csv("data/frozen_val_set.csv")
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    print("Initializing DistilBERT Tokenizer")
    # STRATEGY 1: Left Truncation (To address 512 token context window limit)
    # ---------------------------------------------------------
    # Chops off the harmless beginning (instead of the end) of 1000-word jailbreaks, 
    # forcing the model to read the actual malicious payload at the end.
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "distilbert-base-uncased", 
        truncation_side="left"
    )
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        
    print("Tokenizing datasets. This will take a moment.")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    
    print("Initializing base DistilBERT Model.")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", 
        num_labels=2
    )
    
    # STRATEGY 2 Math Setup: Asymmetric 'w' coefficients
    # ---------------------------------------------------------
    # We heavily penalize the model mathematically if it triggers a False Positive.
    w_benign = 3.0       # 3x penalty for misclassifying a benign prompt (Class 0)
    w_malicious = 1.0    # 1x penalty for misclassifying an attack (Class 1)
    w_tensor = torch.tensor([w_benign, w_malicious], dtype=torch.float)
    print(f"Setting class weights: w_benign={w_benign}, w_malicious={w_malicious}")
    
    # STRATEGY 3: The Validation Loop Configuration
    # ---------------------------------------------------------
    training_args = TrainingArguments(
        output_dir="models/custom_distilbert_checkpoints",
        eval_strategy="epoch",            # Grade the model on the Val set at the end of every epoch
        save_strategy="epoch",            # Save checkpoints incrementally
        learning_rate=2e-5,
        per_device_train_batch_size=16,   # Optimal batch size for Mac MPS
        per_device_eval_batch_size=16,
        num_train_epochs=3,               # Complete 3 full passes of the training data
        weight_decay=0.01,
        load_best_model_at_end=True,      # Automatically reload the best epoch at the very end
        metric_for_best_model="fpr",      # Pick the model strictly based on the lowest FPR
        greater_is_better=False,          # We want FPR to go down (lower FPR is better)
       
    )
    
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        class_weights=w_tensor
    )
    
    print("Starting GPU training on Apple MPS.")
    trainer.train()
    
    print("\n Saving final custom model.")
    os.makedirs("models/custom_distilbert", exist_ok=True)
    trainer.save_model("models/custom_distilbert")
    tokenizer.save_pretrained("models/custom_distilbert")
    print("Custom model successfully built and saved to models/custom_distilbert/")


    # Exporting the Epoch Logs to JSON
    print("\n Exporting epoch training logs to JSON.")
    log_history = trainer.state.log_history
    
    os.makedirs("results", exist_ok=True)
    with open("results/training_epoch_logs.json", "w") as f:
        json.dump(log_history, f, indent=4)
        
    print("Epoch metrics locked in results/training_epoch_logs.json")

if __name__ == "__main__":
    train_custom_model()