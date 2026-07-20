import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from datasets import Dataset

# ASYMMETRIC LOSS TRAINER
# ---------------------------------------------------------
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            class_weights = self.class_weights.to(model.device)
            loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
        return (loss, outputs) if return_outputs else loss

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
        "fpr": float(fpr)
    }

def run_w_benign_sweep():
    print("⏳ Loading strictly frozen Train and Validation datasets...")
    train_df = pd.read_csv("data/frozen_train_set.csv")
    val_df = pd.read_csv("data/frozen_val_set.csv")
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased", truncation_side="left")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        
    print(" Tokenizing datasets.")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    
    
    # W_BENIGN SWEEP (Asymmetric Loss Tuning)
    # ---------------------------------------------------------
    w_malicious = 1.0
    w_benign_candidates = [2.0, 2.5, 3.0]
    
    # Static hyperparameters from Milestone 3 baseline
    static_lr = 2e-5
    static_wd = 0.01
    
    sweep_results = []
    best_f1 = 0.0
    best_w_benign = None

    for w_benign in w_benign_candidates:
        print(f"\n=======================================================")
        print(f" RUNNING SWEEP: w_benign = {w_benign} | w_malicious = {w_malicious}")
        print(f"=======================================================")
        
        # Reset model for a fresh training loop
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        w_tensor = torch.tensor([w_benign, w_malicious], dtype=torch.float)
        
        training_args = TrainingArguments(
            output_dir=f"models/sweep_w_benign_{w_benign}",
            eval_strategy="epoch",            
            save_strategy="epoch",            
            learning_rate=static_lr,         
            per_device_train_batch_size=16,   
            per_device_eval_batch_size=16,
            num_train_epochs=3,               
            weight_decay=static_wd,          
            load_best_model_at_end=True,      
            metric_for_best_model="f1",       
            greater_is_better=True,          
            logging_dir=None,
            report_to="none" # Keep console clean
        )
        
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            compute_metrics=compute_metrics,
            class_weights=w_tensor
        )
        
        trainer.train()
        
        # Get validation results
        eval_metrics = trainer.evaluate()
        val_f1 = eval_metrics['eval_f1']
        val_fpr = eval_metrics['eval_fpr']
        
        print(f"\n Sweep Complete for w_benign={w_benign}")
        print(f"   -> Validation F1:  {val_f1 * 100:.2f}%")
        print(f"   -> Validation FPR: {val_fpr * 100:.2f}%")
        
        sweep_results.append({
            "w_benign": w_benign,
            "f1_score": val_f1,
            "fpr": val_fpr
        })
        
        # Check against enterprise constraints
        if val_fpr < 0.05 and val_f1 > best_f1:
            best_f1 = val_f1
            best_w_benign = w_benign
            print(f"    NEW BEST VALID MODEL: w_benign={w_benign} saves to disk!")
            os.makedirs("models/custom_distilbert_w_benign_opt", exist_ok=True)
            trainer.save_model("models/custom_distilbert_w_benign_opt")
            tokenizer.save_pretrained("models/custom_distilbert_w_benign_opt")

    print("\n W_BENIGN SWEEP LEADERBOARD:")
    for res in sweep_results:
        status = "PASSED" if res['fpr'] < 0.05 else "FAILED (FPR >= 5%)"
        print(f"   w_benign: {res['w_benign']} | F1: {res['f1_score']*100:.2f}% | FPR: {res['fpr']*100:.2f}% | {status}")
        
    if best_f1 > 0:
        print(f"\n Optimized model (w_benign={best_w_benign}) successfully saved to models/custom_distilbert_w_benign_opt/")
    else:
        print(f"\n No model beat the constraints. We may need to test different boundaries.")

if __name__ == "__main__":
    run_w_benign_sweep()