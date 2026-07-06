import pandas as pd
import json
import os
import time
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_vector_db():
    print("Loading frozen train and test sets.")
    df_train = pd.read_csv("data/frozen_train_set.csv")
    df_test = pd.read_csv("data/frozen_test_set.csv")
    
    # STEP 1: Initialize the Embedding Model
    # ---------------------------------------------------------
    # all-MiniLM-L6-v2 is a fast, lightweight industry standard embedding model
    # It converts English strings into a 384-dimensional geometric coordinate.
    print("Loading SentenceTransformer (all-MiniLM-L6-v2).")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # STEP 2: Build the "Vector Database"
    # ---------------------------------------------------------
    print(f"Mapping {len(df_train)} training prompts into the vector space.")
    train_texts = df_train['text'].astype(str).tolist()
    train_labels = df_train['label'].tolist()
    
    # This creates the mathematical coordinates for our known database
    train_embeddings = encoder.encode(train_texts, show_progress_bar=True)
    
    # We use K-Nearest Neighbors (KNN) to act as our search engine.
    # Chose k=5 --> It looks for the 5 closest neighbors using Cosine Similarity (angle between vectors).
    print(" Initializing K-Nearest Neighbors classifier (k=5).")
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn.fit(train_embeddings, train_labels)
    
    # STEP 3: Run Blind Inference on the Test Set
    # ---------------------------------------------------------
    print(f" Running semantic search inference on {len(df_test)} test prompts.")
    test_texts = df_test['text'].astype(str).tolist()
    true_labels = df_test['label'].tolist()
    
    # --- LATENCY TRACKING START ---
    start_time = time.time()
    
    # We loop through 1 by 1 to simulate true sequential real-world user traffic
    predictions = []
    for text in test_texts:
        # 1. Convert incoming text to coordinate
        single_embedding = encoder.encode([text])
        # 2. Find closest neighbors in DB and predict label
        pred = knn.predict(single_embedding)[0]
        predictions.append(pred)
        
    end_time = time.time()
    # --- LATENCY TRACKING END ---
    
    # Calculate Latency
    total_inference_time = end_time - start_time
    avg_latency_ms = (total_inference_time / len(test_texts)) * 1000

    print("\n Calculating final Vector DB production metrics.")
    
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "model": "vector_db_knn_minilm",
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
        "avg_latency_ms": float(round(avg_latency_ms, 2))
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
    with open("results/vector_db_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("\n Milestone 4: Vector DB metrics locked in results/vector_db_metrics.json")

if __name__ == "__main__":
    evaluate_vector_db()