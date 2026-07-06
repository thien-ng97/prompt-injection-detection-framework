import json
import os
import pandas as pd

def generate_master_benchmark():
    print(" Initializing Automated Benchmarking Aggregator.")
    
    # 1. Define the expected output files from evaluation scripts
    target_files = {
        "ProtectAI Baseline (DeBERTa)": "results/baseline_protectai_metrics.json",
        "Custom DistilBERT": "results/custom_distilbert_metrics.json",
        "Vector DB (KNN)": "results/vector_db_metrics.json",
        "LLM-as-a-Judge (Phi-3)": "results/llm_judge_metrics.json"
    }
    
    master_metrics = []
    
    # 2. Extract the data from the individual JSON files
    for model_name, filepath in target_files.items():
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
                data["display_name"] = model_name
                master_metrics.append(data)
        else:
            print(f" Warning: Could not find metrics for {model_name} at {filepath}.")
            print(f"   Make sure you have run its evaluation script first!")
            
    if not master_metrics:
        print(" No metric files found. Run the evaluation scripts first.")
        return
        
    # 3. Save the unified master JSON
    os.makedirs("results", exist_ok=True)
    master_filepath = "results/master_benchmark_report.json"
    
    # Clean up the dictionaries before saving
    clean_json_data = [{k: v for k, v in m.items() if k != "display_name"} for m in master_metrics]
    
    with open(master_filepath, "w") as f:
        json.dump(clean_json_data, f, indent=4)
        
    print(f"\n Unified metrics successfully saved to {master_filepath}\n")
    
    # 4. Generate a console leaderboard using Pandas
    print(" FINAL ARCHITECTURAL STANDINGS ")
    print("-" * 85)
    
    df = pd.DataFrame(master_metrics)
    
    # Select and format columns for the terminal display
    display_df = df[[
        "display_name", "f1_score", "false_positive_rate", "recall", "precision", "avg_latency_ms"
    ]].copy()
    
    display_df["f1_score"] = (display_df["f1_score"] * 100).round(2).astype(str) + "%"
    display_df["false_positive_rate"] = (display_df["false_positive_rate"] * 100).round(2).astype(str) + "%"
    display_df["recall"] = (display_df["recall"] * 100).round(2).astype(str) + "%"
    display_df["precision"] = (display_df["precision"] * 100).round(2).astype(str) + "%"
    display_df["avg_latency_ms"] = display_df["avg_latency_ms"].round(2).astype(str) + " ms"
    
    display_df.columns = ["Model Architecture", "F1-Score", "FPR (Constraint <5%)", "Recall", "Precision", "Latency/Prompt"]
    
    print(display_df.to_string(index=False))
    print("-" * 85)
    
    # 5. Business Logic Output
    print("\n  CONTRACT VERDICT:")
    for idx, row in df.iterrows():
        # Check against our < 5% FPR constraint
        if row["false_positive_rate"] < 0.05:
            print(f" {row['display_name']}: PASSED (Safe for Production)")
        else:
            print(f" {row['display_name']}: FAILED (FPR >= 5.0%)")
            
    print("\n")

if __name__ == "__main__":
    generate_master_benchmark()