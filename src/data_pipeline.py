import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def prepare_and_freeze_data():
    print("Loading raw data streams from Hugging Face...")
    
    # 1. Fetch Deepset (contains ~662 rows)
    deepset_raw = load_dataset("deepset/prompt-injections", split="train")
    df_deepset = pd.DataFrame(deepset_raw)[['text', 'label']]

    # 2. Fetch Rogue Security Benchmark (contains ~5,000 rows)
    rogue_raw = load_dataset("rogue-security/prompt-injections-benchmark", split="test")
    df_rogue = pd.DataFrame(rogue_raw)
    
    # Map string labels to binary integers (0 = benign, 1 = injection)
    df_rogue['label'] = df_rogue['label'].map({'benign': 0, 'jailbreak': 1})
    df_rogue = df_rogue[['text', 'label']]

    # 3. Concatenate and deduplicate the data
    df_combined = pd.concat([df_deepset, df_rogue], ignore_index=True)
    df_combined.dropna(subset=['text', 'label'], inplace=True)
    df_combined['label'] = df_combined['label'].astype(int)
    df_combined.drop_duplicates(subset=['text'], inplace=True)
    
    print(f"Total Aggregated Corpus Size: {len(df_combined)} rows.")

    # 4. Generate Train and Test Splits (80/20)
    # The random_state=42 locks the shuffle so it is perfectly reproducible
    train_df, test_df = train_test_split(
        df_combined, 
        test_size=0.20, 
        random_state=42, 
        stratify=df_combined['label']
    )

    # 5. Lock and Freeze the partitions to CSV formats
    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/frozen_train_set.csv", index=False)
    test_df.to_csv("data/frozen_test_set.csv", index=False)
    
    print("Milestone 2a Success: Splits strictly frozen in data/ directory!")

if __name__ == "__main__":
    prepare_and_freeze_data()