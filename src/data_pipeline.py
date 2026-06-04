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

    # 4. Generate the 3-Way Stratified Split (64% Train, 16% Val, 20% Test)
    # The random_state=42 locks the shuffle so it is perfectly reproducible (ie: every time script is run, same data split instead of diff split)
    # FIRST SPLIT: Carve out the 20% Test Set
    train_val_df, test_df = train_test_split(
        df_combined, 
        test_size=0.20, 
        random_state=42, 
        stratify=df_combined['label']
    )
    
    # SECOND SPLIT: Split the remaining 80% into Train (64%) and Validation (16%)
    # By taking 20% of the 80% chunk, we isolate exactly 16% for validation
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=0.20, 
        random_state=42, 
        stratify=train_val_df['label']
    )
    # 5. Lock and Freeze the partitions to CSV formats
    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/frozen_train_set.csv", index=False)
    val_df.to_csv("data/frozen_val_set.csv", index=False)
    test_df.to_csv("data/frozen_test_set.csv", index=False)
    
    print("\n Milestone 2a Update Success: 3-Way Splits is frozen!")
    print(f"   -> Train Set:      {len(train_df)} rows (~64%)")
    print(f"   -> Validation Set: {len(val_df)} rows (~16%)")
    print(f"   -> Test Set:       {len(test_df)} rows (~20%)")

if __name__ == "__main__":
    prepare_and_freeze_data()