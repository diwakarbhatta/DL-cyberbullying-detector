"""
Data Downloader, Verifier, and Preprocessor
Downloads the Cyberbullying Classification dataset, verifies integrity, checks class distributions,
and creates clean stratified train/val/test splits.
"""

import os
import sys
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "cyberbullying_tweets.csv")
DATA_URL = "https://raw.githubusercontent.com/am-shb/cyberbullying-detection/main/data/cyberbullying_tweets.csv"

def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_DATA_PATH) and os.path.getsize(RAW_DATA_PATH) > 1000000:
        print(f"Dataset already downloaded at: {RAW_DATA_PATH}")
        return
        
    print(f"Downloading dataset from: {DATA_URL} ...")
    urllib.request.urlretrieve(DATA_URL, RAW_DATA_PATH)
    print(f"Download complete: {os.path.getsize(RAW_DATA_PATH) / 1024 / 1024:.2f} MB saved to {RAW_DATA_PATH}")

def verify_and_split():
    print("\n" + "=" * 60)
    print("VERIFYING DATASET INTEGRITY & CLASS DISTRIBUTIONS")
    print("=" * 60)

    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Total rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    # Check for missing values
    null_counts = df.isnull().sum()
    print("\nMissing Values:")
    print(null_counts)

    # Rename columns if needed
    # Usually: tweet_text, cyberbullying_type
    text_col = "tweet_text" if "tweet_text" in df.columns else df.columns[0]
    label_col = "cyberbullying_type" if "cyberbullying_type" in df.columns else df.columns[1]

    # Remove duplicates
    initial_len = len(df)
    df = df.drop_duplicates(subset=[text_col])
    dedup_len = len(df)
    print(f"\nDuplicates removed: {initial_len - dedup_len:,} (Remaining: {dedup_len:,})")

    # Class distribution analysis
    print("\nClass Value Counts & Percentages:")
    class_counts = df[label_col].value_counts()
    class_pct = df[label_col].value_counts(normalize=True) * 100
    dist_df = pd.DataFrame({"Count": class_counts, "Percentage (%)": class_pct.round(2)})
    print(dist_df)

    # Clean text slightly (strip whitespace)
    df = df[df[text_col].str.strip().str.len() > 2].copy()

    # Stratified 80/10/10 Train / Val / Test split
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df[label_col]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df[label_col]
    )

    print("\nDataset Split Summary:")
    print(f"  • Train: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  • Val:   {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  • Test:  {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")

    # Save splits
    train_path = os.path.join(DATA_DIR, "train.csv")
    val_path = os.path.join(DATA_DIR, "val.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSplits successfully saved to:")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print(f"  - {test_path}")
    print("=" * 60)

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    download_dataset()
    verify_and_split()
