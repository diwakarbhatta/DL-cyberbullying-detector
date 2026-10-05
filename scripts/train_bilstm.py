"""
Train and Evaluate BiLSTM Deep Learning Model for Cyberbullying Detection
Uses PyTorch, Macro-F1 evaluation, and exports confusion matrix and weights.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.bilstm_model import Vocab, CyberbullyingBiLSTM
from src.preprocessor import clean_text, normalize_leetspeak

# Ensure UTF-8 output encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

class TextDataset(Dataset):
    def __init__(self, texts: list, labels: list, vocab: Vocab, label2idx: dict, max_length: int = 100):
        self.encoded = [vocab.encode(t, max_length=max_length) for t in texts]
        self.labels = [label2idx[l] for l in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encoded[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

def evaluate(model, loader, device, classes):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))

    return macro_f1, weighted_f1, acc, cm, all_preds, all_labels

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"TRAINING DEEP LEARNING MODEL: BiLSTM (Device: {device})")
    print("=" * 60)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    # Clean text
    print("Preprocessing text...")
    train_texts = [normalize_leetspeak(clean_text(t)) for t in train_df["tweet_text"].astype(str)]
    val_texts = [normalize_leetspeak(clean_text(t)) for t in val_df["tweet_text"].astype(str)]
    test_texts = [normalize_leetspeak(clean_text(t)) for t in test_df["tweet_text"].astype(str)]

    classes = sorted(train_df["cyberbullying_type"].unique().tolist())
    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {i: c for i, c in enumerate(classes)}
    print(f"Classes ({len(classes)}): {classes}")

    # Build Vocabulary
    print("Building vocabulary...")
    vocab = Vocab()
    vocab.build_vocab(train_texts, max_vocab_size=20000, min_freq=2)
    print(f"Vocabulary size: {len(vocab.word2idx):,} tokens")
    vocab_path = os.path.join(MODELS_DIR, "bilstm_vocab.json")
    vocab.save(vocab_path)

    # DataLoaders
    train_dataset = TextDataset(train_texts, train_df["cyberbullying_type"].tolist(), vocab, label2idx)
    val_dataset = TextDataset(val_texts, val_df["cyberbullying_type"].tolist(), vocab, label2idx)
    test_dataset = TextDataset(test_texts, test_df["cyberbullying_type"].tolist(), vocab, label2idx)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Initialize BiLSTM
    model = CyberbullyingBiLSTM(
        vocab_size=len(vocab.word2idx),
        embedding_dim=128,
        hidden_dim=128,
        num_layers=2,
        num_classes=len(classes),
        dropout=0.3
    ).to(device)

    # Loss function with optional class weighting
    class_counts = train_df["cyberbullying_type"].value_counts()[classes].values
    weights = torch.tensor(1.0 / class_counts, dtype=torch.float).to(device)
    weights = weights / weights.sum() * len(classes)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    # Training Loop
    epochs = 4
    best_val_macro_f1 = 0.0
    model_save_path = os.path.join(MODELS_DIR, "bilstm_model.pt")

    print("\nStarting training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        start_t = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_macro_f1, val_weighted_f1, val_acc, _, _, _ = evaluate(model, val_loader, device, classes)
        scheduler.step(val_macro_f1)
        epoch_time = time.time() - start_t

        print(f"Epoch {epoch}/{epochs} [{epoch_time:.1f}s] - Loss: {avg_loss:.4f} | Val Macro-F1: {val_macro_f1:.4f} | Val Acc: {val_acc*100:.2f}%")

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "label2idx": label2idx,
                "vocab_size": len(vocab.word2idx),
                "embedding_dim": 128,
                "hidden_dim": 128,
                "num_layers": 2
            }, model_save_path)
            print(f"  --> Best checkpoint saved! (Val Macro-F1: {val_macro_f1:.4f})")

    # Evaluate on Test Set using best checkpoint
    print("\nLoading best model checkpoint for Test Set Evaluation...")
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_macro_f1, test_weighted_f1, test_acc, cm, test_preds, test_labels = evaluate(
        model, test_loader, device, classes
    )

    print("-" * 60)
    print("BiLSTM TEST SET RESULTS:")
    print(f"  • Macro-F1 Score:    {test_macro_f1:.4f}")
    print(f"  • Weighted-F1 Score: {test_weighted_f1:.4f}")
    print(f"  • Accuracy:          {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("-" * 60)

    # Classification Report
    report_dict = classification_report(test_labels, test_preds, target_names=classes, output_dict=True)
    print("\nClassification Report:\n", classification_report(test_labels, test_preds, target_names=classes, digits=4))

    # Confusion Matrix
    cm_df = pd.DataFrame(cm, index=[f"Actual_{c}" for c in classes], columns=[f"Pred_{c}" for c in classes])
    print("\nConfusion Matrix:\n", cm_df)

    # Save Metrics & Confusion Matrix
    metrics_save_path = os.path.join(REPORTS_DIR, "bilstm_metrics.json")
    with open(metrics_save_path, "w") as f:
        json.dump({
            "model": "BiLSTM Deep Learning",
            "macro_f1": round(test_macro_f1, 4),
            "weighted_f1": round(test_weighted_f1, 4),
            "accuracy": round(test_acc, 4),
            "classes": classes,
            "confusion_matrix": cm.tolist(),
            "classification_report": report_dict
        }, f, indent=2)

    cm_df.to_csv(os.path.join(REPORTS_DIR, "bilstm_confusion_matrix.csv"))
    print(f"\nArtifacts saved to {model_save_path} and {metrics_save_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
