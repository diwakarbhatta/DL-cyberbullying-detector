"""
Fine-tune DistilBERT on the Cyberbullying Classification dataset (6 classes).
Evaluates with Macro-F1 and saves confusion matrix + metrics.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessor import clean_text, normalize_leetspeak

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

def evaluate(model, loader, device, classes):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    acc = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(loader)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))

    return macro_f1, weighted_f1, acc, avg_loss, cm, all_preds, all_labels

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"FINE-TUNING DistilBERT (Device: {device})")
    print("=" * 60)

    # Load data
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    print(f"Loaded {len(train_df):,} train, {len(val_df):,} val, {len(test_df):,} test samples.")

    # Preprocess
    print("Preprocessing text...")
    train_texts = [normalize_leetspeak(clean_text(str(t))) for t in train_df["tweet_text"]]
    val_texts = [normalize_leetspeak(clean_text(str(t))) for t in val_df["tweet_text"]]
    test_texts = [normalize_leetspeak(clean_text(str(t))) for t in test_df["tweet_text"]]

    classes = sorted(train_df["cyberbullying_type"].unique().tolist())
    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {i: c for c, i in label2idx.items()}
    print(f"Classes ({len(classes)}): {classes}")

    train_labels = [label2idx[l] for l in train_df["cyberbullying_type"]]
    val_labels = [label2idx[l] for l in val_df["cyberbullying_type"]]
    test_labels = [label2idx[l] for l in test_df["cyberbullying_type"]]

    # Tokenizer
    print("Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    print("Tokenizing datasets...")
    train_dataset = TweetDataset(train_texts, train_labels, tokenizer, max_length=128)
    val_dataset = TweetDataset(val_texts, val_labels, tokenizer, max_length=128)
    test_dataset = TweetDataset(test_texts, test_labels, tokenizer, max_length=128)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Model
    print("Loading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(classes),
        id2label=idx2label,
        label2id=label2idx
    ).to(device)

    # Class weights for imbalanced data
    class_counts = train_df["cyberbullying_type"].value_counts()[classes].values
    weights = torch.tensor(1.0 / class_counts, dtype=torch.float).to(device)
    weights = weights / weights.sum() * len(classes)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 3
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Loss with class weights
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    best_val_f1 = 0.0
    save_path = os.path.join(MODELS_DIR, "distilbert_cyberbullying")

    print(f"\nTraining for {epochs} epochs ({total_steps} total steps)...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        start_t = time.time()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if (step + 1) % 200 == 0:
                print(f"  Step {step+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        val_macro_f1, val_weighted_f1, val_acc, val_loss, _, _, _ = evaluate(
            model, val_loader, device, classes
        )
        elapsed = time.time() - start_t

        print(f"Epoch {epoch}/{epochs} [{elapsed:.0f}s] - Train Loss: {avg_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Macro-F1: {val_macro_f1:.4f} | Val Acc: {val_acc*100:.1f}%")

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            # Also save label mapping
            with open(os.path.join(save_path, "label_mapping.json"), "w") as f:
                json.dump({"label2idx": label2idx, "idx2label": idx2label, "classes": classes}, f)
            print(f"  --> Best model saved! (Val Macro-F1: {val_macro_f1:.4f})")

    # Final evaluation on test set with best model
    print("\nLoading best model for test evaluation...")
    model = DistilBertForSequenceClassification.from_pretrained(save_path).to(device)

    test_macro_f1, test_weighted_f1, test_acc, test_loss, cm, test_preds, test_true = evaluate(
        model, test_loader, device, classes
    )

    print("-" * 60)
    print("DistilBERT TEST SET RESULTS:")
    print(f"  Macro-F1 Score:    {test_macro_f1:.4f}")
    print(f"  Weighted-F1 Score: {test_weighted_f1:.4f}")
    print(f"  Accuracy:          {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("-" * 60)

    report_dict = classification_report(test_true, test_preds, target_names=classes, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(test_true, test_preds, target_names=classes, digits=4))

    cm_df = pd.DataFrame(cm, index=[f"Actual_{c}" for c in classes], columns=[f"Pred_{c}" for c in classes])
    print("\nConfusion Matrix:")
    print(cm_df)

    # Save metrics
    with open(os.path.join(REPORTS_DIR, "distilbert_metrics.json"), "w") as f:
        json.dump({
            "model": "DistilBERT Fine-tuned",
            "macro_f1": round(test_macro_f1, 4),
            "weighted_f1": round(test_weighted_f1, 4),
            "accuracy": round(test_acc, 4),
            "classes": classes,
            "confusion_matrix": cm.tolist(),
            "classification_report": report_dict
        }, f, indent=2)

    cm_df.to_csv(os.path.join(REPORTS_DIR, "distilbert_confusion_matrix.csv"))
    print(f"\nAll artifacts saved to {save_path}/ and reports/")
    print("=" * 60)

if __name__ == "__main__":
    main()
