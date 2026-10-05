"""
Evaluate saved BiLSTM checkpoint on test set.
Loads the best checkpoint and writes metrics + confusion matrix.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.bilstm_model import Vocab, CyberbullyingBiLSTM
from src.preprocessor import clean_text, normalize_leetspeak

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, label2idx, max_length=100):
        self.encoded = [vocab.encode(t, max_length=max_length) for t in texts]
        self.labels = [label2idx[l] for l in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encoded[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("EVALUATING SAVED BiLSTM CHECKPOINT ON TEST SET")
    print("=" * 60)

    # Load checkpoint
    model_path = os.path.join(MODELS_DIR, "bilstm_model.pt")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    label2idx = checkpoint["label2idx"]
    idx2label = {v: k for k, v in label2idx.items()}

    print(f"Classes: {classes}")

    # Load vocab
    vocab = Vocab.load(os.path.join(MODELS_DIR, "bilstm_vocab.json"))
    print(f"Vocab size: {len(vocab.word2idx):,}")

    # Rebuild model
    model = CyberbullyingBiLSTM(
        vocab_size=checkpoint.get("vocab_size", len(vocab.word2idx)),
        embedding_dim=checkpoint.get("embedding_dim", 128),
        hidden_dim=checkpoint.get("hidden_dim", 128),
        num_layers=checkpoint.get("num_layers", 2),
        num_classes=len(classes),
        dropout=0.0  # no dropout at eval
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test data
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    test_texts = [normalize_leetspeak(clean_text(t)) for t in test_df["tweet_text"].astype(str)]
    test_dataset = TextDataset(test_texts, test_df["cyberbullying_type"].tolist(), vocab, label2idx)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))

    print("-" * 60)
    print("BiLSTM TEST SET RESULTS:")
    print(f"  Macro-F1 Score:    {macro_f1:.4f}")
    print(f"  Weighted-F1 Score: {weighted_f1:.4f}")
    print(f"  Accuracy:          {acc:.4f} ({acc*100:.2f}%)")
    print("-" * 60)

    report_dict = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

    cm_df = pd.DataFrame(cm, index=[f"Actual_{c}" for c in classes], columns=[f"Pred_{c}" for c in classes])
    print("\nConfusion Matrix:")
    print(cm_df)

    # Save
    with open(os.path.join(REPORTS_DIR, "bilstm_metrics.json"), "w") as f:
        json.dump({
            "model": "BiLSTM Deep Learning",
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "accuracy": round(acc, 4),
            "classes": classes,
            "confusion_matrix": cm.tolist(),
            "classification_report": report_dict
        }, f, indent=2)

    cm_df.to_csv(os.path.join(REPORTS_DIR, "bilstm_confusion_matrix.csv"))
    print(f"\nMetrics saved to reports/bilstm_metrics.json")
    print("=" * 60)

if __name__ == "__main__":
    main()
