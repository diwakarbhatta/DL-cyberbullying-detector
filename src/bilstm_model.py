"""
BiLSTM Deep Learning Architecture for Cyberbullying Detection
PyTorch implementation with embedding layer, bidirectional LSTM, and dense classification head.
"""

import torch
import torch.nn as nn
import json
import re
from typing import List, Dict, Tuple, Optional

class Vocab:
    def __init__(self, pad_token="<pad>", unk_token="<unk>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2idx = {pad_token: 0, unk_token: 1}
        self.idx2word = {0: pad_token, 1: unk_token}
        self.freqs = {}

    def build_vocab(self, texts: List[str], max_vocab_size: int = 25000, min_freq: int = 2):
        for text in texts:
            for token in self.tokenize(text):
                self.freqs[token] = self.freqs.get(token, 0) + 1

        sorted_tokens = sorted(self.freqs.items(), key=lambda x: x[1], reverse=True)
        for token, freq in sorted_tokens:
            if freq < min_freq or len(self.word2idx) >= max_vocab_size:
                break
            if token not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[token] = idx
                self.idx2word[idx] = token

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def encode(self, text: str, max_length: int = 100) -> List[int]:
        tokens = self.tokenize(text)
        indices = [self.word2idx.get(t, self.word2idx[self.unk_token]) for t in tokens[:max_length]]
        if len(indices) < max_length:
            indices += [self.word2idx[self.pad_token]] * (max_length - len(indices))
        return indices

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "word2idx": self.word2idx,
                "pad_token": self.pad_token,
                "unk_token": self.unk_token
            }, f)

    @classmethod
    def load(cls, filepath: str):
        vocab = cls()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {v: k for k, v in vocab.word2idx.items()}
        vocab.pad_token = data.get("pad_token", "<pad>")
        vocab.unk_token = data.get("unk_token", "<unk>")
        return vocab

class CyberbullyingBiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 6,
        dropout: float = 0.3
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len]
        embeds = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        lstm_out, (hn, cn) = self.lstm(embeds)  # [batch_size, seq_len, hidden_dim * 2]

        # Global average pooling across valid non-pad tokens
        mask = (x != 0).unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        sum_pooled = torch.sum(lstm_out * mask, dim=1)  # [batch_size, hidden_dim * 2]
        lengths = torch.clamp(mask.sum(dim=1), min=1.0)
        pooled = sum_pooled / lengths

        h = self.dropout(self.relu(self.fc1(pooled)))
        logits = self.fc_out(h)
        return logits
