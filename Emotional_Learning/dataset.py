"""数据加载、分词与词表构建。"""

import json
import pickle
from pathlib import Path

import jieba
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1


class Vocab:
    def __init__(self):
        self.token2id = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        self.id2token = {PAD_ID: PAD_TOKEN, UNK_ID: UNK_TOKEN}

    def __len__(self):
        return len(self.token2id)

    def build(self, texts: list[str], min_freq: int = 2, max_size: int | None = 50000):
        freq: dict[str, int] = {}
        for text in texts:
            for token in tokenize(text):
                freq[token] = freq.get(token, 0) + 1

        sorted_tokens = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        for token, count in sorted_tokens:
            if count < min_freq:
                continue
            if max_size is not None and len(self.token2id) >= max_size:
                break
            if token not in self.token2id:
                idx = len(self.token2id)
                self.token2id[token] = idx
                self.id2token[idx] = token

    def encode(self, text: str, max_len: int) -> list[int]:
        tokens = tokenize(text)[:max_len]
        return [self.token2id.get(token, UNK_ID) for token in tokens]

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"token2id": self.token2id, "id2token": self.id2token}, f)

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        vocab = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        vocab.token2id = data["token2id"]
        vocab.id2token = data["id2token"]
        return vocab


def tokenize(text: str) -> list[str]:
    text = str(text).strip()
    if not text:
        return []
    return [token for token in jieba.lcut(text) if token.strip()]


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[["review", "sentiment"]].copy()
    df["review"] = df["review"].astype(str).str.strip()
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    df = df.dropna(subset=["review", "sentiment"])
    df = df[df["review"].str.len() > 0]
    df["sentiment"] = df["sentiment"].astype(int)
    return df.reset_index(drop=True)


class ReviewDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: Vocab, max_len: int):
        self.reviews = df["review"].tolist()
        self.labels = df["sentiment"].tolist()
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        ids = self.vocab.encode(self.reviews[idx], self.max_len)
        label = self.labels[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def collate_batch(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=PAD_ID)
    labels = torch.stack(labels)
    return padded, lengths, labels


def save_data_config(path: Path, config: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
