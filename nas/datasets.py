"""
Dataset loaders for NAS training

Character-level language modeling on Python code
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class CodeCharDatasetConfig:
    """Configuration for character-level code dataset"""
    train_path: str
    val_path: str
    seq_len: int = 256
    batch_size: int = 32


class CodeCharVocab:
    """
    Character-level vocabulary

    Builds vocabulary from training text only (no leakage)
    """

    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

        print(f"[VOCAB] Built vocabulary: {self.vocab_size} unique characters")
        print(f"[VOCAB] Sample chars: {chars[:20]}")

    def encode(self, s: str) -> list:
        """Encode string to list of integers"""
        return [self.stoi.get(c, 0) for c in s]

    def decode(self, ids: list) -> str:
        """Decode list of integers to string"""
        return ''.join([self.itos.get(i, '?') for i in ids])


class CodeCharDataset(Dataset):
    """
    Character-level code dataset

    Returns consecutive sequences of characters for language modeling
    """

    def __init__(self, path: str, seq_len: int, vocab: CodeCharVocab):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 if utf-8 fails
            with open(path, 'r', encoding='latin-1') as f:
                text = f.read()

        ids = vocab.encode(text)
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len

        print(f"[DATASET] Loaded {path}: {len(text)} chars, {len(ids)} tokens")

    def __len__(self) -> int:
        return max(0, len(self.ids) - self.seq_len - 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: input sequence (seq_len,)
            y: target sequence (seq_len,)
        """
        x = self.ids[idx:idx + self.seq_len]
        y = self.ids[idx + 1:idx + self.seq_len + 1]
        return x, y


def build_code_char_loaders(
    cfg: CodeCharDatasetConfig
) -> Tuple[DataLoader, DataLoader, CodeCharVocab]:
    """
    Build train and validation data loaders

    Args:
        cfg: Dataset configuration

    Returns:
        (train_loader, val_loader, vocab)
    """
    print(f"\n[DATA] Building data loaders...")
    print(f"  Train: {cfg.train_path}")
    print(f"  Val: {cfg.val_path}")
    print(f"  Seq len: {cfg.seq_len}")
    print(f"  Batch size: {cfg.batch_size}")

    # Build vocabulary from train set only
    try:
        with open(cfg.train_path, 'r', encoding='utf-8') as f:
            train_text = f.read()
    except UnicodeDecodeError:
        with open(cfg.train_path, 'r', encoding='latin-1') as f:
            train_text = f.read()

    vocab = CodeCharVocab(train_text)

    # Create datasets
    train_ds = CodeCharDataset(cfg.train_path, cfg.seq_len, vocab)
    val_ds = CodeCharDataset(cfg.val_path, cfg.seq_len, vocab)

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"[DATA] Train batches: {len(train_loader)}")
    print(f"[DATA] Val batches: {len(val_loader)}")

    return train_loader, val_loader, vocab


if __name__ == "__main__":
    # Test
    print("="*60)
    print("Dataset Test")
    print("="*60)

    cfg = CodeCharDatasetConfig(
        train_path="../data/code_char/train.txt",
        val_path="../data/code_char/val.txt",
        seq_len=128,
        batch_size=4
    )

    train_loader, val_loader, vocab = build_code_char_loaders(cfg)

    # Test one batch
    x, y = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  x: {x.shape}")
    print(f"  y: {y.shape}")

    # Decode first sequence
    sample_text = vocab.decode(x[0].tolist())
    print(f"\nSample input (first 100 chars):")
    print(f"  {sample_text[:100]}")

    print("\n" + "="*60)
    print("Dataset test passed!")
    print("="*60)
