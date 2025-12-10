"""
Dataset loaders for NAS training

Supports both character-level and token-level (BPE) language modeling on Python code
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader

# Import tokenizer (optional, only needed for token-level)
try:
    from transformers import GPT2TokenizerFast
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from tokenizers import Tokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False


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


# =============================================================================
# Token-Level (BPE) Support
# =============================================================================

class CodeTokenVocab:
    """
    Token-level vocabulary using BPE tokenizer

    Supports:
    - Pre-trained tokenizers (e.g., "gpt2" - 50K vocab)
    - Custom trained BPE tokenizers (e.g., 4K-8K vocab)

    Much more efficient than character-level:
    - Vocab size: ~4K-50K tokens vs ~100-300 chars
    - Data efficiency: 10-100x better
    - Better compositional understanding
    """

    def __init__(self, tokenizer_name: str = "gpt2", tokenizer_path: Optional[str] = None):
        """
        Args:
            tokenizer_name: Name of pre-trained tokenizer (e.g., "gpt2")
            tokenizer_path: Path to custom tokenizer.json file (takes precedence)
        """
        from pathlib import Path

        # If custom tokenizer path provided, load it
        if tokenizer_path is not None:
            if not TOKENIZERS_AVAILABLE:
                raise ImportError(
                    "tokenizers library not found. Install with: pip install tokenizers"
                )

            tokenizer_path = Path(tokenizer_path)
            if not tokenizer_path.exists():
                raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

            print(f"[VOCAB] Loading custom BPE tokenizer from {tokenizer_path}...")
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
            self.vocab_size = self.tokenizer.get_vocab_size()
            self.is_custom = True

            # Get special token IDs
            vocab = self.tokenizer.get_vocab()
            self.pad_token_id = vocab.get("<|pad|>", vocab.get("<|endoftext|>", 0))
            self.eos_token_id = vocab.get("<|endoftext|>", 0)

            print(f"[VOCAB] Custom tokenizer loaded")
            print(f"[VOCAB] Vocabulary size: {self.vocab_size:,} tokens")
            print(f"[VOCAB] Special tokens: PAD={self.pad_token_id}, EOS={self.eos_token_id}")

        else:
            # Load pre-trained GPT-2 tokenizer
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "transformers library not found. Install with: pip install transformers"
                )

            self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)

            # Set padding token (GPT-2 doesn't have one by default)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.vocab_size = len(self.tokenizer)
            self.is_custom = False
            self.pad_token_id = self.tokenizer.pad_token_id
            self.eos_token_id = self.tokenizer.eos_token_id

            print(f"[VOCAB] Loaded {tokenizer_name} tokenizer")
            print(f"[VOCAB] Vocabulary size: {self.vocab_size:,} tokens")
            print(f"[VOCAB] Special tokens: PAD={self.tokenizer.pad_token_id}, "
                  f"EOS={self.tokenizer.eos_token_id}, BOS={self.tokenizer.bos_token_id}")

    def encode(self, s: str) -> list:
        """Encode string to list of token IDs"""
        if self.is_custom:
            # Custom tokenizer from tokenizers library
            return self.tokenizer.encode(s).ids
        else:
            # HuggingFace transformers tokenizer
            return self.tokenizer.encode(s, add_special_tokens=False)

    def decode(self, ids: list) -> str:
        """Decode list of token IDs to string"""
        if self.is_custom:
            # Custom tokenizer from tokenizers library
            return self.tokenizer.decode(ids, skip_special_tokens=False)
        else:
            # HuggingFace transformers tokenizer
            return self.tokenizer.decode(ids, skip_special_tokens=False)


@dataclass
class CodeTokenDatasetConfig:
    """Configuration for token-level code dataset"""
    train_path: str
    val_path: str
    seq_len: int = 256
    batch_size: int = 32
    tokenizer_name: str = "gpt2"
    tokenizer_path: Optional[str] = None  # Path to custom tokenizer.json


class CodeTokenDataset(Dataset):
    """
    Token-level code dataset using BPE tokenization

    Returns consecutive sequences of tokens for language modeling
    """

    def __init__(self, path: str, seq_len: int, vocab: CodeTokenVocab):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(path, 'r', encoding='latin-1') as f:
                text = f.read()

        ids = vocab.encode(text)
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len

        print(f"[DATASET] Loaded {path}: {len(text)} chars, {len(ids)} tokens")
        print(f"[DATASET] Compression ratio: {len(text)/len(ids):.2f}x (chars/tokens)")

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


def build_code_token_loaders(
    cfg: CodeTokenDatasetConfig
) -> Tuple[DataLoader, DataLoader, CodeTokenVocab]:
    """
    Build train and validation data loaders for token-level modeling

    Args:
        cfg: Dataset configuration

    Returns:
        (train_loader, val_loader, vocab)
    """
    print(f"\n[DATA] Building TOKEN-LEVEL data loaders...")
    print(f"  Train: {cfg.train_path}")
    print(f"  Val: {cfg.val_path}")
    print(f"  Seq len: {cfg.seq_len}")
    print(f"  Batch size: {cfg.batch_size}")
    if cfg.tokenizer_path:
        print(f"  Tokenizer: custom ({cfg.tokenizer_path})")
    else:
        print(f"  Tokenizer: {cfg.tokenizer_name}")

    # Build vocabulary (load tokenizer)
    vocab = CodeTokenVocab(tokenizer_name=cfg.tokenizer_name, tokenizer_path=cfg.tokenizer_path)

    # Create datasets
    train_ds = CodeTokenDataset(cfg.train_path, cfg.seq_len, vocab)
    val_ds = CodeTokenDataset(cfg.val_path, cfg.seq_len, vocab)

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


# =============================================================================
# Character-Level Support (Original)
# =============================================================================

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
