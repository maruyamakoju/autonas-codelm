#!/usr/bin/env python3
"""
train_bpe_tokenizer.py

Train a custom BPE tokenizer with specified vocabulary size from Python corpus.

Usage:
    # Train 8K vocab tokenizer from 100MB corpus
    python scripts/train_bpe_tokenizer.py \
        --input ../data/code_token_bigdata/train.txt \
        --vocab_size 8000 \
        --output_dir ../data/tokenizers/python_bpe_8k

    # Train 4K vocab tokenizer
    python scripts/train_bpe_tokenizer.py \
        --input ../data/code_token_bigdata/train.txt \
        --vocab_size 4096 \
        --output_dir ../data/tokenizers/python_bpe_4k
"""

import argparse
import json
from pathlib import Path
from typing import Optional

try:
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    print("ERROR: 'tokenizers' package not installed")
    print("Install with: pip install tokenizers")


def train_bpe_tokenizer(
    input_path: str,
    vocab_size: int,
    output_dir: str,
    min_frequency: int = 2,
    special_tokens: Optional[list] = None
) -> None:
    """
    Train a BPE tokenizer from scratch.

    Args:
        input_path: Path to training corpus (plain text)
        vocab_size: Target vocabulary size
        output_dir: Directory to save tokenizer
        min_frequency: Minimum frequency for tokens
        special_tokens: List of special tokens to add
    """
    if not TOKENIZERS_AVAILABLE:
        raise ImportError("tokenizers package is required")

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Training Custom BPE Tokenizer")
    print("=" * 70)
    print(f"Input corpus: {input_path}")
    print(f"Vocab size: {vocab_size:,}")
    print(f"Output directory: {output_dir}")
    print(f"Min frequency: {min_frequency}")
    print("=" * 70)
    print()

    # Default special tokens
    if special_tokens is None:
        special_tokens = [
            "<|endoftext|>",  # EOS/BOS/PAD token (compatible with GPT-2 style)
            "<|pad|>",
        ]

    print(f"[INIT] Creating BPE tokenizer...")
    print(f"[INIT] Special tokens: {special_tokens}")

    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<|endoftext|>"))

    # Set pre-tokenizer (split on whitespace and punctuation, similar to GPT-2)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Set decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Set post-processor (add special tokens)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Create trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # Check input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Get file size
    file_size_mb = input_path.stat().st_size / (1024 * 1024)
    print(f"\n[CORPUS] Reading {file_size_mb:.2f} MB from {input_path.name}...")

    # Train tokenizer
    print(f"\n[TRAIN] Training BPE tokenizer (this may take a few minutes)...")
    tokenizer.train(files=[str(input_path)], trainer=trainer)

    # Get actual vocab size
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"\n[TRAIN] Training complete!")
    print(f"[TRAIN] Actual vocab size: {actual_vocab_size:,}")

    # Save tokenizer
    tokenizer_path = output_dir / "tokenizer.json"
    print(f"\n[SAVE] Saving tokenizer to {tokenizer_path}...")
    tokenizer.save(str(tokenizer_path))

    # Save config metadata
    config = {
        "type": "bpe",
        "vocab_size": actual_vocab_size,
        "target_vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "special_tokens": special_tokens,
        "trained_on": str(input_path.absolute()),
        "corpus_size_mb": round(file_size_mb, 2),
    }

    config_path = output_dir / "config.json"
    print(f"[SAVE] Saving config to {config_path}...")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Test tokenizer on sample text
    print(f"\n[TEST] Testing tokenizer on sample code...")
    test_samples = [
        "def add(a, b):",
        "class Stack:",
        "import numpy as np",
        "for i in range(10):",
    ]

    for sample in test_samples:
        encoded = tokenizer.encode(sample)
        tokens = encoded.tokens
        ids = encoded.ids
        decoded = tokenizer.decode(ids)

        print(f"\n  Input:   {sample}")
        print(f"  Tokens:  {tokens}")
        print(f"  IDs:     {ids}")
        print(f"  Decoded: {decoded}")
        print(f"  #Tokens: {len(tokens)}")

    print()
    print("=" * 70)
    print(" Tokenizer Training Complete")
    print("=" * 70)
    print(f"Saved to: {output_dir}")
    print()
    print("Usage:")
    print(f"  from tokenizers import Tokenizer")
    print(f"  tokenizer = Tokenizer.from_file('{tokenizer_path}')")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train a custom BPE tokenizer for Python code"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input corpus (plain text file)"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=True,
        help="Target vocabulary size (e.g., 4096, 8000)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save trained tokenizer"
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Minimum frequency for tokens (default: 2)"
    )

    args = parser.parse_args()

    train_bpe_tokenizer(
        input_path=args.input,
        vocab_size=args.vocab_size,
        output_dir=args.output_dir,
        min_frequency=args.min_frequency
    )


if __name__ == "__main__":
    main()
