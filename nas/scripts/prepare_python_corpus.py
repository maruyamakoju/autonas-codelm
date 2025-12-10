#!/usr/bin/env python3
"""
prepare_python_corpus.py

Prepare large-scale Python corpus for BOTH char-level and token-level language modeling.

Usage:
    # Generate both char-level and token-level corpora (recommended)
    python scripts/prepare_python_corpus.py \
        --src_dir ../data/raw_python \
        --char_train ../data/code_char_big/train.txt \
        --char_val   ../data/code_char_big/val.txt \
        --token_train ../data/code_token_big/train.txt \
        --token_val   ../data/code_token_big/val.txt \
        --mode both \
        --val_ratio 0.01

    # Char-level only
    python scripts/prepare_python_corpus.py \
        --src_dir ../data/raw_python \
        --char_train ../data/code_char_big/train.txt \
        --char_val   ../data/code_char_big/val.txt \
        --mode char

    # Token-level only (GPT-2 tokenizer)
    python scripts/prepare_python_corpus.py \
        --src_dir ../data/raw_python \
        --token_train ../data/code_token_big/train.txt \
        --token_val   ../data/code_token_big/val.txt \
        --mode token

    # Large-scale (100MB-1GB) with size limit
    python scripts/prepare_python_corpus.py \
        --src_dir ../data/the_stack_python \
        --char_train ../data/code_char_bigdata/train.txt \
        --char_val   ../data/code_char_bigdata/val.txt \
        --token_train ../data/code_token_bigdata/train.txt \
        --token_val   ../data/code_token_bigdata/val.txt \
        --mode both \
        --max_file_size 524288 \
        --target_size_mb 500

Features:
    - Char-level & Token-level (GPT-2 BPE) dual support
    - Recursively walks source directory for .py files
    - Filters by file size (skip too large/small files)
    - Target size limit (stops when reaching target MB)
    - Handles encoding errors gracefully
    - Normalizes line endings to \\n
    - Splits into train/val with shuffling
    - Reports detailed statistics:
      - Char-level: files, lines, bytes, MB
      - Token-level: tokens, compression ratio
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple, Optional
import sys

# Token-level support (optional)
try:
    from transformers import GPT2TokenizerFast
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False


def is_valid_python_file(path: Path, min_size: int, max_size: int) -> bool:
    """Check if file is a valid Python source file."""
    # Check extension
    if path.suffix != '.py':
        return False

    # Check if it's a file (not directory)
    if not path.is_file():
        return False

    # Check size
    try:
        size = path.stat().st_size
        if size < min_size or size > max_size:
            return False
    except Exception:
        return False

    return True


def read_python_file(path: Path) -> Optional[str]:
    """Read Python file with encoding fallback."""
    # Try UTF-8 first
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except UnicodeDecodeError:
        pass

    # Try latin-1 as fallback
    try:
        with open(path, 'r', encoding='latin-1') as f:
            content = f.read()
        return content
    except Exception:
        pass

    # Give up
    return None


def collect_python_files(
    src_dir: Path,
    min_size: int = 100,
    max_size: int = 262144,
    target_size_mb: Optional[float] = None,
    exclude_patterns: List[str] = None
) -> List[Tuple[Path, str]]:
    """
    Collect all valid Python files from source directory.

    Args:
        src_dir: Source directory to scan
        min_size: Minimum file size in bytes
        max_size: Maximum file size in bytes
        target_size_mb: Stop when total size reaches this (MB)
        exclude_patterns: Patterns to exclude from collection

    Returns:
        List of (path, content) tuples
    """
    if exclude_patterns is None:
        exclude_patterns = [
            '__pycache__',
            '.git',
            '.venv',
            'venv',
            'env',
            'site-packages',
            'node_modules',
            '.eggs',
            'build',
            'dist',
            '.pytest_cache',
            '.mypy_cache',
            '.tox',
        ]

    print(f"[COLLECT] Scanning {src_dir}...")
    print(f"[COLLECT] File size range: {min_size}-{max_size} bytes")
    if target_size_mb:
        print(f"[COLLECT] Target size: {target_size_mb:.2f} MB")
    print(f"[COLLECT] Exclude patterns: {exclude_patterns}")
    print()

    files = []
    total_bytes = 0
    target_bytes = target_size_mb * 1024 * 1024 if target_size_mb else None

    skipped_size = 0
    skipped_encoding = 0
    skipped_exclude = 0

    # Walk directory tree
    for path in src_dir.rglob('*.py'):
        # Check target size limit
        if target_bytes and total_bytes >= target_bytes:
            print(f"\n[COLLECT] Reached target size ({target_size_mb:.2f} MB), stopping collection")
            break

        # Check exclude patterns
        skip = False
        for pattern in exclude_patterns:
            if pattern in str(path):
                skip = True
                skipped_exclude += 1
                break
        if skip:
            continue

        # Check validity
        if not is_valid_python_file(path, min_size, max_size):
            skipped_size += 1
            continue

        # Read content
        content = read_python_file(path)
        if content is None:
            skipped_encoding += 1
            continue

        # Normalize line endings
        content = content.replace('\r\n', '\n').replace('\r', '\n')

        files.append((path, content))
        total_bytes += len(content.encode('utf-8'))

        if len(files) % 100 == 0:
            mb = total_bytes / (1024 * 1024)
            print(f"[COLLECT] Collected {len(files)} files ({mb:.2f} MB)...", end='\r')

    print()
    print(f"[COLLECT] OK Collected {len(files)} valid Python files")
    print(f"[COLLECT]   Total size: {total_bytes / (1024 * 1024):.2f} MB")
    print(f"[COLLECT]   Skipped (size): {skipped_size}")
    print(f"[COLLECT]   Skipped (encoding): {skipped_encoding}")
    print(f"[COLLECT]   Skipped (exclude): {skipped_exclude}")
    print()

    return files


def split_train_val(
    files: List[Tuple[Path, str]],
    val_ratio: float = 0.01,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Split files into train and val sets.

    Returns:
        (train_contents, val_contents) where each is a list of file contents
    """
    random.seed(seed)
    random.shuffle(files)

    n_val = max(1, int(len(files) * val_ratio))
    n_train = len(files) - n_val

    train_files = files[:n_train]
    val_files = files[n_train:]

    train_contents = [content for path, content in train_files]
    val_contents = [content for path, content in val_files]

    print(f"[SPLIT] Train: {n_train} files")
    print(f"[SPLIT] Val:   {n_val} files")
    print()

    return train_contents, val_contents


def write_corpus(contents: List[str], output_path: Path) -> dict:
    """
    Write corpus to file.

    Returns:
        Statistics dict with 'files', 'lines', 'bytes', 'mb'
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    total_bytes = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for content in contents:
            f.write(content)
            # Ensure file ends with newline
            if not content.endswith('\n'):
                f.write('\n')

            total_lines += content.count('\n')
            total_bytes += len(content.encode('utf-8'))

    stats = {
        'files': len(contents),
        'lines': total_lines,
        'bytes': total_bytes,
        'mb': total_bytes / (1024 * 1024),
    }

    return stats


def compute_token_stats(contents: List[str], tokenizer_name: str = "gpt2") -> dict:
    """
    Compute token-level statistics using GPT-2 tokenizer.

    Returns:
        Statistics dict with 'tokens', 'chars', 'compression_ratio'
    """
    if not TOKENIZER_AVAILABLE:
        print("[WARN] transformers library not available, skipping token stats")
        return None

    print(f"[TOKEN] Loading {tokenizer_name} tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)

    print(f"[TOKEN] Tokenizing {len(contents)} files...")
    total_tokens = 0
    total_chars = 0

    for i, content in enumerate(contents):
        tokens = tokenizer.encode(content, add_special_tokens=False)
        total_tokens += len(tokens)
        total_chars += len(content)

        if (i + 1) % 100 == 0:
            print(f"[TOKEN] Processed {i+1}/{len(contents)} files...", end='\r')

    print()

    compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0

    stats = {
        'tokens': total_tokens,
        'chars': total_chars,
        'compression_ratio': compression_ratio,
        'vocab_size': len(tokenizer),
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare large-scale Python corpus for char-level and/or token-level LM"
    )
    parser.add_argument(
        '--src_dir',
        type=str,
        required=True,
        help='Source directory containing Python files'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['char', 'token', 'both'],
        default='both',
        help='Output mode: char-level, token-level, or both (default: both)'
    )

    # Char-level outputs
    parser.add_argument(
        '--char_train',
        type=str,
        help='Output path for char-level training corpus'
    )
    parser.add_argument(
        '--char_val',
        type=str,
        help='Output path for char-level validation corpus'
    )

    # Token-level outputs
    parser.add_argument(
        '--token_train',
        type=str,
        help='Output path for token-level training corpus'
    )
    parser.add_argument(
        '--token_val',
        type=str,
        help='Output path for token-level validation corpus'
    )

    # Common parameters
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.01,
        help='Validation set ratio (default: 0.01 = 1%%)'
    )
    parser.add_argument(
        '--min_file_size',
        type=int,
        default=100,
        help='Minimum file size in bytes (default: 100)'
    )
    parser.add_argument(
        '--max_file_size',
        type=int,
        default=262144,
        help='Maximum file size in bytes (default: 262144 = 256KB)'
    )
    parser.add_argument(
        '--target_size_mb',
        type=float,
        default=None,
        help='Stop collection when reaching target size in MB (default: no limit)'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='gpt2',
        help='Tokenizer to use for token-level stats (default: gpt2)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for train/val split (default: 42)'
    )

    args = parser.parse_args()

    # Validate outputs
    if args.mode in ['char', 'both']:
        if not args.char_train or not args.char_val:
            print("Error: --char_train and --char_val required for char-level output")
            sys.exit(1)

    if args.mode in ['token', 'both']:
        if not args.token_train or not args.token_val:
            print("Error: --token_train and --token_val required for token-level output")
            sys.exit(1)

    if args.mode == 'token' and not TOKENIZER_AVAILABLE:
        print("Error: transformers library not found. Install with: pip install transformers")
        sys.exit(1)

    src_dir = Path(args.src_dir)

    # Validate paths
    if not src_dir.exists():
        print(f"Error: Source directory not found: {src_dir}")
        sys.exit(1)

    print("=" * 70)
    print(" Python Corpus Preparation (Char + Token)")
    print("=" * 70)
    print(f"Source dir:  {src_dir}")
    print(f"Mode:        {args.mode}")
    if args.mode in ['char', 'both']:
        print(f"Char train:  {args.char_train}")
        print(f"Char val:    {args.char_val}")
    if args.mode in ['token', 'both']:
        print(f"Token train: {args.token_train}")
        print(f"Token val:   {args.token_val}")
    print(f"Val ratio:   {args.val_ratio:.2%}")
    print(f"File size:   {args.min_file_size}-{args.max_file_size} bytes")
    if args.target_size_mb:
        print(f"Target size: {args.target_size_mb:.2f} MB")
    print("=" * 70)
    print()

    # Step 1: Collect files
    files = collect_python_files(
        src_dir,
        min_size=args.min_file_size,
        max_size=args.max_file_size,
        target_size_mb=args.target_size_mb
    )

    if not files:
        print("Error: No valid Python files found")
        sys.exit(1)

    # Step 2: Split train/val
    train_contents, val_contents = split_train_val(
        files,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    # Step 3: Write char-level corpus
    char_train_stats = None
    char_val_stats = None

    if args.mode in ['char', 'both']:
        print("=" * 70)
        print(" CHAR-LEVEL OUTPUT")
        print("=" * 70)

        char_train_path = Path(args.char_train)
        char_val_path = Path(args.char_val)

        print(f"[WRITE] Writing char train corpus to {char_train_path}...")
        char_train_stats = write_corpus(train_contents, char_train_path)
        print(f"[WRITE] OK {char_train_stats['files']} files, "
              f"{char_train_stats['lines']:,} lines, {char_train_stats['mb']:.2f} MB")
        print()

        print(f"[WRITE] Writing char val corpus to {char_val_path}...")
        char_val_stats = write_corpus(val_contents, char_val_path)
        print(f"[WRITE] OK {char_val_stats['files']} files, "
              f"{char_val_stats['lines']:,} lines, {char_val_stats['mb']:.2f} MB")
        print()

    # Step 4: Write token-level corpus (same text, different directory)
    token_train_stats = None
    token_val_stats = None
    train_token_stats = None
    val_token_stats = None

    if args.mode in ['token', 'both']:
        print("=" * 70)
        print(" TOKEN-LEVEL OUTPUT")
        print("=" * 70)

        token_train_path = Path(args.token_train)
        token_val_path = Path(args.token_val)

        # Write same text files (tokenization happens in Dataset)
        print(f"[WRITE] Writing token train corpus to {token_train_path}...")
        token_train_stats = write_corpus(train_contents, token_train_path)
        print(f"[WRITE] OK {token_train_stats['files']} files, "
              f"{token_train_stats['lines']:,} lines, {token_train_stats['mb']:.2f} MB")
        print()

        print(f"[WRITE] Writing token val corpus to {token_val_path}...")
        token_val_stats = write_corpus(val_contents, token_val_path)
        print(f"[WRITE] OK {token_val_stats['files']} files, "
              f"{token_val_stats['lines']:,} lines, {token_val_stats['mb']:.2f} MB")
        print()

        # Compute token statistics
        if TOKENIZER_AVAILABLE:
            print(f"[TOKEN] Computing token statistics with {args.tokenizer} tokenizer...")
            train_token_stats = compute_token_stats(train_contents, args.tokenizer)
            val_token_stats = compute_token_stats(val_contents, args.tokenizer)

            if train_token_stats:
                print(f"[TOKEN] Train: {train_token_stats['tokens']:,} tokens, "
                      f"compression {train_token_stats['compression_ratio']:.2f}x")
                print(f"[TOKEN] Val:   {val_token_stats['tokens']:,} tokens, "
                      f"compression {val_token_stats['compression_ratio']:.2f}x")
                print()

    # Summary
    print("=" * 70)
    print(" SUMMARY")
    print("=" * 70)

    total_files = len(files)
    print(f"Total files collected: {total_files:,}")
    print()

    if char_train_stats:
        total_char_mb = char_train_stats['mb'] + char_val_stats['mb']
        total_char_lines = char_train_stats['lines'] + char_val_stats['lines']

        print("CHAR-LEVEL:")
        print(f"  Total:  {total_char_mb:.2f} MB, {total_char_lines:,} lines")
        print(f"  Train:  {char_train_stats['mb']:.2f} MB ({char_train_stats['files']} files)")
        print(f"  Val:    {char_val_stats['mb']:.2f} MB ({char_val_stats['files']} files)")
        print()

    if token_train_stats and train_token_stats:
        total_token_mb = token_train_stats['mb'] + token_val_stats['mb']
        total_tokens = train_token_stats['tokens'] + val_token_stats['tokens']
        avg_compression = (train_token_stats['compression_ratio'] * train_token_stats['chars'] +
                          val_token_stats['compression_ratio'] * val_token_stats['chars']) / \
                         (train_token_stats['chars'] + val_token_stats['chars'])

        print("TOKEN-LEVEL:")
        print(f"  Total:       {total_token_mb:.2f} MB, {total_tokens:,} tokens")
        print(f"  Train:       {token_train_stats['mb']:.2f} MB ({token_train_stats['files']} files)")
        print(f"               {train_token_stats['tokens']:,} tokens")
        print(f"  Val:         {token_val_stats['mb']:.2f} MB ({token_val_stats['files']} files)")
        print(f"               {val_token_stats['tokens']:,} tokens")
        print(f"  Compression: {avg_compression:.2f}x (chars/tokens)")
        print(f"  Vocab size:  {train_token_stats['vocab_size']:,} ({args.tokenizer})")
        print()

    print("=" * 70)
    print("OK Corpus preparation complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
