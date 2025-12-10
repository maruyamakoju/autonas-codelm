"""
Analyze token frequency distribution in 8K BPE tokenized data

Goal: Check if the collapse tokens (like `):`  or `:`) are abnormally frequent
in the training data itself.
"""

import sys
from pathlib import Path
from collections import Counter
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import CodeTokenVocab


def analyze_token_distribution(train_path: str, tokenizer_path: str, top_k: int = 50):
    """
    Analyze token frequency distribution

    Args:
        train_path: Path to training data
        tokenizer_path: Path to custom BPE tokenizer
        top_k: Number of top tokens to display
    """
    print("="*70)
    print("TOKEN FREQUENCY ANALYSIS (8K BPE)")
    print("="*70)

    # Load tokenizer
    print(f"\n[1] Loading tokenizer from: {tokenizer_path}")
    vocab = CodeTokenVocab(tokenizer_path=tokenizer_path)
    print(f"    Vocab size: {vocab.vocab_size:,}")

    # Load and tokenize training data
    print(f"\n[2] Loading training data from: {train_path}")
    with open(train_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"    Text length: {len(text):,} chars")

    print(f"\n[3] Tokenizing...")
    tokens = vocab.encode(text)
    print(f"    Total tokens: {len(tokens):,}")
    print(f"    Compression ratio: {len(text)/len(tokens):.2f}x")

    # Count token frequencies
    print(f"\n[4] Counting token frequencies...")
    counter = Counter(tokens)

    # Get top tokens
    top_tokens = counter.most_common(top_k)

    print(f"\n[5] TOP {top_k} TOKENS")
    print("-"*70)
    print(f"{'Rank':>5} {'Token':>8} {'Count':>12} {'%':>8} {'Repr'}")
    print("-"*70)

    total = len(tokens)
    cumulative = 0

    # Known collapse patterns to highlight
    collapse_patterns = [':', ')', '):', '::', '(', ',', '.']

    for rank, (token_id, count) in enumerate(top_tokens, 1):
        pct = count / total * 100
        cumulative += pct
        decoded = vocab.decode([token_id])

        # Escape for display
        repr_str = repr(decoded)

        # Highlight if this is a collapse pattern
        marker = ""
        for pattern in collapse_patterns:
            if pattern in decoded and len(decoded) <= 3:
                marker = " <-- COLLAPSE?"
                break

        print(f"{rank:>5} {token_id:>8} {count:>12,} {pct:>7.2f}% {repr_str[:30]}{marker}")

    print("-"*70)
    print(f"Top {top_k} tokens cover: {cumulative:.1f}% of all tokens")

    # Check specific collapse tokens
    print(f"\n[6] COLLAPSE TOKEN ANALYSIS")
    print("-"*70)

    # Look for specific patterns
    patterns_to_check = [
        (':',),
        (')',),
        ('):',),
        ('::',),
        ('(',),
        (',',),
        ('.',),
        ('\n',),
        ('    ',),  # 4 spaces (indent)
        ('        ',),  # 8 spaces
    ]

    for pattern in patterns_to_check:
        pattern_str = pattern[0]
        # Find token ID(s) that decode to this pattern
        pattern_ids = []
        for token_id in range(min(vocab.vocab_size, 10000)):
            decoded = vocab.decode([token_id])
            if decoded == pattern_str:
                pattern_ids.append(token_id)

        if pattern_ids:
            total_count = sum(counter.get(tid, 0) for tid in pattern_ids)
            pct = total_count / total * 100
            print(f"  {repr(pattern_str):20} -> IDs {pattern_ids[:5]} -> {total_count:>10,} ({pct:.2f}%)")

    # Distribution stats
    print(f"\n[7] DISTRIBUTION STATISTICS")
    print("-"*70)

    unique_tokens = len(counter)
    print(f"  Unique tokens used: {unique_tokens:,} / {vocab.vocab_size:,} ({unique_tokens/vocab.vocab_size*100:.1f}%)")

    # Tokens appearing only once
    singletons = sum(1 for c in counter.values() if c == 1)
    print(f"  Tokens appearing once: {singletons:,} ({singletons/unique_tokens*100:.1f}%)")

    # Tokens appearing < 10 times
    rare = sum(1 for c in counter.values() if c < 10)
    print(f"  Tokens appearing <10 times: {rare:,} ({rare/unique_tokens*100:.1f}%)")

    # Check if top 100 tokens dominate
    top100_count = sum(c for _, c in counter.most_common(100))
    print(f"  Top 100 tokens cover: {top100_count/total*100:.1f}% of data")

    top1000_count = sum(c for _, c in counter.most_common(1000))
    print(f"  Top 1000 tokens cover: {top1000_count/total*100:.1f}% of data")

    # Save detailed report
    report = {
        "vocab_size": vocab.vocab_size,
        "total_tokens": len(tokens),
        "unique_tokens": unique_tokens,
        "compression_ratio": len(text) / len(tokens),
        "top_50": [
            {
                "rank": i+1,
                "token_id": tid,
                "count": cnt,
                "percentage": cnt / total * 100,
                "decoded": vocab.decode([tid])
            }
            for i, (tid, cnt) in enumerate(top_tokens)
        ]
    }

    report_path = Path(__file__).parent / "token_freq_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[8] Saved report to: {report_path}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    return counter, vocab


if __name__ == "__main__":
    # Paths
    train_path = Path(__file__).parent.parent.parent / "data" / "code_token_bigdata" / "train.txt"
    tokenizer_path = Path(__file__).parent.parent.parent / "data" / "tokenizers" / "python_bpe_8k" / "tokenizer.json"

    if not train_path.exists():
        print(f"ERROR: Training data not found: {train_path}")
        sys.exit(1)

    if not tokenizer_path.exists():
        print(f"ERROR: Tokenizer not found: {tokenizer_path}")
        sys.exit(1)

    analyze_token_distribution(str(train_path), str(tokenizer_path))
