"""
Check token frequency distribution in training corpus
"""
import sys
sys.path.insert(0, '..')

from pathlib import Path
from collections import Counter

from datasets import CodeTokenVocab

def main():
    print("=" * 70)
    print("Token Frequency Analysis")
    print("=" * 70)

    vocab = CodeTokenVocab(
        tokenizer_path="../../data/tokenizers/python_bpe_8k/tokenizer.json"
    )

    # Load first 2MB of training data
    path = Path("../../data/code_token_bigdata/train.txt")
    print(f"\nReading first 2MB from: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        text = f.read(2_000_000)  # First 2MB

    print(f"Text length: {len(text):,} chars")

    # Tokenize
    ids = vocab.encode(text)
    print(f"Token count: {len(ids):,}")

    # Count frequencies
    counter = Counter(ids)
    total = len(ids)

    print(f"\n{'ID':>5}  {'Count':>8}  {'Freq':>7}  {'Token'}")
    print("-" * 70)

    for tid, cnt in counter.most_common(30):
        try:
            piece = vocab.decode([tid])
            print(f"{tid:5d}  {cnt:8d}  {cnt/total:7.3%}  {repr(piece)}")
        except:
            print(f"{tid:5d}  {cnt:8d}  {cnt/total:7.3%}  [decode error]")

    # Check specific tokens of interest
    print("\n" + "=" * 70)
    print("Checking problem tokens:")
    print("-" * 70)

    problem_tokens = [320, 27, 68]  # ):, :, c
    for tid in problem_tokens:
        if tid in counter:
            cnt = counter[tid]
            piece = vocab.decode([tid])
            print(f"Token {tid} ({repr(piece)}): {cnt:,} occurrences ({cnt/total:.3%})")
        else:
            print(f"Token {tid}: Not found in sample")

    print("\n" + "=" * 70)
    print("Analysis:")
    print("  - If any single token is >20%, data is likely broken")
    print("  - If top-5 tokens account for >50%, data has severe bias")
    print("  - Normal code should have more uniform distribution")
    print("=" * 70)

if __name__ == "__main__":
    main()
