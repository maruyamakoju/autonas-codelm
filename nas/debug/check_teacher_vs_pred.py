"""
Check teacher forcing: true next token vs model prediction
"""
import sys
sys.path.insert(0, '..')

import torch
from pathlib import Path
import json

from search_space import ArchitectureConfig
from models import build_model
from datasets import CodeTokenVocab

print("=" * 70)
print("Teacher Forcing Analysis: True vs Predicted Tokens")
print("=" * 70)

# Load model
checkpoint_path = Path("../logs/train_v1_bpe8k_bigdata/v1_bpe8k_bigdata_best.pt")
model_path = Path("../models/codenas_best_current.json")

with open(model_path) as f:
    data = json.load(f)
    arch = data.get("architecture", data)
    config = ArchitectureConfig(**arch)

vocab = CodeTokenVocab(tokenizer_path="../../data/tokenizers/python_bpe_8k/tokenizer.json")
config.vocab_size = vocab.vocab_size

model = build_model(config)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"\nModel loaded: {config.arch_type} L{config.num_layers} H{config.hidden_dim}")
print(f"Device: {device}")

# Load a snippet from train.txt containing "def add(a, b):"
train_path = Path("../../data/code_token_bigdata/train.txt")
print(f"\nSearching for 'def add(a, b):' in {train_path}...")

with open(train_path, 'r', encoding='utf-8') as f:
    content = f.read(500_000)  # Read first 500KB

# Find snippet
target = "def add(a, b):"
idx = content.find(target)

if idx == -1:
    print("Pattern not found, using synthetic snippet instead")
    snippet = "def add(a, b):\n    return a + b\n\ndef subtract(x, y):\n    return x - y\n"
else:
    # Extract context: 50 chars before, 150 chars after
    start = max(0, idx - 50)
    end = min(len(content), idx + 150)
    snippet = content[start:end]
    print(f"Found at position {idx}")

print(f"\nSnippet ({len(snippet)} chars):")
print("-" * 70)
print(repr(snippet))
print("-" * 70)

# Tokenize
input_ids = vocab.encode(snippet)
print(f"\nTokenized to {len(input_ids)} tokens")

# Prepare input and targets
x = torch.tensor([input_ids[:-1]], dtype=torch.long).to(device)  # All but last
y_true = input_ids[1:]  # Shifted by 1 (true next tokens)

print(f"\nInput shape: {x.shape}")
print(f"Target length: {len(y_true)}")

# Get model predictions
with torch.no_grad():
    logits = model(x)  # (1, seq_len, vocab_size)
    pred_ids = logits[0].argmax(dim=-1).cpu().tolist()  # (seq_len,)

# Compare position by position
print(f"\n{'Pos':>4}  {'True ID':>8}  {'Pred ID':>8}  {'Match':>5}  {'True Token':>20}  {'Pred Token':>20}")
print("-" * 100)

matches = 0
total = min(len(y_true), len(pred_ids))

for t in range(total):
    true_id = y_true[t]
    pred_id = pred_ids[t]
    match = "YES" if true_id == pred_id else "NO"

    true_token = vocab.decode([true_id])
    pred_token = vocab.decode([pred_id])

    if match == "YES":
        matches += 1

    # Show all positions, but highlight mismatches
    marker = "***" if match == "NO" else "   "
    print(f"{marker} {t:4d}  {true_id:8d}  {pred_id:8d}  {match:>5}  {repr(true_token):>20}  {repr(pred_token):>20}")

accuracy = matches / total * 100
print("-" * 100)
print(f"\nTeacher forcing accuracy: {matches}/{total} = {accuracy:.1f}%")

# Analyze mismatch positions
print("\n" + "=" * 70)
print("Analysis:")
if accuracy > 95:
    print(f"  - Model predictions match ground truth {accuracy:.1f}% of the time")
    print("  - This suggests the model LEARNED the training distribution correctly")
    print("  - Mode collapse during generation is likely due to:")
    print("    1. Different behavior without teacher forcing")
    print("    2. Model defaulting to high-frequency patterns")
    print("    3. Insufficient capacity for diverse context handling")
elif accuracy > 80:
    print(f"  - Model predictions match ground truth {accuracy:.1f}% of the time")
    print("  - Partial learning - some positions consistently mispredicted")
    print("  - Check if mismatches occur at specific token types")
elif accuracy < 50:
    print(f"  - Model predictions match ground truth only {accuracy:.1f}% of the time")
    print("  - Model failed to learn the training distribution")
    print("  - This indicates a fundamental training issue")
print("=" * 70)
