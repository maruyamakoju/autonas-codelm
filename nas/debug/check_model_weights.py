"""
Check if model output layer weights collapsed
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
print("Model Weight Analysis")
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

print(f"\nChecking output projection layer (lm_head)...")

# Get output layer
lm_head = None
for name, module in model.named_modules():
    if "lm_head" in name or "output" in name:
        print(f"Found layer: {name}")
        lm_head = module
        break

if lm_head is None:
    # Try to find the last linear layer
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            lm_head = module
            print(f"Using last linear: {name}")

if lm_head is not None and isinstance(lm_head, torch.nn.Linear):
    weight = lm_head.weight.data  # (vocab_size, hidden_dim)
    print(f"\nOutput layer shape: {weight.shape}")
    print(f"Vocab size: {weight.shape[0]}")
    print(f"Hidden dim: {weight.shape[1]}")

    # Check token 320 (the collapsed token)
    token_320_weight = weight[320, :]
    print(f"\nToken 320 ('):') weight stats:")
    print(f"  Mean: {token_320_weight.mean().item():.6f}")
    print(f"  Std: {token_320_weight.std().item():.6f}")
    print(f"  Min: {token_320_weight.min().item():.6f}")
    print(f"  Max: {token_320_weight.max().item():.6f}")
    print(f"  Norm: {token_320_weight.norm().item():.6f}")

    # Compare with other tokens
    print(f"\nComparing with 10 random tokens:")
    import random
    for i in random.sample(range(weight.shape[0]), 10):
        token_weight = weight[i, :]
        print(f"  Token {i}: norm={token_weight.norm().item():.6f}, mean={token_weight.mean().item():.6f}")

    # Check if token 320 has unusually high norm
    all_norms = weight.norm(dim=1)
    print(f"\nAll token norms:")
    print(f"  Mean: {all_norms.mean().item():.6f}")
    print(f"  Std: {all_norms.std().item():.6f}")
    print(f"  Min: {all_norms.min().item():.6f}")
    print(f"  Max: {all_norms.max().item():.6f}")
    print(f"  Token 320 norm: {all_norms[320].item():.6f}")
    print(f"  Token 320 percentile: {(all_norms < all_norms[320]).float().mean().item() * 100:.1f}%")
else:
    print("Could not find output layer!")

print("\n" + "=" * 70)
