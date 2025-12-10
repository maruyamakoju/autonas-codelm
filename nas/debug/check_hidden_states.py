"""
Check if hidden states collapse to same vector
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
print("Hidden State Analysis")
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

# Test different prompts
prompts = [
    "def add(a, b):",
    "class MyClass:",
    "import numpy as np",
    "x = 42",
    "for i in range(10):"
]

print(f"\nTesting {len(prompts)} different prompts...")
print("-" * 70)

hidden_states = []

for prompt in prompts:
    input_ids = vocab.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(input_tensor)  # (1, seq_len, vocab_size)
        # Get last hidden state before output projection
        # Need to extract from model internals

        # For now, just check the logits
        last_logits = logits[0, -1, :]  # (vocab_size,)

        # Get top-5
        top5_values, top5_indices = torch.topk(last_logits, 5)

        print(f"\nPrompt: {repr(prompt)}")
        print(f"  Top-5 logits: {top5_values.tolist()}")
        print(f"  Top-5 IDs: {top5_indices.tolist()}")
        print(f"  Top-5 tokens: {[vocab.decode([i]) for i in top5_indices.tolist()]}")

        # Check if 320 is always top
        if top5_indices[0].item() == 320:
            print("  *** Token 320 is #1 ***")

print("\n" + "=" * 70)
print("If Token 320 is always #1 regardless of prompt,")
print("then the model's hidden states likely collapsed.")
print("=" * 70)
