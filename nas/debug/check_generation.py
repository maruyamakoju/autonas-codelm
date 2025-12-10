"""
Debug script to verify generation loop
"""
import sys
sys.path.insert(0, '..')

import torch
import torch.nn.functional as F
from pathlib import Path

from datasets import CodeTokenVocab
from search_space import ArchitectureConfig
from models import build_model

print("=" * 70)
print("Generation Loop Debug")
print("=" * 70)

# Load model
checkpoint_path = Path("../logs/train_v1_bpe8k_bigdata/v1_bpe8k_bigdata_best.pt")
model_path = Path("../models/codenas_best_current.json")

import json
with open(model_path) as f:
    data = json.load(f)
    arch = data.get("architecture", data)
    config = ArchitectureConfig(**arch)

# Load vocab
vocab = CodeTokenVocab(tokenizer_path="../../data/tokenizers/python_bpe_8k/tokenizer.json")
config.vocab_size = vocab.vocab_size

# Build model
model = build_model(config)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load weights
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"\nModel loaded: {config.arch_type} L{config.num_layers} H{config.hidden_dim}")
print(f"Device: {device}")
print(f"Vocab size: {vocab.vocab_size}")

# Test generation
prompt = "def add(a, b):"
print(f"\nPrompt: {repr(prompt)}")

input_ids = vocab.encode(prompt)
print(f"Encoded: {input_ids}")

input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

# Generate 10 tokens and inspect
print(f"\nGenerating 10 tokens (inspecting each step):")
print("-" * 70)

for step in range(10):
    with torch.no_grad():
        logits = model(input_tensor)  # (1, seq_len, vocab_size)
        next_token_logits = logits[0, -1, :]  # (vocab_size,)

        # Check logits
        top5_values, top5_indices = torch.topk(next_token_logits, 5)

        # Apply temperature
        temp_logits = next_token_logits / 0.8
        probs = F.softmax(temp_logits, dim=-1)

        # Sample
        next_token_id = torch.multinomial(probs, num_samples=1).item()

        # Decode
        next_token_text = vocab.decode([next_token_id])

        print(f"Step {step+1}:")
        print(f"  Top-5 logits: {top5_values.tolist()}")
        print(f"  Top-5 indices: {top5_indices.tolist()}")
        print(f"  Sampled ID: {next_token_id}")
        print(f"  Sampled text: {repr(next_token_text)}")
        print(f"  Max prob: {probs.max().item():.4f}")

        # Append
        next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
        input_tensor = torch.cat([input_tensor, next_token_tensor], dim=1)

print("\n" + "=" * 70)
print("Check if sampled IDs are always the same - if yes, bug found!")
print("=" * 70)
