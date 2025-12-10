#!/usr/bin/env python3
"""
Quick test of token-level model generation

Test if mode collapse is solved with token-level modeling
"""

import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

from models import build_model
from search_space import ArchitectureConfig

def load_model(checkpoint_path: str, device: str = "cuda:0"):
    """Load trained model from checkpoint"""

    # Architecture config (v1 L4 H256 but with token vocab)
    arch_cfg = ArchitectureConfig(
        arch_type="transformer",
        num_layers=4,
        hidden_dim=256,
        num_heads=8,
        ffn_multiplier=3.0,
        normalization="rmsnorm",
        activation="gelu",
        position_encoding="rope",
        vocab_size=50257,  # GPT-2 tokenizer
        max_seq_length=256
    )

    model = build_model(arch_cfg).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"[MODEL] Loaded from {checkpoint_path}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Checkpoint step: {checkpoint.get('step', 'unknown')}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'unknown'):.4f}")

    return model, arch_cfg


def generate(model, tokenizer, prompt: str, max_tokens: int = 50, temperature: float = 0.8, device: str = "cuda:0"):
    """Generate completion for prompt"""

    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    print(f"\n[PROMPT] {prompt}")
    print(f"[TOKENS] {input_ids.shape[1]} tokens")

    # Generate
    generated = input_ids

    with torch.no_grad():
        for _ in range(max_tokens):
            # Get logits for last position
            logits = model(generated)[:, -1, :]

            # Sample with temperature
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

            # Stop at max length
            if generated.shape[1] >= 256:
                break

    # Decode
    completion = tokenizer.decode(generated[0].cpu().tolist(), skip_special_tokens=False)

    print(f"\n[COMPLETION]\n{completion}\n")
    print("="*70)

    return completion


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("="*70)
    print("TOKEN-LEVEL MODEL GENERATION TEST")
    print("="*70)

    # Load model
    checkpoint_path = "logs/train_v1_token_test/v1_token_test_best.pt"
    model, arch_cfg = load_model(checkpoint_path, device)

    # Load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    print(f"\n[TOKENIZER] GPT-2, vocab size: {len(tokenizer):,}")

    # Test prompts (same as char-level evaluation)
    test_prompts = [
        "def add(a, b):",
        "class DataLoader:",
        "import numpy as np\n",
        "for i in range(10):",
        "x = [1, 2, 3]\n"
    ]

    print("\n" + "="*70)
    print("GENERATION TESTS")
    print("="*70)

    for prompt in test_prompts:
        completion = generate(model, tokenizer, prompt, max_tokens=30, temperature=0.7, device=device)

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
