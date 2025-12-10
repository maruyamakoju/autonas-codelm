#!/usr/bin/env python3
"""
Quick test of v2 LoRA model inference.

Tests the model on a few simple code completion prompts.
"""

import json
import torch
from peft import PeftModel
from models import build_model
from datasets import CodeTokenVocab
from search_space import ArchitectureConfig


def generate_code(model, tokenizer, prompt, max_tokens=100, temperature=0.7, device='cuda:0'):
    """Generate code completion from prompt."""
    model.eval()

    # Tokenize prompt
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

    # Generate
    with torch.no_grad():
        for _ in range(max_tokens):
            # Forward pass
            logits = model(input_ids)  # [1, seq_len, vocab_size]
            next_token_logits = logits[0, -1, :]  # [vocab_size]

            # Sample next token
            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            else:
                next_token = next_token_logits.argmax().item()

            # Stop if EOS
            if next_token == tokenizer.eos_token_id:
                break

            # Append token
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)

    # Decode
    generated_tokens = input_ids[0].cpu().tolist()
    generated_text = tokenizer.decode(generated_tokens)

    return generated_text


def main():
    print("=" * 70)
    print("v2 LoRA MODEL TEST")
    print("=" * 70)

    # Load architecture
    arch_json = "models/codenas_l8h512_regularized.json"
    with open(arch_json) as f:
        arch_data = json.load(f)
    arch_cfg = ArchitectureConfig.from_dict(arch_data['architecture'])

    # Load tokenizer
    tokenizer_path = "../data/tokenizers/python_bpe_8k/tokenizer.json"
    vocab = CodeTokenVocab(tokenizer_path=tokenizer_path)
    arch_cfg.vocab_size = vocab.vocab_size
    arch_cfg.max_seq_length = 256

    print(f"[MODEL] Loading base model...")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load base model
    base_model = build_model(arch_cfg).to(device)

    # Load v1 checkpoint
    v1_checkpoint = "logs/train_v1_8k_strongreg/v1_8k_strongreg_best.pt"
    ckpt = torch.load(v1_checkpoint, map_location=device)
    base_model.load_state_dict(ckpt['model_state_dict'])
    print(f"[OK] v1 checkpoint loaded (step {ckpt.get('step', 'unknown')})")

    # Load LoRA adapters
    lora_path = "logs/train_v2_lora/v2_lora_best_lora"
    model = PeftModel.from_pretrained(base_model, lora_path)
    print(f"[OK] LoRA adapters loaded from {lora_path}")
    print()

    # Test prompts (match training format: signature + closed docstring)
    test_prompts = [
        # Simple function
        "def add(a, b):\n    \"\"\"Add two numbers\"\"\"\n",

        # List operation
        "def sum_list(numbers):\n    \"\"\"Sum all numbers in a list\"\"\"\n",

        # String operation
        "def reverse_string(s):\n    \"\"\"Reverse a string\"\"\"\n",
    ]

    print("=" * 70)
    print("GENERATION TESTS")
    print("=" * 70)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[Test {i}]")
        print(f"Prompt:\n{prompt}")
        print(f"\nGenerated:")

        generated = generate_code(
            model, vocab, prompt,
            max_tokens=50,
            temperature=0.3,  # Low temp for more deterministic output
            device=device
        )

        # Extract just the completion (remove prompt)
        completion = generated[len(prompt):]
        print(completion)
        print("-" * 70)

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
