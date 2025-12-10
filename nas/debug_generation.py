#!/usr/bin/env python3
"""Debug v2 LoRA model generation."""

import json
import torch
from peft import PeftModel
from models import build_model
from datasets import CodeTokenVocab
from search_space import ArchitectureConfig


def main():
    # Load model
    with open('models/codenas_l8h512_regularized.json', encoding='utf-8') as f:
        arch_cfg = ArchitectureConfig.from_dict(json.load(f)['architecture'])
    vocab = CodeTokenVocab(tokenizer_path='../data/tokenizers/python_bpe_8k/tokenizer.json')
    arch_cfg.vocab_size = vocab.vocab_size
    arch_cfg.max_seq_length = 256

    device = 'cuda:0'
    base_model = build_model(arch_cfg).to(device)
    ckpt = torch.load('logs/train_v1_8k_strongreg/v1_8k_strongreg_best.pt', map_location=device)
    base_model.load_state_dict(ckpt['model_state_dict'])
    model = PeftModel.from_pretrained(base_model, 'logs/train_v2_lora/v2_lora_best_lora')
    model.eval()

    print("Model loaded successfully")

    # Try a simple prompt
    prompt = "def add(a, b):\n    \"\"\"Add two numbers\"\"\"\n"
    print(f"Prompt: {prompt!r}")

    # Tokenize
    tokens = vocab.encode(prompt)
    print(f"Tokens ({len(tokens)}): {tokens}")

    # Generate 20 tokens
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    generated_ids = []

    print("\nGenerating tokens:")
    with torch.no_grad():
        for i in range(20):
            logits = model(input_ids)
            next_token = logits[0, -1, :].argmax().item()
            generated_ids.append(next_token)

            # Print token info
            print(f"Step {i+1}: token_id={next_token}")

            # Check if it's a special token
            if next_token == vocab.eos_token_id:
                print(f"  -> EOS token (id={vocab.eos_token_id})")
                break
            elif next_token == vocab.pad_token_id:
                print(f"  -> PAD token (id={vocab.pad_token_id})")

            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)

    print(f"\nGenerated token IDs: {generated_ids}")

    # Decode
    try:
        full_ids = tokens + generated_ids
        decoded = vocab.decode(full_ids)
        print(f"\nDecoded output:\n{decoded}")
    except Exception as e:
        print(f"Decoding error: {e}")

    # Check what the most common tokens are
    print(f"\nToken frequency in first 20 generated:")
    from collections import Counter
    counts = Counter(generated_ids)
    for tid, count in counts.most_common(5):
        print(f"  Token {tid}: appears {count} times")


if __name__ == "__main__":
    main()
