#!/usr/bin/env python3
"""
Simple demo for v1 model (29M params, 8K BPE, few-shot)

Usage:
    python demo_v1.py

    # Custom prompt
    python demo_v1.py --template algorithm --task "def merge_sort(arr: list) -> list:"
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from models import build_model
from datasets import CodeTokenVocab
from search_space import ArchitectureConfig


# Load few-shot templates
TEMPLATE_PATH = Path(__file__).parent / "eval" / "prompts" / "few_shot_templates.json"

with open(TEMPLATE_PATH) as f:
    TEMPLATES = json.load(f)


def generate_with_fewshot(
    model,
    vocab,
    context: str,
    max_new_tokens: int = 60,
    temperature: float = 0.7,
    top_k: int = 40,
    device: str = 'cuda:0'
) -> str:
    """Generate code completion with top-k sampling."""
    tokens = vocab.encode(context)
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if x.size(1) >= 256:  # Max sequence length
                break

            logits = model(x)[0, -1, :] / temperature

            # Top-k filtering
            top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
            probs = torch.softmax(top_k_logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            next_token = top_k_indices[next_idx]

            x = torch.cat([x, next_token.unsqueeze(0)], dim=1)

    return vocab.decode(x[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Demo v1 model with few-shot prompts")
    parser.add_argument("--template", type=str, default="basic_function",
                       choices=list(TEMPLATES['templates'].keys()),
                       help="Template to use")
    parser.add_argument("--task", type=str, default=None,
                       help="Custom task (overrides template example)")
    parser.add_argument("--checkpoint", type=str,
                       default="logs/train_v1_8k_strongreg/v1_8k_strongreg_best.pt",
                       help="Model checkpoint path")
    parser.add_argument("--arch_json", type=str,
                       default="models/codenas_l8h512_regularized.json",
                       help="Architecture JSON path")
    parser.add_argument("--tokenizer", type=str,
                       default="../data/tokenizers/python_bpe_8k/tokenizer.json",
                       help="Tokenizer path")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--samples", type=int, default=1,
                       help="Number of samples to generate")

    args = parser.parse_args()

    # Load model
    print("="*70)
    print("V1 MODEL DEMO (29M params, 8K BPE, Few-Shot)")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Template: {args.template}")
    print()

    # Load architecture
    with open(args.arch_json) as f:
        arch_data = json.load(f)
    arch_cfg = ArchitectureConfig.from_dict(arch_data['architecture'])

    # Load tokenizer
    vocab = CodeTokenVocab(tokenizer_path=args.tokenizer)
    arch_cfg.vocab_size = vocab.vocab_size
    arch_cfg.max_seq_length = 256

    # Build model
    model = build_model(arch_cfg).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"Model loaded (step {ckpt['step']}, val_loss={ckpt['val_loss']:.4f})")
    print()

    # Get template
    template = TEMPLATES['templates'][args.template]
    context = template['context']

    # Add task
    if args.task:
        task = args.task
    else:
        task = template['example_tasks'][0]

    full_prompt = context + task

    print("="*70)
    print("PROMPT")
    print("="*70)
    print(full_prompt)
    print()

    # Generate samples
    print("="*70)
    print("GENERATED COMPLETIONS")
    print("="*70)

    for i in range(args.samples):
        if args.samples > 1:
            print(f"\n--- Sample {i+1}/{args.samples} ---")

        result = generate_with_fewshot(
            model, vocab, full_prompt,
            max_new_tokens=60,
            temperature=0.7,
            top_k=40,
            device=args.device
        )

        output_only = result[len(full_prompt):]
        print(output_only)

    print()
    print("="*70)
    print("NOTES")
    print("="*70)
    print("• Few-shot context reduces collapse rate to ~15% (vs ~90% without)")
    print("• Use 1-2 example functions for best results")
    print("• Model generates Python-like code but may not be task-specific")
    print(f"• See {TEMPLATE_PATH} for more templates")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
