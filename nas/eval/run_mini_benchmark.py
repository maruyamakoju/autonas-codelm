#!/usr/bin/env python3
"""
Mini HumanEval Benchmark for v1 Model

Evaluates model on 20 simple Python tasks with executable tests.

Usage:
    python eval/run_mini_benchmark.py

    # Custom settings
    python eval/run_mini_benchmark.py \
      --checkpoint logs/train_v1_8k_strongreg/v1_8k_strongreg_best.pt \
      --temperature 0.8 \
      --samples 3
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import build_model
from datasets import CodeTokenVocab
from search_space import ArchitectureConfig


def generate_completion(
    model,
    vocab,
    context: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    device: str = 'cuda:0'
) -> str:
    """Generate code completion."""
    tokens = vocab.encode(context)
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if x.size(1) >= 256:
                break

            logits = model(x)[0, -1, :] / temperature
            top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
            probs = torch.softmax(top_k_logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            next_token = top_k_indices[next_idx]

            x = torch.cat([x, next_token.unsqueeze(0)], dim=1)

    return vocab.decode(x[0].tolist())


def extract_function_body(full_completion: str, prompt: str) -> str:
    """Extract function body from completion (remove prompt)."""
    if prompt in full_completion:
        return full_completion[len(prompt):]
    return full_completion


def test_completion(task_id: str, prompt: str, completion: str, tests: List[str]) -> Dict:
    """Test a completion against test cases."""
    result = {
        'task_id': task_id,
        'passed': False,
        'error': None,
        'collapse_detected': False
    }

    # Check for mode collapse patterns
    collapse_patterns = [
        ')' * 10,
        ':' * 10,
        '#' * 20,
        'def def',
        'return return',
    ]

    for pattern in collapse_patterns:
        if pattern in completion:
            result['collapse_detected'] = True
            result['error'] = f"Mode collapse: {repr(pattern)}"
            return result

    # Try to execute
    full_code = prompt + completion

    try:
        # Create namespace
        namespace = {}

        # Execute function definition
        exec(full_code, namespace)

        # Run tests
        for test in tests:
            exec(test, namespace)

        result['passed'] = True

    except SyntaxError as e:
        result['error'] = f"SyntaxError: {e}"
    except AssertionError as e:
        result['error'] = f"AssertionError: {e}"
    except Exception as e:
        result['error'] = f"{type(e).__name__}: {e}"

    return result


def load_few_shot_template(template_name: str = "basic_function") -> str:
    """Load few-shot template."""
    template_path = Path(__file__).parent / "prompts" / "few_shot_templates.json"

    with open(template_path) as f:
        templates = json.load(f)

    template = templates['templates'].get(template_name)
    if not template:
        raise ValueError(f"Template {template_name} not found")

    return template['context']


def run_benchmark(
    checkpoint_path: str,
    arch_json_path: str,
    tokenizer_path: str,
    benchmark_path: str = "eval/benchmarks/mini_humaneval.jsonl",
    device: str = 'cuda:0',
    temperature: float = 0.8,
    samples: int = 1,
    template: str = "basic_function"
) -> Dict:
    """Run mini benchmark."""

    # Load model
    print("="*70)
    print("MINI HUMANEVAL BENCHMARK (v1 Model)")
    print("="*70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Template: {template}")
    print(f"Samples per task: {samples}")
    print(f"Temperature: {temperature}")
    print()

    # Load architecture
    with open(arch_json_path) as f:
        arch_data = json.load(f)
    arch_cfg = ArchitectureConfig.from_dict(arch_data['architecture'])

    # Load tokenizer
    vocab = CodeTokenVocab(tokenizer_path=tokenizer_path)
    arch_cfg.vocab_size = vocab.vocab_size
    arch_cfg.max_seq_length = 256

    # Build model
    model = build_model(arch_cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"Model loaded (step {ckpt['step']}, val_loss={ckpt['val_loss']:.4f})")
    print()

    # Load benchmark
    benchmark_path = Path(__file__).parent.parent / benchmark_path
    tasks = []
    with open(benchmark_path) as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))

    print(f"Loaded {len(tasks)} tasks")
    print()

    # Load few-shot context
    few_shot_context = load_few_shot_template(template)

    # Run benchmark
    results = []
    passed_count = 0
    collapse_count = 0

    print("="*70)
    print("RUNNING BENCHMARK")
    print("="*70)

    for task in tasks:
        task_id = task['id']
        prompt = task['prompt']
        tests = task['tests']

        # Generate with few-shot context
        full_prompt = few_shot_context + prompt

        best_result = None

        for sample_idx in range(samples):
            # Generate
            full_completion = generate_completion(
                model, vocab, full_prompt,
                max_new_tokens=100,
                temperature=temperature,
                top_k=40,
                device=device
            )

            # Extract function body
            completion = extract_function_body(full_completion, full_prompt)

            # Test
            result = test_completion(task_id, prompt, completion, tests)

            if result['passed']:
                best_result = result
                break  # Success on first try

            if best_result is None or (result['collapse_detected'] and not best_result['collapse_detected']):
                best_result = result

        # Record result
        results.append(best_result)

        if best_result['passed']:
            passed_count += 1
            status = "[PASS]"
        elif best_result['collapse_detected']:
            collapse_count += 1
            status = "[COLLAPSE]"
        else:
            status = "[FAIL]"

        error_msg = f" ({best_result['error'][:40]}...)" if best_result['error'] else ""
        print(f"{task_id:20s} {status}{error_msg}")

    # Summary
    total = len(tasks)
    failed = total - passed_count

    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total tasks: {total}")
    print(f"Passed: {passed_count} ({passed_count/total*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total*100:.1f}%)")
    print(f"  - Mode collapse: {collapse_count}")
    print(f"  - Logic/syntax errors: {failed - collapse_count}")
    print()

    return {
        'total': total,
        'passed': passed_count,
        'failed': failed,
        'collapse': collapse_count,
        'pass_rate': passed_count / total,
        'results': results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mini HumanEval benchmark")
    parser.add_argument("--checkpoint", type=str,
                       default="logs/train_v1_8k_strongreg/v1_8k_strongreg_best.pt")
    parser.add_argument("--arch_json", type=str,
                       default="models/codenas_l8h512_regularized.json")
    parser.add_argument("--tokenizer", type=str,
                       default="../data/tokenizers/python_bpe_8k/tokenizer.json")
    parser.add_argument("--benchmark", type=str,
                       default="eval/benchmarks/mini_humaneval.jsonl")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--samples", type=int, default=1,
                       help="Samples per task (best of N)")
    parser.add_argument("--template", type=str, default="basic_function",
                       choices=["basic_function", "recursive_function", "algorithm"])
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file")

    args = parser.parse_args()

    summary = run_benchmark(
        checkpoint_path=args.checkpoint,
        arch_json_path=args.arch_json,
        tokenizer_path=args.tokenizer,
        benchmark_path=args.benchmark,
        device=args.device,
        temperature=args.temperature,
        samples=args.samples,
        template=args.template
    )

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to: {args.output}")
