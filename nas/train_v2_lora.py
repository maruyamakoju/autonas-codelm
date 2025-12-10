#!/usr/bin/env python3
"""
v2 LoRA Training: Fine-tune v1 with LoRA on 500 instruction samples

Uses LoRA (Low-Rank Adaptation) to fine-tune v1 model without modifying base weights.

Key differences from v1 instruction tuning:
- LoRA adapters (rank=8, alpha=16) on attention layers
- Base model frozen, only LoRA weights trainable
- Larger dataset (500 vs 49 samples)
- Longer training (20k vs 6k steps)
- Validation split for monitoring

Usage:
    python nas/train_v2_lora.py \
        --init_checkpoint logs/train_v1_8k_strongreg/v1_8k_strongreg_best.pt \
        --arch_json models/codenas_l8h512_regularized.json \
        --tokenizer_path ../data/tokenizers/python_bpe_8k/tokenizer.json \
        --train_path ../data/instruction_tuning/v2_train.jsonl \
        --val_path ../data/instruction_tuning/v2_val.jsonl \
        --output_dir logs/train_v2_lora \
        --max_steps 20000 \
        --lr 1e-4 \
        --batch_size 8
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType

from models import build_model
from datasets import CodeTokenVocab
from datasets_instruction import InstructionCodeDataset
from collate_instruction import collate_instruction_batch
from search_space import ArchitectureConfig


def train_v2_lora(
    init_checkpoint: str,
    arch_json: str,
    tokenizer_path: str,
    train_path: str,
    val_path: str,
    output_dir: str,
    experiment_name: str = 'v2_lora',
    max_steps: int = 20000,
    lr: float = 1e-4,
    batch_size: int = 8,
    warmup_steps: int = 500,
    eval_interval: int = 2000,
    save_interval: int = 2000,
    device: str = 'cuda:0',
    grad_clip: float = 1.0,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
):
    """
    Train v2 with LoRA adapters.
    """
    print("=" * 70)
    print("v2 LoRA TRAINING")
    print("=" * 70)
    print(f"Init checkpoint: {init_checkpoint}")
    print(f"Train data: {train_path}")
    print(f"Val data: {val_path}")
    print(f"Output: {output_dir}")
    print(f"Max steps: {max_steps}")
    print(f"Learning rate: {lr}")
    print(f"Batch size: {batch_size}")
    print(f"LoRA config: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")
    print()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load architecture
    with open(arch_json) as f:
        arch_data = json.load(f)
    arch_cfg = ArchitectureConfig.from_dict(arch_data['architecture'])

    # Load tokenizer
    vocab = CodeTokenVocab(tokenizer_path=tokenizer_path)
    arch_cfg.vocab_size = vocab.vocab_size
    arch_cfg.max_seq_length = 256

    print(f"[MODEL] Loading architecture from {arch_json}")
    print(f"  Layers: {arch_cfg.num_layers}, Hidden: {arch_cfg.hidden_dim}")
    print(f"  Vocab size: {arch_cfg.vocab_size}")
    print()

    # Build base model and load checkpoint
    base_model = build_model(arch_cfg).to(device)
    ckpt = torch.load(init_checkpoint, map_location=device)
    base_model.load_state_dict(ckpt['model_state_dict'])

    print(f"[CHECKPOINT] Loaded v1 from step {ckpt.get('step', 'unknown')}")
    if 'val_loss' in ckpt:
        print(f"  Val loss: {ckpt['val_loss']:.4f}")
    print()

    # Apply LoRA
    print("[LORA] Applying LoRA adapters...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["qkv", "out_proj"],  # Attention projections
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    print()

    # Load datasets
    train_dataset = InstructionCodeDataset(
        path=train_path,
        tokenizer=vocab,
        seq_len=256,
        ignore_index=-100,
    )

    val_dataset = InstructionCodeDataset(
        path=val_path,
        tokenizer=vocab,
        seq_len=256,
        ignore_index=-100,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_instruction_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_instruction_batch,
    )

    # Loss function (ignore prompt tokens)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Optimizer (only LoRA parameters)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def get_lr(step):
        """Linear warmup + constant LR"""
        if step < warmup_steps:
            return lr * step / warmup_steps
        return lr

    # Training loop
    model.train()
    step = 0
    total_loss = 0
    best_val_loss = float('inf')
    start_time = time.time()

    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    # Infinite data loader (cycle through dataset)
    def cycle(loader):
        while True:
            for batch in loader:
                yield batch

    train_iter = cycle(train_loader)

    for step in range(1, max_steps + 1):
        x, y = next(train_iter)
        x = x.to(device)
        y = y.to(device)

        # Forward
        logits = model(x)  # [B, T, V]
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Update LR
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(step)

        optimizer.step()

        # Track loss
        total_loss += loss.item()

        # Log
        if step % 50 == 0:
            avg_loss = total_loss / 50
            elapsed = time.time() - start_time
            speed = 50 / elapsed
            current_lr = get_lr(step)

            print(f"Step {step:5d}/{max_steps} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | {speed:.1f} steps/s")

            total_loss = 0
            start_time = time.time()

        # Validation
        if step % eval_interval == 0:
            model.eval()
            val_loss = 0
            val_count = 0

            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)

                    logits = model(x_val)
                    loss = criterion(logits.view(-1, logits.size(-1)), y_val.view(-1))
                    val_loss += loss.item()
                    val_count += 1

            val_loss /= val_count

            print(f"  [EVAL] Step {step} | Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = output_dir / f"{experiment_name}_best.pt"
                model.save_pretrained(str(output_dir / f"{experiment_name}_best_lora"))
                print(f"  [BEST] Saved LoRA weights: {best_path.parent / (experiment_name + '_best_lora')}")

            model.train()

        # Save checkpoint
        if step % save_interval == 0 or step == max_steps:
            checkpoint_path = output_dir / f"{experiment_name}_step{step}_lora"
            model.save_pretrained(str(checkpoint_path))
            print(f"  [SAVE] LoRA checkpoint: {checkpoint_path}")

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"LoRA weights: {output_dir / (experiment_name + '_best_lora')}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v2 LoRA training")

    # Required
    parser.add_argument("--init_checkpoint", type=str, required=True,
                       help="Path to v1 checkpoint")
    parser.add_argument("--arch_json", type=str, required=True,
                       help="Architecture JSON file")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                       help="Path to BPE tokenizer")
    parser.add_argument("--train_path", type=str, required=True,
                       help="Path to training data (JSONL)")
    parser.add_argument("--val_path", type=str, required=True,
                       help="Path to validation data (JSONL)")

    # Output
    parser.add_argument("--output_dir", type=str, default="logs/train_v2_lora",
                       help="Output directory for LoRA weights")
    parser.add_argument("--experiment_name", type=str, default="v2_lora",
                       help="Experiment name")

    # Training
    parser.add_argument("--max_steps", type=int, default=20000,
                       help="Maximum training steps")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Warmup steps")
    parser.add_argument("--eval_interval", type=int, default=2000,
                       help="Validation interval")
    parser.add_argument("--save_interval", type=int, default=2000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                       help="Gradient clipping norm")

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout")

    args = parser.parse_args()

    train_v2_lora(
        init_checkpoint=args.init_checkpoint,
        arch_json=args.arch_json,
        tokenizer_path=args.tokenizer_path,
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        max_steps=args.max_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        device=args.device,
        grad_clip=args.grad_clip,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
