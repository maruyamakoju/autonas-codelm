#!/usr/bin/env python3
"""
Instruction Tuning for v1 Model

Fine-tune the v1 checkpoint on task-solution pairs to improve task-specific generation.

Usage:
    python train_instruction_tuning.py \
        --init_checkpoint logs/train_v1_8k_strongreg/v1_8k_strongreg_best.pt \
        --data_path ../data/instruction_tuning/mini_supervised.jsonl \
        --output_dir logs/train_instruction_tuning \
        --max_steps 2000 \
        --lr 5e-5 \
        --batch_size 8
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import build_model
from datasets import CodeTokenVocab
from datasets_instruction import InstructionCodeDataset
from collate_instruction import collate_instruction_batch
from search_space import ArchitectureConfig


def train_instruction_tuning(
    init_checkpoint: str,
    arch_json: str,
    tokenizer_path: str,
    data_path: str,
    output_dir: str,
    experiment_name: str,
    max_steps: int = 2000,
    lr: float = 5e-5,
    batch_size: int = 8,
    warmup_steps: int = 200,
    eval_interval: int = 200,
    save_interval: int = 500,
    device: str = 'cuda:0',
    grad_clip: float = 1.0,
):
    """
    Fine-tune v1 model on instruction data.
    """
    print("=" * 70)
    print("INSTRUCTION TUNING (v1 -> v1+IT)")
    print("=" * 70)
    print(f"Init checkpoint: {init_checkpoint}")
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print(f"Max steps: {max_steps}")
    print(f"Learning rate: {lr}")
    print(f"Batch size: {batch_size}")
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

    # Build model and load checkpoint
    model = build_model(arch_cfg).to(device)
    ckpt = torch.load(init_checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    print(f"[CHECKPOINT] Loaded from step {ckpt.get('step', 'unknown')}")
    print(f"  Val loss: {ckpt.get('val_loss', 'N/A'):.4f}" if 'val_loss' in ckpt else "")
    print()

    # Load instruction dataset
    dataset = InstructionCodeDataset(
        path=data_path,
        tokenizer=vocab,
        seq_len=256,
        ignore_index=-100,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_instruction_batch,  # Dynamic padding
    )

    # Loss function (ignore prompt tokens)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Optimizer with warmup
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
    start_time = time.time()

    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    # Infinite data loader (cycle through dataset)
    def cycle(loader):
        while True:
            for batch in loader:
                yield batch

    data_iter = cycle(loader)

    for step in range(1, max_steps + 1):
        x, y = next(data_iter)
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

        # Save checkpoint
        if step % save_interval == 0 or step == max_steps:
            checkpoint_path = output_dir / f"{experiment_name}_step{step}.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss if step % 50 == 0 else 0.0,
            }, checkpoint_path)
            print(f"  [SAVE] Checkpoint saved: {checkpoint_path}")

    # Save final checkpoint
    final_path = output_dir / f"{experiment_name}_final.pt"
    torch.save({
        'step': max_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Final checkpoint: {final_path}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Instruction tuning for v1 model")

    # Required
    parser.add_argument("--init_checkpoint", type=str, required=True,
                       help="Path to v1 checkpoint to fine-tune")
    parser.add_argument("--arch_json", type=str, required=True,
                       help="Architecture JSON file")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                       help="Path to BPE tokenizer")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to instruction data (JSONL)")

    # Output
    parser.add_argument("--output_dir", type=str, default="logs/train_instruction_tuning",
                       help="Output directory for checkpoints")
    parser.add_argument("--experiment_name", type=str, default="v1_itune",
                       help="Experiment name for checkpoint files")

    # Training
    parser.add_argument("--max_steps", type=int, default=2000,
                       help="Maximum training steps")
    parser.add_argument("--lr", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--warmup_steps", type=int, default=200,
                       help="Warmup steps")
    parser.add_argument("--save_interval", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                       help="Gradient clipping norm")

    args = parser.parse_args()

    train_instruction_tuning(
        init_checkpoint=args.init_checkpoint,
        arch_json=args.arch_json,
        tokenizer_path=args.tokenizer_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        max_steps=args.max_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        save_interval=args.save_interval,
        device=args.device,
        grad_clip=args.grad_clip,
    )
