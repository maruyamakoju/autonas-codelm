"""
Instruction Tuning Dataset for Code Generation

Loads task-solution pairs and prepares them for supervised fine-tuning.
"""

import json
import torch
from torch.utils.data import Dataset
from typing import List, Tuple


class InstructionCodeDataset(Dataset):
    """
    Dataset for instruction tuning on code tasks.

    Format: Each sample has 'prompt' (task) and 'solution' (implementation).
    Loss is computed only on the solution part (prompt is ignored).
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        seq_len: int = 256,
        ignore_index: int = -100,
        few_shot_prefix: str = ""
    ):
        """
        Args:
            path: Path to JSONL file with {'id', 'prompt', 'solution'}
            tokenizer: Tokenizer with encode() method
            seq_len: Maximum sequence length
            ignore_index: Index to use for ignored tokens (prompt part)
            few_shot_prefix: Optional few-shot context to prepend to prompt
        """
        self.samples = []
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.ignore_index = ignore_index

        # Load data
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)
                prompt = obj['prompt']
                solution = obj['solution']

                # Build full context
                full_text = few_shot_prefix + prompt + solution

                # Tokenize
                prompt_ids = tokenizer.encode(few_shot_prefix + prompt)
                solution_ids = tokenizer.encode(solution)

                # Build input (x) and target (y)
                # For autoregressive modeling: model(x[:i]) predicts x[i+1]
                # So output[i] predicts input[i+1]
                #
                # full_seq = prompt + solution
                # x = full_seq[:-1]  (input up to second-to-last token)
                # y = full_seq[1:]   (shifted by 1, but ignore prompt part)
                #
                # We want to ignore loss on prompt tokens, only compute on solution
                full_seq = prompt_ids + solution_ids
                x = full_seq[:-1]
                y_raw = full_seq[1:]

                # Mask out prompt tokens (set to ignore_index)
                y = [ignore_index] * (len(prompt_ids) - 1) + solution_ids

                # Truncate if needed (but keep variable length, no padding)
                if len(x) > seq_len:
                    x = x[:seq_len]
                    y = y[:seq_len]

                # DON'T pad here - will pad dynamically in collate_fn
                # This ensures we don't train on meaningless padding tokens
                self.samples.append({
                    'id': obj['id'],
                    'x': torch.tensor(x, dtype=torch.long),
                    'y': torch.tensor(y, dtype=torch.long),
                    'prompt_len': len(prompt_ids),
                    'solution_len': len(solution_ids),
                    'seq_len': len(x)
                })

        print(f"[InstructionDataset] Loaded {len(self.samples)} samples from {path}")
        if len(self.samples) > 0:
            avg_prompt_len = sum(s['prompt_len'] for s in self.samples) / len(self.samples)
            avg_sol_len = sum(s['solution_len'] for s in self.samples) / len(self.samples)
            print(f"  Avg prompt length: {avg_prompt_len:.1f} tokens")
            print(f"  Avg solution length: {avg_sol_len:.1f} tokens")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['x'], sample['y']


class InstructionCodeDatasetWithContext(InstructionCodeDataset):
    """
    Variant with few-shot context prepended to each prompt.

    This helps the model avoid mode collapse during generation.
    """

    DEFAULT_FEW_SHOT_CONTEXT = """# Python function examples

def add(a: int, b: int) -> int:
    \"\"\"Add two integers.\"\"\"
    return a + b

def multiply(a: int, b: int) -> int:
    \"\"\"Multiply two integers.\"\"\"
    return a * b

# Task
"""

    def __init__(self, path: str, tokenizer, seq_len: int = 256, ignore_index: int = -100):
        super().__init__(
            path=path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            ignore_index=ignore_index,
            few_shot_prefix=self.DEFAULT_FEW_SHOT_CONTEXT
        )
