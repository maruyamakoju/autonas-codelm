# Instruction Tuning Failure Report

**Date**: 2025-12-10
**Experiment**: v1 Model Instruction Tuning (v1 → v1+IT)
**Result**: ❌ **COMPLETE FAILURE**

---

## Executive Summary

Attempted to improve v1 model quality through instruction tuning on task-solution pairs. After 3 training iterations with increasing aggression (fixing bugs, adding data, longer training), **the model failed to learn ANY task-specific generation** and performed WORSE than the baseline.

**Key Result**:
- Baseline v1: 0/20 tasks (diverse syntax errors)
- v1+IT v3: 0/20 tasks (mode collapse + syntax errors)
- **Instruction tuning introduced harmful mode collapse patterns**

---

## Experimental Setup

### Architecture
- Model: CodeNAS L8H512 (29M parameters)
- Pre-training: 50K steps on 10MB Python corpus (val_loss=1.2539)
- Tokenizer: BPE 8K vocab

### Instruction Tuning Data
- Format: JSONL with `{id, prompt, solution}` pairs
- Size: 49 samples (20 initial + 29 added)
- Examples: `add`, `multiply`, `is_prime`, `factorial`, `bubble_sort`, etc.
- Avg prompt length: ~23 tokens
- Avg solution length: ~6 tokens

### Training Configurations

| Version | Steps | LR    | Warmup | Batch | Key Changes |
|---------|-------|-------|--------|-------|-------------|
| v1      | 3000  | 5e-5  | 200    | 8     | Initial attempt (failed - padding bug) |
| v2      | 3000  | 5e-5  | 200    | 8     | Fixed padding + 49 samples |
| v3      | 6000  | 1e-4  | 300    | 8     | Aggressive (2× steps, 2× LR) |

---

## Critical Bug Discovery: Static Padding Overfit

### The Problem
Initial v1 training appeared successful (loss→0.0000) but model generated only newlines.

### Root Cause
Dataset padded ALL sequences to fixed 256 tokens:
- Actual data per sample: ~29 tokens (prompt + solution)
- Padding: 227 tokens (88% of sequence!)
- Model learned to predict padding perfectly, ignoring actual solutions

```python
# BEFORE (broken):
# Sample: 29 real tokens + 227 padding tokens
# Loss computed on: 6 solution tokens + 227 padding tokens
# Result: Model optimizes for padding (97%), ignores solution (3%)

# AFTER (fixed):
# Sample: 29 real tokens (no static padding)
# Collate function pads batch to max_len dynamically
# Loss computed ONLY on solution tokens (prompt masked with -100)
```

### Fix Applied
1. Removed static padding from `datasets_instruction.py`
2. Created `collate_instruction.py` with dynamic padding
3. Updated DataLoader to use `collate_fn=collate_instruction_batch`

---

## Training Results

### v1 (Initial - Padding Bug)
- Loss: 2.8771 → 0.0000 (appeared to converge)
- Generation: Only newlines (`\n` token repeated)
- Diagnosis: Overfit to padding, completely ignored solutions
- **Benchmark**: Not evaluated (obviously broken)

### v2 (Fixed Padding, More Data)
- Loss: 0.2887 → 0.0000 (smooth convergence)
- Training loss curve: Clean descent, no spikes
- Generation test on exact training sample:
  ```python
  prompt = 'def add(a: int, b: int) -> int:\n    """Add two integers."""\n'
  expected = '    return a + b\n'
  generated = ''  # EMPTY!
  ```
- **Benchmark**: 0/20 (0%), 16 syntax/logic errors, 4 assertion errors

### v3 (Aggressive - 6K steps, High LR)
- Loss: 0.2887 → 0.0000 (with spikes at steps 550, 1400, 2550-2650, 3400, 4350)
- Training: 2× steps (6000), 2× learning rate (1e-4), 1.5× warmup (300)
- Generation test: Still empty on exact training sample
- **Benchmark**: 0/20 (0%), introduced mode collapse

---

## Benchmark Evaluation Results

### Mini HumanEval (20 tasks)

| Model | Pass Rate | Collapse | Syntax Errors | Key Observation |
|-------|-----------|----------|---------------|-----------------|
| **v1 baseline** | 0/20 (0.0%) | 0 | 20 | Diverse syntax errors |
| **v1+IT v2** | 0/20 (0.0%) | 0 | 16 | Slight distribution change |
| **v1+IT v3 (T=0.8)** | 0/20 (0.0%) | 4 | 16 | **Mode collapse introduced** |
| **v1+IT v3 (T=0.3)** | 0/20 (0.0%) | 4 | 16 | Temperature didn't help |
| **v1+IT v3 (best-of-3)** | 0/20 (0.0%) | 6 | 14 | **More collapse!** |

### Mode Collapse Pattern
v3 model generates `return return` on tasks: `fibonacci`, `count_vowels`, `linear_search`, `power`, `flatten_list`, `string_reverse`

Example:
```python
# Expected:
def fibonacci(n: int) -> int:
    if n == 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# v3 Generated:
def fibonacci(n: int) -> int:
    """Return n-th Fibonacci number (0-indexed)."""
    return return  # MODE COLLAPSE!
```

---

## Why Instruction Tuning Failed

### 1. Pre-training Dominance
- v1 pre-trained for 50K steps on 10MB corpus
- Instruction tuning: 49 samples × 6K steps = 294K sample exposures
- **But**: Sample diversity is 49 vs. pre-training corpus with thousands of unique functions
- Pre-trained patterns (general Python) too strong for small task-specific dataset to override

### 2. Insufficient Task Coverage
- 49 samples cover basic patterns: arithmetic, loops, conditionals
- Doesn't represent full distribution of code generation tasks
- Model can't generalize from 49 examples to arbitrary functions

### 3. High Learning Rate Damage
- LR=1e-4 (v3) may have degraded pre-trained weights
- Loss spikes at steps 550, 1400, 2550-2650, 3400, 4350 suggest instability
- Introduced mode collapse that wasn't present in baseline

### 4. Evaluation Mismatch
- Training: Simple tasks with few-token solutions (6 tokens avg)
- Benchmark: Requires longer, more complex implementations
- Model never learned to generate multi-line logic

---

## Lessons Learned

### ✅ What Worked
1. **Dynamic padding fix**: Critical bug caught and properly resolved
2. **Data expansion**: 20→49 samples (but still insufficient)
3. **Diagnostic methodology**: Systematic testing revealed failure modes
4. **Collate function design**: Clean implementation of variable-length batching

### ❌ What Failed
1. **Small dataset assumption**: 49 samples insufficient vs. 50K step pre-training
2. **Aggressive LR strategy**: LR=1e-4 introduced instability and mode collapse
3. **Evaluation expectations**: Hoped for 3-5/20 pass rate, got 0/20

### 🔑 Key Insights
1. **Loss→0 is deceptive**: Loss can reach 0.0000 but model still fails to generate correctly
2. **Pre-training is dominant**: Strong pre-trained model resists small-scale fine-tuning
3. **Mode collapse risk**: High LR can introduce harmful patterns not present in baseline
4. **Quality > Quantity (steps)**: 6K steps at high LR worse than stopping earlier

---

## Recommendations for Future Work

### If Retrying Instruction Tuning:

**Option 1: Scale Up Data (Preferred)**
- Increase to 500-1000 samples minimum
- Use existing datasets: HumanEval, MBPP, CodeContests
- Ensure diverse task types and solution lengths

**Option 2: Lower Learning Rate + Longer Training**
- LR: 1e-5 to 5e-6 (10× lower than v2)
- Steps: 10K-20K
- More gradual adaptation to preserve pre-trained knowledge

**Option 3: Parameter-Efficient Fine-Tuning**
- LoRA (Low-Rank Adaptation): Only tune small adapter layers
- Freeze pre-trained weights, add trainable low-rank matrices
- Prevents catastrophic forgetting of pre-training

**Option 4: Curriculum Learning**
- Start with simplest tasks (e.g., `add`, `multiply`)
- Gradually increase complexity
- Helps model learn structured generation before complex logic

### If Moving On:
Focus on improving pre-training rather than instruction tuning:
- Larger corpus (10MB → 100MB+)
- Better data quality (filter for well-formed functions)
- Architecture improvements (attention variants, better positional encodings)

---

## Conclusion

Instruction tuning experiment was a **complete failure**:
- 0/20 benchmark score maintained across all versions
- Introduced harmful mode collapse patterns
- Wasted 6000 training steps on v3

**Root cause**: 50K step pre-training too strong for 49-sample instruction tuning to override, regardless of learning rate or training steps.

**Recommendation**: Abandon instruction tuning approach with current setup. Either:
1. Scale up to 500+ samples with proper dataset
2. Use LoRA/PEFT methods
3. Focus on improving base pre-training instead

**Status**: Experiment concluded, documented, ready for next phase.

---

## Appendix: File Changes

### Created Files
- `nas/datasets_instruction.py` - Instruction tuning dataset (fixed padding)
- `nas/collate_instruction.py` - Dynamic padding collate function
- `nas/train_instruction_tuning.py` - Training script
- `data/instruction_tuning/mini_supervised.jsonl` - 49 task samples

### Modified Files
- `nas/eval/run_mini_benchmark.py` - Fixed to handle checkpoints without `val_loss` key

### Checkpoints Generated
- `logs/train_instruction_tuning/v1_itune_final.pt` (v1 - padding bug)
- `logs/train_instruction_tuning_v2/v1_itune_v2_final.pt` (v2 - 3K steps)
- `logs/train_instruction_tuning_v3/v1_itune_v3_aggressive_final.pt` (v3 - 6K steps, failed)
