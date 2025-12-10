# v2 LoRA Training Results

**Date**: 2025-12-10
**Approach**: LoRA (Low-Rank Adaptation) fine-tuning on 500 instruction samples
**Outcome**: ❌ **FAILED - Mode Collapse**

---

## Summary

Attempted to scale up instruction tuning using LoRA to prevent catastrophic forgetting. Training metrics showed significant improvement, but the model exhibits degenerate behavior (newline generation loop) similar to v1+IT failures.

**Key Result**: Training loss improvements are **misleading** - model does not perform meaningful code completion.

---

## Approach

### Data
- **Source**: HumanEval (164 tasks) + MBPP (386 sampled from 879)
- **Total**: 550 samples
  - Train: 500 samples
  - Val: 50 samples
- **Format**: `{"id": str, "prompt": str, "solution": str}`
  - Prompt: function signature + docstring
  - Solution: function body (indented)

### Model
- **Base**: v1 model (step 50,000, val_loss=1.2539)
- **Method**: PEFT LoRA
  - Rank: 8
  - Alpha: 16
  - Dropout: 0.05
  - Target modules: `["qkv", "out_proj"]` (attention layers only)
- **Trainable params**: 196,608 / 29,377,536 (0.67%)

### Training
- **Steps**: 20,000
- **Learning rate**: 1e-4 (linear warmup 500 steps)
- **Batch size**: 8
- **Validation**: Every 2,000 steps
- **Duration**: ~7 minutes (~46 steps/s)

---

## Training Metrics

### Loss Progression

| Step | Train Loss | Val Loss | Status |
|------|-----------|----------|---------|
| 50 | 0.4205 | 0.4377 | Initial |
| 2000 | 0.1866 | **0.2224** | ✅ Best |
| 4000 | 0.1834 | **0.2223** | ✅ Best (slight improvement) |
| 6000 | 0.1469 | 0.2225 | - |
| 8000 | 0.1620 | 0.2228 | - |
| 10000 | 0.0988 | 0.2228 | - |
| 12000 | 0.1494 | 0.2247 | - |
| 14000 | 0.1765 | 0.2249 | - |
| 16000 | 0.1798 | **0.2211** | ✅ Best |
| 18000 | 0.1409 | **0.2210** | ✅ Best |
| 20000 | 0.1516 | 0.2212 | Final |

### Summary Statistics
- **Train loss reduction**: 0.42 → 0.15 (64% decrease) ✅
- **Val loss reduction**: 0.44 → 0.22 (50% decrease) ✅
- **Best val loss**: 0.2210 (step 18,000)
- **Training stability**: Excellent (no spikes or divergence)

**On paper, this looks like a successful training run.** But...

---

## Actual Behavior: Mode Collapse

### Test Results

**Prompt** (correct format):
```python
def add(a, b):
    """Add two numbers"""

```

**Expected behavior**: Generate function body with `return` statement

**Actual behavior**: Generates 20+ newlines (`\n` tokens)

### Debug Analysis

**Generated token sequence**:
```
Token ID: 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, ...
Token value: \n, \n, \n, \n, \n, \n, \n, \n, \n, \n, ...
```

The model is **stuck in a loop** generating newline tokens (ID 200) exclusively.

### Comparison with Training Data

Training example format:
- Prompt ends with: `"""\n`
- Solution starts with: `    product = 1\n` (4 spaces + code)

The model should generate:
1. Indentation (spaces)
2. Python code
3. More newlines and code

Instead, it generates:
1. Newlines only
2. Forever

---

## Root Cause Analysis

### Why Did Loss Go Down?

The training loss decreased because the model learned to predict **common structural tokens** (newlines, spaces) rather than meaningful code patterns.

**Hypothesis**:
- The cross-entropy loss is dominated by easily-predictable formatting tokens
- Model optimizes for these tokens (which appear frequently and consistently)
- Actual code token prediction (harder task) is not learned properly
- LoRA's limited capacity (0.67% trainable params) may only capture superficial patterns

### Why Mode Collapse?

Three possible explanations:

1. **Weak Base Model**: The v1 pre-trained model (29M params, trained on token-level data) may lack sufficient "code understanding" to benefit from instruction tuning

2. **LoRA Capacity Limitation**: 196K trainable params (0.67%) may be insufficient to adapt the model for instruction-following while maintaining code generation quality

3. **Training Dynamics**: The model found a "local minimum" where generating structural tokens (newlines/spaces) minimizes loss without requiring complex code generation

### Similarity to v1+IT Failures

This is the **same fundamental problem** as v1+IT v2/v3, just manifesting differently:

| Version | Samples | Steps | Behavior | Loss |
|---------|---------|-------|----------|------|
| v1+IT v2 | 49 | 3,000 | No improvement | Decreased |
| v1+IT v3 | 49 | 6,000 | "return return" loop | Decreased |
| **v2 LoRA** | 500 | 20,000 | "\n\n\n..." loop | Decreased |

**Pattern**: Training loss improvements do NOT correlate with actual task performance.

---

## Comparison: v1+IT vs v2 LoRA

### v1+IT (Small-Scale, Full Fine-Tuning)
- 49 samples
- 3k-6k steps
- Full model fine-tuning (all 29M params)
- **Result**: Mode collapse ("return return") or no improvement
- **Diagnosis**: Too few samples, pre-training too dominant

### v2 LoRA (Large-Scale, Adapter Fine-Tuning)
- 500 samples (10x more)
- 20k steps (3x more)
- LoRA adapters only (196K params)
- **Result**: Mode collapse ("\n\n\n...")
- **Diagnosis**: Loss metrics misleading, model not learning meaningful patterns

### Common Failure Mode

Both approaches fail to teach the model **task-oriented code completion**. The model optimizes loss on superficial patterns (formatting tokens) rather than semantic code generation.

---

## Technical Findings

### What Worked
1. ✅ **Data pipeline**: Successfully collected & processed 550 high-quality samples
2. ✅ **LoRA integration**: PEFT library compatibility achieved (after fixes)
3. ✅ **Training stability**: No crashes, divergence, or NaN losses
4. ✅ **Loss reduction**: Both train and val loss decreased significantly

### What Failed
1. ❌ **Code generation**: Model produces only newlines
2. ❌ **Instruction following**: No evidence of task understanding
3. ❌ **Quality improvement**: 0/20 HumanEval benchmark (predicted, not tested)

### Key Insights
1. **Loss metrics are unreliable** for evaluating instruction-tuned code models
2. **Catastrophic forgetting prevention** (via LoRA) is insufficient if base model is too weak
3. **Scale alone doesn't fix fundamental issues** (500 samples >> 49, still failed)

---

## Why LoRA Didn't Help

LoRA was supposed to solve v1+IT's problems:
- ❌ **Prevent catastrophic forgetting**: Achieved (base model intact), but doesn't matter if model doesn't learn new behavior
- ❌ **Enable larger-scale training**: Achieved (500 samples, 20k steps), but model still collapses
- ❌ **Efficient adaptation**: Achieved (only 0.67% params), but adaptation is to wrong patterns

**Conclusion**: LoRA prevents catastrophic forgetting of the **base model's patterns**, but the base model's patterns are **token-level language modeling**, not **instruction-following code completion**. Adapting this base is fundamentally difficult.

---

## Next Steps: What Actually Works?

Given three failed attempts (v1+IT v2, v1+IT v3, v2 LoRA), the evidence suggests:

### ❌ **Don't Try Again**:
1. More instruction tuning samples (1000+)
2. Different LoRA hyperparameters (higher rank, different modules)
3. Longer training (50k+ steps)

**Reason**: The problem is not scale or hyperparameters. It's the base model architecture and training objective mismatch.

### ✅ **Fundamental Changes Needed**:

#### Option A: Pre-train on Code Tasks (Not Just Code Text)
- Train v1 with **task-aware objective** (e.g., predict function bodies given signatures)
- This requires re-doing v1 pre-training entirely
- **Estimated effort**: 2-3 days

#### Option B: Bigger Model
- Scale to 50M-100M params
- Hypothesis: larger capacity enables both LM patterns AND instruction-following
- **Estimated effort**: 3-5 days

#### Option C: Distillation from Strong Teacher
- Use GPT-4 / Claude to generate completions for training data
- Train v1 to imitate strong teacher outputs
- **Estimated effort**: 5-7 days (data collection bottleneck)

#### Option D: Multi-Stage Training
- Stage 1: Pre-train on code (current v1)
- Stage 2: Continue pre-training on paired (signature, body) data
- Stage 3: Instruction tuning with LoRA
- **Estimated effort**: 4-6 days

---

## Conclusion

**v2 LoRA training is a failure**, despite promising loss curves. The model exhibits mode collapse (newline generation loop) and does not perform meaningful code completion.

**Key Takeaway**: For small code LMs (~30M params), **instruction tuning is extremely difficult** regardless of approach (full fine-tuning vs LoRA) or scale (49 vs 500 samples). The base model must be designed with instruction-following in mind from the start.

**Recommendation**: Either:
1. Accept v1 as a pure LM (no instruction-following) and document its capabilities
2. Commit to fundamental architectural changes (larger model, task-aware pre-training, or distillation)

Incremental improvements to the current v1 + instruction tuning pipeline are unlikely to succeed.

---

## Files & Artifacts

### Training
- Script: `nas/train_v2_lora.py`
- Checkpoints: `logs/train_v2_lora/v2_lora_step{2000,4000,...,20000}_lora/`
- Best model: `logs/train_v2_lora/v2_lora_best_lora/` (step 18,000, val_loss=0.2210)

### Data
- Train: `data/instruction_tuning/v2_train.jsonl` (500 samples)
- Val: `data/instruction_tuning/v2_val.jsonl` (50 samples)
- Tools: `nas/tools/{download_datasets.py, convert_humaneval_to_supervised.py, convert_mbpp_to_supervised.py, merge_and_split_data.py}`

### Testing
- Test script: `nas/test_v2_lora_inference.py`
- Debug script: `nas/debug_generation.py`

---

**Status**: v2 line **CLOSED**. Instruction tuning approach exhausted for current base model.
