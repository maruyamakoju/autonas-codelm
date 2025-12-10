# v1.0 Project Completion Notes

**Date**: 2025-12-10
**Status**: ✅ Complete (Phase 3 finished)

## What Was Accomplished

Successfully built a **29M parameter CodeLM (v1.0-strongreg)** with 8K BPE tokenization that:
- Reduced mode collapse from **85%+ → 14.5%** (with few-shot context)
- Generates syntactically valid Python code
- Trained on 100MB Python corpus (~5.4 hours on RTX 5090)

**Key Finding**: Mode collapse is **solvable** through strong regularization (label_smoothing=0.1 was critical). However, next-token prediction alone produces a "syntax generator" rather than a task solver (0/20 on mini benchmark).

## What Was Learned

1. **Label smoothing (0.1)** prevents overconfident predictions that cause repetition patterns
2. **Context length matters**: Short prompts collapse ~90%, few-shot (1-2 examples) reduces to ~15%
3. **Model capacity threshold**: 29M params + strong regularization was the first to succeed (smaller models failed)
4. **Task-solving ≠ syntax generation**: 29M params can learn Python patterns but not problem-solving logic

## Release Artifacts

- **Git tag**: `v1.0-strongreg` (commit e2f5156)
- **Checkpoint**: `nas/logs/train_v1_8k_strongreg/v1_8k_strongreg_best.pt`
- **Tokenizer**: `data/tokenizers/python_bpe_8k/tokenizer.json`
- **Documentation**: `STRONGREG_SUMMARY.md` (complete training & evaluation results)
- **Demo**: `nas/demo_v1.py` (few-shot generation script)
- **Benchmark**: `nas/eval/benchmarks/mini_humaneval.jsonl` (20 tasks)

## If Resuming This Project

**Recommended next step** (small experiment, high value):

1. Create 10-50 task-solution pairs (JSONL format)
   - Use mini_humaneval.jsonl as a template
   - Add correct solutions as "solution" field

2. Fine-tune v1 for 1k-5k steps on this data
   - Start from v1_8k_strongreg_best.pt
   - Use same hyperparameters but lower learning rate (1e-4)

3. Re-run mini benchmark
   - Goal: 0/20 → 3-5/20 would confirm instruction tuning works
   - If successful, scale to 50-200 examples

**If that's insufficient** → v2 model (50M+ params, longer context, full instruction dataset)

## What NOT to Do

- ❌ Don't train from scratch again without instruction tuning
- ❌ Don't try to "fix" mode collapse further (it's already solved with few-shot)
- ❌ Don't make architectural changes without clear hypothesis

## Project Value

This project successfully demonstrated:
- Multi-objective NAS for code models works
- Mode collapse in small LMs is a regularization problem
- Few-shot context dramatically changes model behavior
- Clear quantification of limitations (0/20 benchmark)

**Result**: A well-documented research prototype showing both capabilities and limitations.

---

## Post-v1: Instruction Tuning Attempt (2025-12-10)

**Tried the recommended small-scale instruction tuning** (49 samples, multiple training runs):

- **v1+IT v2**: 3k steps, LR=5e-5, dynamic collate fix → 0/20 (no improvement)
- **v1+IT v3**: 6k steps, LR=1e-4 (aggressive) → 0/20 (introduced mode collapse "return return")
- **Result**: Small-scale instruction tuning (<100 samples) **completely failed** to improve task-solving

**Root cause**: 50k-step pre-training too dominant for 49-sample fine-tuning to override, regardless of LR/steps.

**Conclusion**: "Cheap levers" (small data, hyperparameter tuning) exhausted. Next improvement requires:
- **Large-scale data** (500-1000+ samples from HumanEval/MBPP)
- **Parameter-efficient fine-tuning** (LoRA to prevent catastrophic forgetting)
- **Bigger model** (50M-100M params) or **distillation** from stronger teacher

**See**: `INSTRUCTION_TUNING_FAILURE_REPORT.md` for full analysis.

---

---

## Summary: What This Project Achieved

**v1 line (29M, 8K BPE)**: Solved mode collapse through regularization, completed as syntax generator research artifact.

**Instruction tuning experiments**: Exhaustively tested small-scale approaches (v1+IT v2, v3) and confirmed failure. All "cheap levers" (hyperparameters, small data) have been attempted.

**v2 planning**: Designed 4 options for quality scaling. Recommended: LoRA + 500-1000 samples (see `V2_PLAN.md` and `DATA_EXPANSION_GUIDE.md`).

**Next action**:
- If continuing: Follow v2 Option A (LoRA + HumanEval/MBPP data)
- If pausing: Project is fully documented, backed up, and ready for archival

**GitHub Repository**: https://github.com/maruyamakoju/autonas-codelm

---

**Final Status**: v1 complete and closed. Instruction tuning with small data confirmed unfeasible. Next step requires v2 approach or project archival.
