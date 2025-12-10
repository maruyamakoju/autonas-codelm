# v2 Plan: Scaling Up Task-Solving Quality

**Date**: 2025-12-10
**Context**: v1 (29M, 0/20) exhausted "cheap levers" (small data, hyperparameter tuning). Next step requires **big levers**.

---

## 🎯 Goals

### Primary Target
**HumanEval mini: 5-10/20 tasks correct** (25-50% pass rate)

Specifically solve these categories reliably:
- ✅ Simple arithmetic (`add`, `multiply`, `subtract`)
- ✅ Basic list operations (`sum_list`, `reverse_list`, `filter_even`)
- ✅ Simple conditionals (`is_even`, `is_positive`, `max_value`)
- 🎯 Slightly complex logic (`is_prime`, `factorial`, `gcd`)

### Success Criteria
- **0 mode collapse** (no "return return", "def def" patterns)
- **Syntax validity** >95% (executable Python)
- **Logic correctness** 25-50% (passes test cases)

---

## 🛠️ Approach Options (Ranked by Feasibility)

### **Option A: Large-Scale Instruction Tuning + LoRA** (Recommended First Try)

**Core idea**: Use v1 architecture but fine-tune on 500-1000 task-solution pairs with LoRA to prevent catastrophic forgetting.

**Why this first**:
- ✅ Keeps v1 architecture (proven to work for syntax)
- ✅ LoRA prevents degrading pre-trained weights
- ✅ 500-1000 samples feasible to create/curate
- ✅ Can reuse v1 training infrastructure

**Implementation**:
```
1. Data Collection (500-1000 samples)
   - HumanEval: 164 tasks (use all)
   - MBPP: ~1000 tasks (filter for simple ones)
   - Synthetic generation: GPT-4 to generate 500+ simple tasks

2. Model Setup
   - Init: v1_8k_strongreg_best.pt (50k steps)
   - Architecture: Add LoRA adapters (rank=8, alpha=16)
   - Freeze base weights, only train LoRA layers

3. Training
   - Steps: 10k-20k (longer than v1+IT)
   - LR: 1e-4 for LoRA layers (higher OK since base is frozen)
   - Batch size: 8 (fit in 4090)
   - Warmup: 500 steps

4. Evaluation
   - Mini HumanEval every 2k steps
   - Track: pass rate, mode collapse, syntax errors
```

**Expected timeline**: 2-3 days (1 day data prep, 1-2 days training)

**Risk**: LoRA may still fail if base model fundamentally can't learn task logic

---

### **Option B: Bigger Model (50M-100M params)**

**Core idea**: 29M params may be insufficient for task-solving. Scale up to 50-100M.

**Why this works**:
- Research shows task-solving emerges around 50-100M params
- More capacity for memorizing patterns beyond syntax
- Can use same training recipe as v1

**Implementation**:
```
Architecture options:
- L16H768 (88M params) - 2× depth of v1
- L12H1024 (105M params) - wider than v1
- L20H512 (52M params) - deeper but narrow

Training:
- Corpus: Same 100MB Python (or expand to 500MB)
- Steps: 50k (same as v1)
- Regularization: label_smoothing=0.1 (proven)
- Then: Instruction tuning on 500+ samples
```

**Expected timeline**: 5-7 days (2-3 days pre-training, 2-3 days instruction tuning, 1 day eval)

**Risk**: May still get 0/20 if data quality is the bottleneck, not model size

---

### **Option C: Distillation from GPT-4**

**Core idea**: Generate 10k-100k synthetic task-solution pairs using GPT-4 as teacher.

**Why this is powerful**:
- High-quality, diverse training data
- Can control task difficulty (start simple, gradually increase)
- Teacher model already solves tasks correctly

**Implementation**:
```
1. Data Generation (via GPT-4 API)
   - Prompt: "Generate 10,000 simple Python programming tasks with docstrings and solutions"
   - Filter: Only keep functions <50 lines, with clear test cases
   - Validate: Run tests to ensure correctness

2. Training
   - Init: v1 or train from scratch with distillation loss
   - Mix pre-training + instruction tuning (curriculum learning)

3. Curriculum
   - Phase 1 (10k steps): Train on "Level 1" tasks (add, multiply)
   - Phase 2 (10k steps): Train on "Level 2" tasks (loops, basic logic)
   - Phase 3 (10k steps): Train on "Level 3" tasks (recursion, algorithms)
```

**Expected timeline**: 7-14 days (3-5 days data generation, 3-5 days training, 1-2 days eval)

**Cost**: GPT-4 API for 10k generations ~$50-100

**Risk**: Expensive, time-consuming, may still fail if 29M is too small

---

### **Option D: Hybrid Approach (Best Long-Term)**

**Core idea**: Combine A + B (bigger model + LoRA + large data)

```
1. Train 50M-100M model on 100MB corpus (like v1)
2. Add LoRA adapters
3. Fine-tune on 1000+ samples from HumanEval/MBPP
4. Optionally: Add GPT-4 synthetic data if needed
```

**Expected timeline**: 10-14 days

**Why this is "best"**: Addresses all failure modes (capacity, data, catastrophic forgetting)

---

## 📊 Resource Planning

### Hardware Available
- RTX 5090 (24GB VRAM) - primary training
- RTX 4090 (24GB VRAM) - secondary/eval

### Time Budget for Each Option

| Option | Data Prep | Training | Eval | Total |
|--------|-----------|----------|------|-------|
| A (LoRA) | 1 day | 1-2 days | 0.5 day | 2-3 days |
| B (Bigger) | 0.5 day | 4-5 days | 0.5 day | 5-6 days |
| C (Distill) | 3-5 days | 3-5 days | 1 day | 7-11 days |
| D (Hybrid) | 3-5 days | 5-7 days | 1 day | 9-13 days |

---

## 🚀 Recommended First Step

**Start with Option A** (LoRA + 500 samples) because:
1. Lowest time investment (2-3 days)
2. Directly tests "is small data the problem?"
3. If it fails → we know we need bigger model (Option B/D)
4. If it succeeds → fast path to 5-10/20

### Concrete Action Plan for Option A

**Week 1: Data Collection**
```
Day 1: Data collection
- Download HumanEval dataset (164 tasks)
- Download MBPP dataset (~1000 tasks)
- Filter for "simple" tasks (<=20 lines of code)
- Target: 500 task-solution pairs

Day 2: Data formatting
- Convert to JSONL format: {"id": ..., "prompt": ..., "solution": ...}
- Split: 450 train, 50 validation
- Verify: All solutions pass test cases
```

**Week 1-2: LoRA Implementation & Training**
```
Day 3: LoRA setup
- Install peft library: pip install peft
- Modify train_instruction_tuning.py to add LoRA adapters
- Config: rank=8, alpha=16, target_modules=["qkv_proj", "out_proj"]

Day 4-5: Training
- Run: python train_instruction_tuning_lora.py \
    --init_checkpoint logs/train_v1_8k_strongreg/v1_8k_strongreg_best.pt \
    --data_path data/instruction_tuning/humaneval_mbpp_500.jsonl \
    --max_steps 20000 \
    --lr 1e-4 \
    --batch_size 8 \
    --eval_interval 2000
- Monitor: Loss curve, sample generations every 2k steps

Day 6: Evaluation
- Run mini benchmark every 2k steps
- Compare: v1 baseline (0/20) vs v1+LoRA (goal: 5+/20)
- Analyze: Which task types succeed/fail
```

**Decision point after Week 1-2**:
- ✅ If 5+/20: SUCCESS! Document, publish, consider scaling to full HumanEval
- ⚠️ If 1-4/20: Partial success. Increase data to 1000 samples or try Option B
- ❌ If 0/20: Data not the bottleneck. Need bigger model (Option B/D)

---

## 📝 Data Expansion Plan (for v2)

If we go with Option A or C, we need to expand beyond 49 samples. Here's the task category breakdown:

### Target: 500-1000 Samples

**Category 1: Arithmetic & Math (50 samples)**
- Basic ops: add, subtract, multiply, divide, modulo, power
- Comparisons: max, min, abs, sign
- Math functions: gcd, lcm, factorial, fibonacci, prime check

**Category 2: List Operations (100 samples)**
- Access: first, last, nth element
- Aggregation: sum, product, count, length, average
- Filtering: filter positive/negative/even/odd, remove duplicates
- Transformation: double, square, map operations
- Search: linear search, binary search, find max/min

**Category 3: String Operations (100 samples)**
- Basic: length, reverse, uppercase, lowercase
- Checking: starts_with, ends_with, contains, is_palindrome
- Transformation: repeat, trim, split, join
- Counting: count vowels, count words, count char

**Category 4: Algorithms (100 samples)**
- Sorting: bubble sort, selection sort, insertion sort
- Searching: binary search, linear search
- Two pointers: merge sorted arrays, remove duplicates
- Recursion: factorial, fibonacci, power, tree traversal

**Category 5: Data Structures (50 samples)**
- Arrays: flatten, reshape, rotate
- Stacks: push, pop, peek, is_empty
- Queues: enqueue, dequeue
- Linked lists: insert, delete, reverse

**Category 6: Logic & Conditionals (50 samples)**
- Boolean ops: all_positive, any_negative, all_equal
- Conditional logic: if-elif-else chains
- Early returns: multiple exit points
- Edge cases: empty lists, None handling

**Category 7: Loops & Iteration (50 samples)**
- For loops: range iteration, enumerate
- While loops: until condition
- Nested loops: 2D array operations
- Loop control: break, continue

**Total: 500 samples (minimum viable)**

**Scaling to 1000**: Double each category OR add:
- Category 8: File I/O (50)
- Category 9: Error handling (50)
- Category 10: Object-oriented (50)
- More complex variants of existing categories (300)

---

## 🔬 Experiments Tracking

### Completed (v1 era)
- ✅ v1.0-strongreg: 29M, 0/20, mode collapse solved
- ✅ v1+IT (49 samples): 0/20, complete failure

### Planned (v2 era)
- 🎯 **v2.0-lora-500**: v1 arch + LoRA + 500 samples (Option A)
- 🎯 **v2.1-50m-baseline**: 50M model, pre-training only
- 🎯 **v2.2-distill-gpt4**: Distillation from GPT-4 on 10k samples

---

## 💡 Key Insights to Remember

1. **Pre-training dominance is real**: 50k steps overwhelms <100 sample fine-tuning
2. **LoRA is essential**: Prevents catastrophic forgetting when fine-tuning small models
3. **Data quality > quantity**: 500 good samples better than 5k noisy ones
4. **Benchmark early, benchmark often**: Eval every 2k steps to catch overfitting
5. **Mode collapse can return**: Monitor for "return return", "def def" patterns

---

## ❌ What NOT to Do (Learned from v1)

- ❌ Don't fine-tune without LoRA on small datasets (<1000 samples)
- ❌ Don't use LR > 1e-4 for fine-tuning (causes instability)
- ❌ Don't trust loss=0.0000 (verify with actual generation tests)
- ❌ Don't use static padding (causes overfit to padding tokens)
- ❌ Don't expect miracles from <100 samples

---

## 🎬 Next Action

**Immediate (today/tomorrow)**:
1. Decide: Option A (LoRA) or Option B (bigger model)?
2. If Option A: Start data collection (HumanEval + MBPP download)
3. If Option B: Design 50M architecture (L16H768 or L12H1024)

**This week**:
- Set up data pipeline (JSONL creation, validation)
- Implement LoRA wrappers (if Option A)
- Start first training run

**Next week**:
- Monitor training, generate samples at checkpoints
- Run mini benchmark evaluations
- Decide: continue scaling or pivot strategy

---

**Status**: v2 plan ready. Awaiting execution decision.
