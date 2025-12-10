# BigData Training Progress

**Date**: 2025-12-08
**Goal**: データサイズ拡大によるモード崩壊の解決
**Status**: Phase 1 (小規模テスト) 実行中

---

## 📊 Current Status

### Phase 1: Small-scale Test (7.92MB corpus)

**データ収集** ✅ COMPLETED
- リポジトリ: requests, flask, pytest, boto3, fastapi
- ファイル数: 1,535 Python files
- データサイズ: **7.92 MB** (元の1.3MBから **6倍増**)
  - Train: 7.79 MB (1,520 files, 246,144 lines)
  - Val: 0.13 MB (15 files, 3,630 lines)

**トレーニング** 🔄 RUNNING
```bash
Experiment: v1_bigdata_test
Steps: 5,000 (元の10K stepの半分、様子見)
Dataset: code_char_big (7.92 MB)
Device: cuda:0
Log: logs/train_v1_bigdata_test/
```

**期待される結果**:
- 元データ(1.3MB): Mode collapse 85.2%, Keywords 0%
- 6倍データ(7.92MB): Mode collapse <70%?, Keywords >5%?
- データが小さすぎるので劇的改善は期待しないが、**傾向**を見る

---

## 🛠️ Infrastructure Ready

### ツール完成 ✅

| Tool | Status | Purpose |
|------|--------|---------|
| `scripts/prepare_python_corpus.py` | ✅ | 大規模コーパス生成 |
| `data/DATA_COLLECTION_GUIDE.md` | ✅ | データ収集ガイド |
| `nas/EXPERIMENTS.md` Section 12 | ✅ | BigData実験手順書 |
| `eval/` pipeline | ✅ | バッチ評価・解析 |

### ワークフロー確立 ✅

```bash
# 1. データ収集
cd data/raw_python
git clone <repos>

# 2. コーパス生成
cd ../../nas
python scripts/prepare_python_corpus.py \
  --src_dir ../data/raw_python \
  --train_out ../data/code_char_big/train.txt \
  --val_out ../data/code_char_big/val.txt

# 3. トレーニング
python train_best.py \
  --arch_json models/codenas_best_current.json \
  --experiment_name v1_bigdata_char \
  --train_path ../data/code_char_big/train.txt \
  --max_steps 100000

# 4. 評価
python eval_playground.py \
  --checkpoint logs/train_v1_bigdata_char/*.pt \
  --eval_file eval/prompts/simple_python.txt \
  --output eval/results_bigdata.jsonl

python eval/inspect_results.py eval/results_bigdata.jsonl
```

---

## 📈 Next Steps

### Immediate (Phase 1 完了後)

1. **Phase 1 結果を確認** (~5-10分後)
   ```bash
   cd nas
   python eval_playground.py \
     --checkpoint logs/train_v1_bigdata_test/v1_bigdata_test_best.pt \
     --eval_file eval/prompts/simple_python.txt \
     --output eval/results_bigdata_test.jsonl

   python eval/inspect_results.py eval/results_bigdata_test.jsonl --show_quality_examples
   ```

2. **結果を分析**
   - Mode collapse率の変化
   - Python keywords出現率
   - 生成品質の主観評価

3. **Phase 2 を決定**:
   - ✅ **改善あり** → さらにデータを集めて50-100MBへ
   - ⚠️ **改善不十分** → Token-level modeling へ切り替え検討

### Phase 2: Large-scale Training (50-100MB)

**データ収集オプション**:

| Option | Size | Effort | Time |
|--------|------|--------|------|
| **A: GitHub repos (15-20個)** | 50-100MB | Low | 10-30分 |
| **B: The Stack dataset** | 100MB-1GB | Medium | 1-2時間 |
| **C: CodeSearchNet** | 500MB | High | 2-3時間 |

**推奨**: まずOption A（GitHub repos追加）で50-100MBを目指す

**追加候補リポジトリ**:
```bash
cd data/raw_python

# ML/Data Science系 (大きめ)
git clone --depth 1 https://github.com/scikit-learn/scikit-learn.git  # ~30MB
git clone --depth 1 https://github.com/pandas-dev/pandas.git           # ~40MB
git clone --depth 1 https://github.com/numpy/numpy.git                 # ~30MB

# Web frameworks
git clone --depth 1 https://github.com/django/django.git               # ~15MB
git clone --depth 1 https://github.com/tornadoweb/tornado.git          # ~2MB
git clone --depth 1 https://github.com/aio-libs/aiohttp.git            # ~5MB

# Tools
git clone --depth 1 https://github.com/pypa/pip.git                    # ~5MB
git clone --depth 1 https://github.com/python-poetry/poetry.git        # ~3MB
git clone --depth 1 https://github.com/celery/celery.git               # ~5MB

# Testing
git clone --depth 1 https://github.com/robotframework/robotframework.git
git clone --depth 1 https://github.com/tox-dev/tox.git

# 合計: 50-100MB達成可能
```

### Phase 3: Full-scale (100K steps, 100MB+ data)

```bash
cd nas

python train_best.py \
  --arch_json models/codenas_best_current.json \
  --experiment_name v1_bigdata_production \
  --train_path ../data/code_char_big/train.txt \
  --val_path ../data/code_char_big/val.txt \
  --max_steps 100000 \
  --log_dir logs/train_v1_bigdata_production \
  --device cuda:0

# 期待学習時間: 1-2時間 (100MB data, 100K steps)
```

---

## 🎯 Success Criteria

| Metric | Baseline (1.3MB) | Target (100MB+) | Phase 1 (7.92MB) |
|--------|------------------|-----------------|------------------|
| Mode collapse率 | 85.2% | <30% | TBD |
| Keywords出現率 | 0% | >50% | TBD |
| Repetition比率 | 92.22% | <30% | TBD |
| Unique chars (avg) | 2.1 | >20 | TBD |

---

## 📝 Lessons Learned

### Infrastructure Phase (完了)

1. **コーパス生成スクリプト** が正常動作
   - 1,535 files処理 (数秒)
   - エンコーディングエラーハンドリング ✓
   - 除外パターンフィルタ ✓

2. **既存パイプライン** がそのまま使える
   - train_best.py は新データセットで追加変更不要
   - eval_playground.py もそのまま再利用可能
   - 評価ツール (inspect_results.py) も流用

3. **Git cloneアプローチ** が簡単・高品質
   - 5リポジトリで 7.92MB（10-15分）
   - 15-20リポジトリで 50-100MB見込み

### 次のフェーズで検証すること

1. **データサイズ vs 品質の関係**
   - 7.92MB → どの程度改善するか？
   - 50MB → 劇的改善の閾値？
   - 100MB → 十分なのか、それとも1GB必要？

2. **Char-level の限界**
   - 100MBでもダメなら → Token-level へ
   - Token-level なら 10-20MB で同等品質の可能性

3. **Training steps の最適値**
   - 10K steps (現状) → 不足？
   - 100K steps → 過剰？
   - Early stopping が適切に機能するか？

---

## 🔗 References

- [EXPERIMENTS.md Section 12](nas/EXPERIMENTS.md#12-bigdata-training-データ拡張実験)
- [DATA_COLLECTION_GUIDE.md](data/DATA_COLLECTION_GUIDE.md)
- [EVALUATION_SUMMARY.md](nas/eval/EVALUATION_SUMMARY.md)

---

**Last Updated**: 2025-12-08
**Status**: Phase 1 training running, ~5min ETA
