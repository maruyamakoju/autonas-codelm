# Python Corpus Data Collection Guide

**目的**: 100MB-1GB規模のPythonコードを収集して、char-level言語モデルのトレーニングに使う

---

## 🎯 収集目標

| 規模 | サイズ | 期待効果 |
|------|--------|---------|
| **最小** | 50MB | モード崩壊が部分的に改善 |
| **推奨** | 100-500MB | 大幅な品質向上、coherentなコード生成 |
| **理想** | 1GB+ | GPT-2 smallレベルの品質 |

**現状**: 1.3MB (data/code_char/train.txt) → **100倍以上**のデータが必要

---

## 📦 Option A: The Stack Dataset (推奨)

### 概要
- **提供元**: HuggingFace BigCode
- **URL**: https://huggingface.co/datasets/bigcode/the-stack
- **サイズ**: Python部分で数GB-数十GB
- **品質**: 高品質、deduplicated、license-filtered

### 方法1: HuggingFace Datasets ライブラリ (推奨)

```bash
# HuggingFace datasetsをインストール
pip install datasets

# Pythonスクリプトでダウンロード
cd data
python download_the_stack.py
```

**`download_the_stack.py` の内容:**

```python
from datasets import load_dataset
from pathlib import Path

# The Stack の Python サブセットをロード (100MB サンプル)
print("Loading The Stack (Python)...")
ds = load_dataset(
    "bigcode/the-stack",
    data_dir="data/python",
    split="train",
    streaming=True  # ストリーミングモードで大きなデータセットを扱う
)

# 100MB分のPythonファイルを保存
output_dir = Path("raw_python/the_stack")
output_dir.mkdir(parents=True, exist_ok=True)

total_bytes = 0
target_bytes = 100 * 1024 * 1024  # 100MB
file_count = 0

for example in ds:
    content = example['content']

    # ファイルに書き出し
    file_path = output_dir / f"sample_{file_count:06d}.py"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    total_bytes += len(content.encode('utf-8'))
    file_count += 1

    if file_count % 100 == 0:
        print(f"Collected {file_count} files, {total_bytes / 1024 / 1024:.1f} MB", end='\r')

    # 目標サイズに達したら終了
    if total_bytes >= target_bytes:
        break

print(f"\nOK Downloaded {file_count} files, {total_bytes / 1024 / 1024:.2f} MB")
```

**実行:**

```bash
cd data
python download_the_stack.py

# 出力: raw_python/the_stack/*.py (100MB分)
```

### 方法2: 手動ダウンロード

```bash
# HuggingFace CLI でダウンロード
pip install huggingface-hub

# ログイン (初回のみ)
huggingface-cli login

# The Stack Python サブセットをダウンロード
huggingface-cli download bigcode/the-stack --repo-type dataset --include "data/python/*.parquet" --local-dir data/raw_python/the_stack_parquet

# Parquet を .py ファイルに変換 (別途スクリプト必要)
```

---

## 📦 Option B: GitHub 人気リポジトリ (簡単)

### 概要
- **提供元**: GitHub
- **サイズ**: リポジトリごとに数MB-数十MB
- **品質**: 高品質、実際のプロダクションコード
- **推奨リポジトリ**: Python界の有名ライブラリ

### ダウンロード手順

```bash
cd data/raw_python

# Python標準ライブラリ風の人気プロジェクト
git clone https://github.com/psf/requests.git              # ~5MB
git clone https://github.com/pallets/flask.git             # ~3MB
git clone https://github.com/django/django.git             # ~15MB
git clone https://github.com/pytest-dev/pytest.git         # ~5MB
git clone https://github.com/numpy/numpy.git               # ~30MB (C含む)
git clone https://github.com/pandas-dev/pandas.git         # ~40MB
git clone https://github.com/scikit-learn/scikit-learn.git # ~30MB
git clone https://github.com/fastapi/fastapi.git           # ~3MB
git clone https://github.com/tornadoweb/tornado.git        # ~2MB
git clone https://github.com/boto/boto3.git                # ~5MB

# より小規模のライブラリ
git clone https://github.com/kennethreitz/records.git
git clone https://github.com/jazzband/pip-tools.git
git clone https://github.com/pypa/pip.git
git clone https://github.com/pytestarch/pytestarch.git

# 合計: 100MB+ 達成可能
```

**注意点:**
- `.git/` ディレクトリは不要 (削除してOK)
- テストコード、ドキュメントも含まれる (むしろ多様性が増えて良い)
- C拡張やJSが混ざっているリポジトリもあるが、`prepare_python_corpus.py` が `.py` だけフィルタする

### .gitディレクトリ削除 (容量節約)

```bash
cd data/raw_python
find . -name ".git" -type d -exec rm -rf {} +

# または Windows の場合:
# PowerShell で各ディレクトリの .git を削除
Get-ChildItem -Path . -Recurse -Directory -Filter ".git" | Remove-Item -Recurse -Force
```

---

## 📦 Option C: CodeSearchNet Dataset

### 概要
- **提供元**: GitHub + Microsoft Research
- **URL**: https://github.com/github/CodeSearchNet
- **サイズ**: Python部分で ~500MB
- **品質**: GitHubから収集、docstring付きコード多め

### ダウンロード手順

```bash
# CodeSearchNet の Python サブセットをダウンロード
cd data
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip

# 解凍
unzip python.zip -d raw_python/codesearchnet

# JSONLからPythonコードを抽出 (別途スクリプト必要)
```

**JSONLからコード抽出スクリプト** (`extract_codesearchnet.py`):

```python
import json
from pathlib import Path

input_dir = Path("raw_python/codesearchnet/python/final/jsonl/train")
output_dir = Path("raw_python/codesearchnet_extracted")
output_dir.mkdir(parents=True, exist_ok=True)

file_count = 0
for jsonl_file in input_dir.glob("*.jsonl.gz"):
    import gzip
    with gzip.open(jsonl_file, 'rt', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            code = entry.get('code', '')

            if code.strip():
                file_path = output_dir / f"sample_{file_count:06d}.py"
                with open(file_path, 'w', encoding='utf-8') as out:
                    out.write(code)
                file_count += 1

                if file_count % 1000 == 0:
                    print(f"Extracted {file_count} files...", end='\r')

print(f"\nOK Extracted {file_count} Python code samples")
```

---

## 🔄 データ前処理フロー

### 1. データ収集完了後の確認

```bash
cd data/raw_python

# Pythonファイル数を確認
find . -name "*.py" | wc -l

# 合計サイズを確認 (Windows)
dir /s *.py | find "File(s)"

# 合計サイズを確認 (Unix)
find . -name "*.py" -exec du -ch {} + | grep total$
```

### 2. コーパス生成

```bash
cd nas

# raw_python → code_char_big に変換
python scripts/prepare_python_corpus.py \
  --src_dir ../data/raw_python \
  --train_out ../data/code_char_big/train.txt \
  --val_out ../data/code_char_big/val.txt \
  --val_ratio 0.01

# 結果確認
ls -lh ../data/code_char_big/
# 期待:
#   train.txt: 100MB-1GB
#   val.txt: 1MB-10MB
```

### 3. サイズ確認

```bash
cd ../data/code_char_big

# ファイルサイズ
du -h train.txt val.txt

# 行数
wc -l train.txt val.txt

# 文字数
wc -m train.txt val.txt
```

**期待される出力:**

```
train.txt: 123.45 MB, 1,234,567 lines
val.txt:   1.23 MB, 12,345 lines
```

---

## 📊 データ品質チェック

### Python コードの sanity check

```python
# Check if train.txt contains valid Python patterns
import re

with open('../data/code_char_big/train.txt', 'r', encoding='utf-8') as f:
    sample = f.read(10000)  # 最初の10KB

# 期待されるPythonパターン
patterns = [
    r'\bdef\s+\w+',       # 関数定義
    r'\bclass\s+\w+',     # クラス定義
    r'\bimport\s+\w+',    # import文
    r'\bfor\s+\w+\s+in',  # forループ
    r'\bif\s+.+:',        # if文
]

print("=== Data Quality Check ===")
for pattern in patterns:
    matches = len(re.findall(pattern, sample))
    print(f"{pattern:30s}: {matches:3d} matches")

# 期待: 各パターンが数回以上出現
```

---

## 🚀 次のステップ

データ収集が完了したら → [EXPERIMENTS.md Section 12](../nas/EXPERIMENTS.md#12-bigdata-training-データ拡張実験) に従って学習開始

```bash
cd nas

python train_best.py \
  --arch_json models/codenas_best_current.json \
  --experiment_name v1_bigdata_char \
  --train_path ../data/code_char_big/train.txt \
  --val_path ../data/code_char_big/val.txt \
  --max_steps 100000 \
  --log_dir logs/train_v1_bigdata_char \
  --device cuda:0
```

---

## 💡 Tips

### データ収集の優先順位

1. **最初**: Option B (GitHub repos) で 50-100MB 集めてテスト
   - 簡単、速い (git clone のみ)
   - 品質が高い
   - すぐに学習開始できる

2. **次**: 効果があれば Option A (The Stack) で 500MB-1GB
   - より大規模、多様性高い
   - ダウンロードに時間かかるが品質は保証されている

3. **オプション**: CodeSearchNet は docstring が豊富
   - 関数/クラスの説明文が多い
   - Code-to-Text タスクに向いている

### ディスク容量の目安

- raw_python (生データ): 200MB-2GB
- code_char_big (変換後): 100MB-1GB
- ログ・チェックポイント: 100MB-500MB

**合計**: 500MB-4GB のディスク容量を確保

---

**Last Updated**: 2024-12-08
