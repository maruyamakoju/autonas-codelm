# トラブルシューティング

## OpenCUA バージョン不整合エラー

### 症状
```
AttributeError: 'OpenCUAForConditionalGeneration' object has no attribute '_supports_sdpa'
```

### 原因
transformers, torch, pillow などのバージョンが新しすぎて、OpenCUA のカスタムモデルと噛み合っていない。

### 解決方法

#### 方法1: バッチファイルを使う（推奨）
```bash
fix_dependencies.bat
```

#### 方法2: 手動でインストール
```bash
.venv\Scripts\activate

pip install ^
  "transformers==4.53.0" ^
  "torch==2.8.0" ^
  "pillow==11.3.0" ^
  "tiktoken==0.11.0" ^
  "blobfile==3.0.0" ^
  "accelerate==1.10.0"

pip install mss pyautogui pyperclip pyyaml
```

### 確認
```bash
python -m pip show transformers torch pillow tiktoken accelerate
```

transformers が 4.53.0、torch が 2.8.0 になっていればOK。

---

## CUDA available: False

### 症状
```
CUDA available: False
```

### 原因
- GPU が認識されていない
- CUDA ドライバが未インストール
- PyTorch のCUDAバージョンが不一致

### 確認
```bash
nvidia-smi  # GPUが見えるか確認
```

### 解決方法

#### GPUが見えない場合
- NVIDIA ドライバを最新版にアップデート
- Windows デバイスマネージャーで GPU を確認

#### GPUは見えるが PyTorch が認識しない場合
PyTorch 2.8.0 + CUDA 11.8/12.1 の組み合わせが必要です。

**CUDA 11.8 版:**
```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1 版:**
```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu121
```

### CPUモードで動作確認する場合
- `config.yaml` で `model_id: "xlangai/OpenCUA-7B"` に設定（32Bより軽い）
- `dry_run: true` でまず動作確認
- GPU が必須ではないが、推論は遅くなる

---

## メモリ不足エラー

### 症状
```
RuntimeError: CUDA out of memory
```

### 原因
- GPU VRAMが不足（32Bモデルは大きい）
- 他のプロセスがVRAMを使用中

### 解決方法

#### 1. 7Bモデルを使う
`config.yaml`:
```yaml
model:
  model_id: "xlangai/OpenCUA-7B"
```

#### 2. 他のGPUプロセスを終了
```bash
nvidia-smi
# 不要なプロセスを kill
```

#### 3. device_map="auto" を利用（既に設定済み）
CPUとGPUで自動分散してくれる。

---

## クリック位置がずれる

### 症状
- ボタンをクリックしているが、微妙に外している
- 全く違う場所をクリックしている

### 原因
- Windows のスケーリング設定（125%, 150%など）
- マルチモニタ環境
- クロップ領域の設定ミス

### 解決方法

#### 1. スケーリングを100%に設定
1. Windows設定 → ディスプレイ
2. 拡大縮小とレイアウト → 100%

#### 2. プライマリモニタに Claude Code を配置
マルチモニタの場合、Claude Code を プライマリモニタ（モニタ1）に配置。

#### 3. screen.png で確認
`screen.png` を開いて、キャプチャが正しいか確認。

#### 4. クロップ領域を調整
ブラウザだけをキャプチャする場合、`config.yaml`:
```yaml
capture:
  use_crop: true
  crop_region:
    x: 100     # ブラウザの左上X座標
    y: 80      # ブラウザの左上Y座標
    width: 1720
    height: 900
```

---

## 日本語が正しく入力されない

### 症状
- メッセージが途中で切れる
- 文字化けする

### 原因
- pyperclip のクリップボードアクセスが失敗

### 解決方法

#### 1. pyperclip を再インストール
```bash
pip uninstall pyperclip
pip install pyperclip==1.9.0
```

#### 2. メッセージを短くテスト
`config.yaml`:
```yaml
messages:
  on_test_failed: "エラー修正してください"
  on_test_success: "いいね"
```

#### 3. write → click の順番を確認
入力欄がフォーカスされているか確認。必要なら click → write → press の順に。

---

## ループ検知で止まる

### 症状
```
[LOOP DETECTED] 同じ状態が 5 回連続しました。
```

### 原因
- 同じエラーが繰り返されている
- モデルの出力が固定化している

### 確認
```bash
cat logs/steps.jsonl
```

### 解決方法

#### 1. ループ閾値を上げる
`config.yaml`:
```yaml
loop_protection:
  max_same_state_repetitions: 10  # 5 → 10 に変更
```

#### 2. プロンプトを調整
`claude_coding_agent.py` の `INSTRUCTION_TEXT` を修正。

#### 3. dry-run で確認
`config.yaml`:
```yaml
safety:
  dry_run: true
```

ログだけ見て、状態判定が正しいか確認。

---

## dry-run モードでテスト

### 有効化
`config.yaml`:
```yaml
safety:
  dry_run: true
```

### 期待される動作
- スクショは撮る
- モデル推論は実行
- 状態判定は実行
- **アクション（click/write/press）は実行しない**
- ログに `[DRY-RUN] Click skipped` などが出る

### ログ確認
```bash
# コンソール出力
# [STATE], [ACTION SUMMARY] を確認

# JSONログ
cat logs/steps.jsonl
```

### 本番実行に切り替え
`config.yaml`:
```yaml
safety:
  dry_run: false
```

---

## 環境チェックスクリプト

問題が起きたら、まず環境チェック:
```bash
python check_environment.py
```

全て [OK] になるまで修正してください。

---

## サポート

それでも解決しない場合:
1. エラーメッセージ全文をコピー
2. `check_environment.py` の出力をコピー
3. `config.yaml` の内容を確認
4. GitHub Issues に報告
