# Claude Coding Autopilot Agent - クイックスタートガイド

## v0.3 (Phase 3準備版)

### 1. 環境チェック

```bash
python check_environment.py
```

全てのチェックに合格することを確認してください。

### 2. セットアップ（初回のみ）

```bash
# Windowsの場合
setup.bat

# または手動で
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 設定ファイルの確認

`config.yaml` を確認してください。デフォルト設定で動作しますが、必要に応じてカスタマイズできます。

**重要な設定項目:**

```yaml
# ループ検知閾値（同じ状態が何回連続したら停止するか）
loop_protection:
  max_same_state_repetitions: 5

# 画面クロップ（ブラウザ領域のみを対象）
capture:
  use_crop: false  # trueにすると指定領域のみキャプチャ
  crop_region:
    x: 0
    y: 0
    width: 1920
    height: 1080

# dry-runモード（アクションを実行せずログのみ）
safety:
  dry_run: false  # trueにするとテスト実行
```

### 4. テスト用プロジェクトで試運転

1. `test_project/` ディレクトリに移動
2. Claude Code でこのプロジェクトを開く
3. Claude に「Run the tests」と指示
4. エージェントを起動:

```bash
python claude_coding_agent.py
```

### 5. 期待される動作

- テスト実行 → 1つ失敗
- エージェントが「テストが失敗しています...」メッセージを送信
- Claude が修正
- エージェントが「いいね！素晴らしい...」メッセージを送信
- テスト再実行 → 全て成功

### 6. 停止方法

- **Ctrl+C**: 通常の停止
- **マウスを画面左上(0,0)に移動**: 緊急停止（FAILSAFE）
- **同じ状態が5回連続**: 自動停止（ループ検知）

### 7. ログ確認

```bash
# JSONログ（1ステップ1行）
cat logs/steps.jsonl

# または分析
python -c "import json; [print(json.loads(line)['state']) for line in open('logs/steps.jsonl')]"
```

### 8. dry-runモードでテスト

アクションを実行せずにログだけ確認したい場合:

1. `config.yaml` で `safety.dry_run: true` に設定
2. エージェントを起動
3. ログで状態判定とアクション抽出を確認
4. 問題なければ `dry_run: false` に戻して本番実行

### 9. 画面クロップの設定

ブラウザ領域のみを対象にして誤クリックを防ぐ:

1. ブラウザを開いて位置とサイズを確認
2. `config.yaml` で設定:

```yaml
capture:
  use_crop: true
  crop_region:
    x: 100      # ブラウザの左上X座標
    y: 80       # ブラウザの左上Y座標
    width: 1720 # ブラウザの幅
    height: 900 # ブラウザの高さ
```

3. `screen.png` を確認してクロップ領域が正しいか確認

### 10. トラブルシューティング

**Q: クリック位置がずれる**
- `check_environment.py` でディスプレイ設定を確認
- Windowsスケーリングを100%に設定
- クロップ領域を確認

**Q: 日本語が正しく入力されない**
- `pyperclip` が正しくインストールされているか確認
- Windowsのクリップボードが正常に動作するか確認

**Q: ループ検知で停止する**
- 正常な動作です（同じエラーが5回連続した場合）
- ログを確認して原因を特定
- プロンプトやconfig設定を調整

**Q: GPUが認識されない**
- `check_environment.py` でCUDA確認
- PyTorchのCUDAバージョンを確認
- nvidia-smi でGPUステータスを確認

### 11. 次のステップ

1. テストプロジェクトで30-60分の連続運転テスト
2. ログを分析してプロンプトをチューニング
3. 本番プロジェクトに適用
4. config.yamlで細かい調整

---

**さらに詳しい情報:**
- `USAGE.txt`: 詳細な使い方
- `PROJECT_SUMMARY.txt`: 実装の詳細
- `PHASE2_CHANGELOG.txt`: Phase 2の変更履歴
