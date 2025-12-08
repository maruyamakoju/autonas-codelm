# Claude Code セットアップガイド

## 目的
Claude Coding Autopilot Agent を Claude Code で動作させるためのセットアップ手順。

---

## 前提条件
- ✅ Python 環境セットアップ済み
- ✅ OpenCUA-7B モデルロード成功
- ✅ GPU 動作確認済み（CUDA available: True）
- ✅ dry-run モードで動作確認済み

---

## Step 1: Claude Code を開く

### オプションA: ブラウザ版
1. ブラウザで https://claude.ai にアクセス
2. ログイン
3. 「Code」モードに切り替え（またはプロジェクトを開く）

### オプションB: デスクトップアプリ
1. Claude Code デスクトップアプリを起動
2. プロジェクトを開く

---

## Step 2: test_project を開く

1. Claude Code で `C:\Users\07013\Desktop\1205muzi5090\test_project` を開く
2. Claude に以下を指示:
   ```
   Run the tests
   ```
3. テストが実行され、**1つ失敗する**ことを確認:
   ```
   FAILED test_calculator.py::test_add_intentional_fail
   ```

これでテスト準備完了。

---

## Step 3: 画面配置を調整

### 推奨レイアウト
```
┌─────────────────────────────────────────┐
│  Claude Code（ブラウザ） - 全画面      │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ Claude のチャット画面            │   │
│  │                                  │   │
│  │ [入力欄]                         │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### または（VS Code と並べる場合）
```
┌──────────────┬──────────────────────────┐
│  VS Code     │  Claude Code（ブラウザ） │
│  (左半分)    │  (右半分)                │
│              │                          │
│              │  ┌─────────────────┐    │
│              │  │ チャット画面     │    │
│              │  │                  │    │
│              │  │ [入力欄]         │    │
│              │  └─────────────────┘    │
└──────────────┴──────────────────────────┘
```

**重要**: Claude Code ブラウザの位置とサイズをメモしてください。

---

## Step 4: クロップ領域を設定

1. Claude Code ブラウザの位置を確認:
   - 左上のX座標（例: 720）
   - 左上のY座標（例: 0）
   - 幅（例: 1200）
   - 高さ（例: 1080）

2. `config.yaml` を編集:
   ```yaml
   capture:
     use_crop: true  # ★ true に変更
     crop_region:
       x: 720        # ブラウザの左上X座標
       y: 0          # ブラウザの左上Y座標
       width: 1200   # ブラウザの幅
       height: 1080  # ブラウザの高さ
   ```

---

## Step 5: dry-run テスト

1. エージェントを起動:
   ```powershell
   python claude_coding_agent.py
   ```

2. 確認ポイント:
   - ✅ `[CONFIG] Crop enabled: True`
   - ✅ `Captured screen: 1200x1080 (crop offset: 720,0)`
   - ✅ `[DRY-RUN] Click skipped`

3. `screen.png` を確認:
   ```powershell
   start screen.png
   ```
   - **Claude Code の画面だけ** が映っているか確認
   - VS Code や他のアプリが映っていないか確認

4. 5-10ステップ実行したら `Ctrl+C` で停止

---

## Step 6: 本番実行

dry-run で問題なければ:

1. `config.yaml` を編集:
   ```yaml
   safety:
     dry_run: false  # ★ false に変更
   ```

2. Claude Code で待機状態にする:
   - テストが失敗した状態で止まっている
   - または入力欄が空で待機

3. エージェント起動:
   ```powershell
   python claude_coding_agent.py
   ```

4. 期待される動作:
   - テスト失敗を検知
   - 「テストが失敗しています...」メッセージを入力
   - Enter を押す
   - Claude が修正開始
   - テスト再実行
   - 成功したら「いいね！素晴らしい...」メッセージ

---

## Step 7: ログ確認

```powershell
# リアルタイムで最新ログを表示
Get-Content logs/steps.jsonl -Wait

# 最新5件を表示
cat logs/steps.jsonl | Select-Object -Last 5
```

---

## トラブルシューティング

### クリック位置がずれる
- `screen.png` を確認
- `crop_region` の x, y, width, height を調整

### [STATE] が常に "unknown"
- Claude Code の画面が正しくキャプチャされているか確認
- INSTRUCTION_TEXT の判定キーワードを確認

### エラーループ（同じ状態が5回連続）
- 正常な防御機能
- ログを確認して原因特定
- INSTRUCTION_TEXT を調整

---

## 緊急停止

- **Ctrl+C**: 通常停止
- **マウスを画面左上(0,0)に移動**: FAILSAFE 停止
- **ループ検知**: 自動停止

---

## 次のステップ

1. 30-60分の連続運転テスト
2. ログ分析とプロンプトチューニング
3. 7B → 32B モデルへのアップグレード（GPU メモリが十分な場合）
4. 他のプロジェクトでテスト
