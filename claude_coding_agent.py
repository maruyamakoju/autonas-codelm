"""
Claude Coding Autopilot Agent - v0.3 (Phase 3準備版)
- CoF (Chain of Frames): 3フレーム履歴
- ダイアログ判定: proceed / allow edits
- テスト状態判定: RUNNING / FAILED / WAITING
- 複数アクション対応: click / write / press
- 状態マシン: 明示的な状態管理
- エラーループ検知: 同じ状態の連続検知と自動停止
- config.yaml: 設定の外部化
- 画面クロップ: ブラウザ領域のみを対象
- JSONログ: 1ステップ1行のログ出力
"""

import base64
import hashlib
import json
import os
import re
import time
from collections import deque
from datetime import datetime
from enum import Enum
from io import BytesIO

import torch
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from PIL import Image
import mss
import mss.tools
import pyautogui
import pyperclip

from config_loader import load_config, validate_config


# ========== 安全設定 ==========
pyautogui.FAILSAFE = True


# ========== 状態定義 ==========
class AgentState(Enum):
    """エージェントの状態"""
    UNKNOWN = "unknown"                   # 不明な状態
    PROCEED_DIALOG = "proceed_dialog"     # "Do you want to proceed?" ダイアログ
    ALLOW_EDITS_DIALOG = "allow_edits"    # "Do you want to make this edit" ダイアログ
    TESTS_RUNNING = "tests_running"       # テスト実行中
    TESTS_FAILED = "tests_failed"         # テスト失敗
    WAITING_FOR_INPUT = "waiting"         # 入力待ち（テスト成功後など）
    NO_ACTION_NEEDED = "no_action"        # 何もする必要がない


# ========== グローバル変数 ==========
# 設定（main()で初期化）
CONFIG = None

# 画像履歴: 最大3フレーム（古→新の順）
IMAGE_HISTORY = deque(maxlen=3)

# 状態履歴: (state, screen_hash, action_summary) のタプルを保持
STATE_HISTORY = deque(maxlen=10)

# クロップ設定（CONFIG から設定される）
USE_CROP = False
CROP_REGION = {"x": 0, "y": 0, "width": 1920, "height": 1080}


# ========== 基本ユーティリティ ==========

def pil_to_base64(pil_image: Image.Image) -> str:
    """PIL Image を base64 文字列に変換"""
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def compute_image_hash(pil_image: Image.Image) -> str:
    """PIL Image の SHA256 ハッシュを計算"""
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return hashlib.sha256(buffered.getvalue()).hexdigest()[:16]  # 最初の16文字のみ


def capture_screen_and_update_history():
    """
    プライマリモニタのスクショを撮り、IMAGE_HISTORY に追加。
    クロップが有効な場合は指定領域のみを取得。
    最新フレームを screen.png に保存（デバッグ用）。
    戻り値: (width, height, crop_offset_x, crop_offset_y)
    """
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # プライマリモニタ
        img_bytes = sct.grab(monitor)

        # PNG に変換
        png_bytes = mss.tools.to_png(img_bytes.rgb, img_bytes.size)
        pil_img_full = Image.open(BytesIO(png_bytes)).convert("RGB")

        # クロップ処理
        if USE_CROP:
            x = CROP_REGION["x"]
            y = CROP_REGION["y"]
            w = CROP_REGION["width"]
            h = CROP_REGION["height"]

            # クロップ（左上x, 左上y, 右下x, 右下y）
            pil_img = pil_img_full.crop((x, y, x + w, y + h))

            width = w
            height = h
            crop_offset_x = x
            crop_offset_y = y
        else:
            pil_img = pil_img_full
            width = monitor["width"]
            height = monitor["height"]
            crop_offset_x = 0
            crop_offset_y = 0

        # 履歴に追加
        IMAGE_HISTORY.append(pil_img)

        # デバッグ用に保存
        if CONFIG and CONFIG["capture"]["save_debug_screenshot"]:
            debug_path = CONFIG["capture"]["debug_screenshot_path"]
            pil_img.save(debug_path)

    return width, height, crop_offset_x, crop_offset_y


def load_opencua_model(model_id: str):
    """OpenCUA-32B を GPU 優先でロード"""
    print(f"Loading model: {model_id}")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))
        print("Capability:", torch.cuda.get_device_capability(0))

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    image_processor = AutoImageProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    return model, tokenizer, image_processor


# ========== メッセージ生成 ==========

# System Prompt: OpenCUA-32B への指示
SYSTEM_PROMPT = """You are a GUI automation agent for controlling Claude Code environment.

Your task is to analyze the screen state and decide what action to take.

OUTPUT FORMAT:
- You MUST respond with ONLY a Python code block containing pyautogui commands.
- Use ONLY these commands:
  - pyautogui.click(x=..., y=...)
  - pyautogui.write("...")
  - pyautogui.press("enter")
- If no action is needed, respond with: NO_ACTION
- Do NOT include explanations, markdown formatting, or any other text outside the code block.

Example output (when action is needed):
```python
pyautogui.click(x=1234, y=567)
pyautogui.write("Hello")
pyautogui.press("enter")
```

Example output (when no action is needed):
NO_ACTION
"""


# Instruction Text: 具体的な判断基準
INSTRUCTION_TEXT = """You are controlling the Claude Code browser interface. Analyze the screen and follow these rules IN ORDER of priority:

RULE 1: "Do you want to proceed?" Dialog
- If you see a dialog asking "Do you want to proceed?" with options:
  "❯ 1. Yes"
  "  2. Type here to tell Claude what to do differently"
- Click on the line that says "1. Yes" (or the area around it)
- This is the HIGHEST priority action.

RULE 2: "Do you want to make this edit" Dialog
- If you see a dialog asking:
  "Do you want to make this edit to [filename]?"
- And you see an option that says:
  "Yes, allow all edits during this session (alt+m)"
- Click on that option to allow all edits.
- This is the SECOND highest priority.

RULE 3: Test/Command Output Analysis
- If NO dialogs are present, look at the terminal/console area of Claude Code.
- Determine the current state:

  a) RUNNING state:
     - If you see log output scrolling, or "Running...", or active test execution
     - Do NOT interrupt. Respond with: NO_ACTION

  b) FAILED/ERROR state:
     - If you see "FAILED", "Error", "Exception", red error messages, or test failures
     - Click on the input text field at the bottom
     - Type the following message (in Japanese):
       "テストが失敗しています。表示されている主なエラーを修正して、テストを再実行してください。"
     - Press Enter

  c) WAITING state:
     - If tests passed (or no test output), and Claude Code is waiting for input
     - The input field is visible and empty
     - Click on the input field
     - Type the following message (in Japanese):
       "いいね！素晴らしい次もばんばん進めて。"
     - Press Enter

IMPORTANT:
- Check rules in order: RULE 1 → RULE 2 → RULE 3
- Only output pyautogui commands in a ```python code block, or NO_ACTION
- Be conservative: when in doubt, respond with NO_ACTION
"""


def create_messages_with_history(instruction: str):
    """
    IMAGE_HISTORY に入っている画像（最大3枚）と instruction を使って
    OpenCUA-32B 用のメッセージを生成。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # User message: 画像（古→新の順）+ テキスト
    content = []
    for img in IMAGE_HISTORY:
        b64_img = pil_to_base64(img)
        content.append({
            "type": "image",
            "image": f"data:image/png;base64,{b64_img}"
        })

    content.append({
        "type": "text",
        "text": instruction
    })

    messages.append({
        "role": "user",
        "content": content
    })

    return messages


# ========== モデル推論 ==========

def run_inference(model, tokenizer, image_processor, messages):
    """
    OpenCUA-32B に推論を実行。
    IMAGE_HISTORY に入っている画像をすべて使用。
    """
    # デバイス取得
    if hasattr(model, "device"):
        device = model.device
    else:
        device = next(model.parameters()).device

    print("Model main device:", device)

    # テキスト入力
    input_ids_list = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True
    )
    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)

    # 画像入力: IMAGE_HISTORY のすべての画像
    images = list(IMAGE_HISTORY)
    image_info = image_processor.preprocess(images=images)

    pixel_values = torch.tensor(
        image_info["pixel_values"],
        dtype=torch.bfloat16,
        device=device,
    )
    grid_thws = torch.tensor(
        image_info["image_grid_thw"],
        device=device,
    )

    # 生成
    max_tokens = CONFIG["model"]["max_new_tokens"] if CONFIG else 256
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            max_new_tokens=max_tokens,
        )

    prompt_len = input_ids.shape[1]
    generated_ids = generated_ids[:, prompt_len:]
    output_text = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text


# ========== アクション抽出 ==========

# 正規表現: pyautogui.click / write / press
ACTION_RE = re.compile(
    r"pyautogui\.(click|write|press)\("
    r"(?:x=(\d+),\s*y=(\d+)|"  # click(x=..., y=...)
    r"['\"]([^'\"]*)['\"]|"    # write("...") or press("...")
    r"([a-z]+))"               # press(enter) など
    r"\)"
)


def extract_actions(text: str):
    """
    モデル出力から pyautogui アクションを抽出。
    戻り値: [("click", x, y), ("write", text), ("press", key), ...]
    """
    # NO_ACTION 判定
    if "NO_ACTION" in text and "pyautogui" not in text:
        return []

    # ```python ... ``` ブロックを抽出
    code_blocks = re.findall(r"```python\s+(.*?)\s+```", text, re.DOTALL)
    if code_blocks:
        code = "\n".join(code_blocks)
    else:
        code = text

    actions = []
    for match in ACTION_RE.finditer(code):
        action_type = match.group(1)

        if action_type == "click":
            x = match.group(2)
            y = match.group(3)
            if x and y:
                actions.append(("click", int(x), int(y)))

        elif action_type == "write":
            text_content = match.group(4)
            if text_content:
                actions.append(("write", text_content))

        elif action_type == "press":
            key = match.group(4) or match.group(5)
            if key:
                actions.append(("press", key))

    return actions


def determine_state(output_text: str, actions: list) -> AgentState:
    """
    モデル出力とアクションから現在の状態を判定。
    """
    # NO_ACTION の場合
    if not actions:
        # 実行中かどうかの判定（出力テキストから推測）
        if any(keyword in output_text.lower() for keyword in ["running", "executing", "in progress"]):
            return AgentState.TESTS_RUNNING
        return AgentState.NO_ACTION_NEEDED

    # アクションがある場合、writeの内容から判定
    for action in actions:
        if action[0] == "write":
            text_content = action[1].lower()

            # エラー修正メッセージ → テスト失敗状態
            if "失敗" in text_content or "エラー" in text_content or "修正" in text_content:
                return AgentState.TESTS_FAILED

            # 「いいね」メッセージ → 入力待ち状態
            if "いいね" in text_content or "素晴らしい" in text_content or "進めて" in text_content:
                return AgentState.WAITING_FOR_INPUT

    # クリックのみの場合、出力テキストから判定
    output_lower = output_text.lower()

    if "proceed" in output_lower or "1. yes" in output_lower:
        return AgentState.PROCEED_DIALOG

    if "allow all edits" in output_lower or "alt+m" in output_lower:
        return AgentState.ALLOW_EDITS_DIALOG

    # どれにも該当しない
    return AgentState.UNKNOWN


def get_action_summary(actions: list) -> str:
    """アクションのサマリー文字列を生成（ループ検知用）"""
    if not actions:
        return "NO_ACTION"

    summary_parts = []
    for action in actions:
        if action[0] == "click":
            summary_parts.append(f"click({action[1]},{action[2]})")
        elif action[0] == "write":
            # テキストは最初の20文字のみ
            text = action[1][:20] + "..." if len(action[1]) > 20 else action[1]
            summary_parts.append(f"write({repr(text)})")
        elif action[0] == "press":
            summary_parts.append(f"press({action[1]})")

    return "|".join(summary_parts)


def check_loop_detection() -> tuple[bool, int]:
    """
    エラーループを検知。
    戻り値: (is_loop_detected, repetition_count)
    """
    max_reps = CONFIG["loop_protection"]["max_same_state_repetitions"] if CONFIG else 5

    if len(STATE_HISTORY) < max_reps:
        return False, 0

    # 最新の状態を取得
    latest_state, latest_hash, latest_action = STATE_HISTORY[-1]

    # 最新のN件が全て同じ状態+ハッシュ+アクションかチェック
    repetition_count = 0
    for state, screen_hash, action_summary in reversed(STATE_HISTORY):
        if state == latest_state and screen_hash == latest_hash and action_summary == latest_action:
            repetition_count += 1
        else:
            break

    is_loop = repetition_count >= max_reps
    return is_loop, repetition_count


# ========== 座標変換 ==========

def qwen25_smart_resize_to_absolute(model_x, model_y, original_width, original_height):
    """
    OpenCUA-32B の出力座標（smart_resize後の絶対座標）を
    元のスクリーン座標に変換。
    """
    resized_h, resized_w = smart_resize(
        original_height,
        original_width,
        factor=28,
        min_pixels=3136,
        max_pixels=12845056,
    )
    rel_x = model_x / resized_w
    rel_y = model_y / resized_h
    abs_x = int(rel_x * original_width)
    abs_y = int(rel_y * original_height)
    return abs_x, abs_y


# ========== アクション実行 ==========

def execute_actions(actions, screen_w, screen_h, crop_offset_x=0, crop_offset_y=0):
    """
    抽出したアクションリストを順に実行。
    クロップ使用時はオフセットを追加。
    """
    # dry-runモードチェック
    dry_run = CONFIG and CONFIG["safety"]["dry_run"]

    for action in actions:
        action_type = action[0]

        if action_type == "click":
            _, mx, my = action
            sx, sy = qwen25_smart_resize_to_absolute(mx, my, screen_w, screen_h)
            # クロップオフセットを追加
            final_x = sx + crop_offset_x
            final_y = sy + crop_offset_y
            print(f"  → Click: model({mx},{my}) -> screen({sx},{sy}) -> final({final_x},{final_y})")
            if not dry_run:
                pyautogui.click(x=final_x, y=final_y)
            else:
                print(f"    [DRY-RUN] Click skipped")
            time.sleep(0.3)

        elif action_type == "write":
            _, text = action
            print(f"  → Write: {repr(text)}")
            if not dry_run:
                # pyautogui.write()は日本語非対応のため、クリップボード経由で貼り付け
                pyperclip.copy(text)
                pyautogui.hotkey('ctrl', 'v')
            else:
                print(f"    [DRY-RUN] Write skipped")
            time.sleep(0.3)

        elif action_type == "press":
            _, key = action
            print(f"  → Press: {key}")
            if not dry_run:
                pyautogui.press(key)
            else:
                print(f"    [DRY-RUN] Press skipped")
            time.sleep(0.2)


# ========== メインループ ==========

def write_json_log(step, state, screen_hash, actions, output_text=""):
    """JSONログを1行書き込み"""
    if not CONFIG or not CONFIG["logging"]["enable_json_log"]:
        return

    log_path = CONFIG["logging"]["json_log_path"]

    # ディレクトリ作成
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    log_entry = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "state": state.value if isinstance(state, Enum) else state,
        "screen_hash": screen_hash,
        "actions": get_action_summary(actions),
        "num_actions": len(actions),
        "output_snippet": output_text[:100] if output_text else ""
    }

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[WARNING] Failed to write JSON log: {e}")


def run_one_step(model, tokenizer, image_processor, step_num=0):
    """
    1ステップ実行:
      1. スクショ撮影 & 履歴更新
      2. モデル推論
      3. アクション抽出
      4. 状態判定
      5. 状態履歴への追加
      6. エラーループ検知
      7. アクション実行
      8. JSONログ出力
    戻り値: (should_stop, reason)
    """
    # スクショ & 履歴更新
    screen_w, screen_h, crop_offset_x, crop_offset_y = capture_screen_and_update_history()
    latest_image = IMAGE_HISTORY[-1]
    screen_hash = compute_image_hash(latest_image)

    crop_info = f" (crop offset: {crop_offset_x},{crop_offset_y})" if USE_CROP else ""
    print(f"Captured screen: {screen_w}x{screen_h}{crop_info} (History size: {len(IMAGE_HISTORY)}, Hash: {screen_hash})")

    # メッセージ生成
    messages = create_messages_with_history(INSTRUCTION_TEXT)

    # 推論
    output_text = run_inference(model, tokenizer, image_processor, messages)

    if CONFIG and CONFIG["logging"]["verbose"]:
        print("\n===== RAW MODEL OUTPUT =====")
        print(output_text)
        print("================================\n")

    # アクション抽出
    actions = extract_actions(output_text)

    # 状態判定
    current_state = determine_state(output_text, actions)
    action_summary = get_action_summary(actions)

    print(f"[STATE] {current_state.value}")
    print(f"[ACTION SUMMARY] {action_summary}")

    # 状態履歴に追加
    STATE_HISTORY.append((current_state, screen_hash, action_summary))

    # JSONログ出力
    write_json_log(step_num, current_state, screen_hash, actions, output_text)

    # エラーループ検知
    is_loop, repetition_count = check_loop_detection()
    if is_loop:
        print(f"\n[LOOP DETECTED] 同じ状態が {repetition_count} 回連続しました。")
        print(f"  状態: {current_state.value}")
        print(f"  ハッシュ: {screen_hash}")
        print(f"  アクション: {action_summary}")
        print("  エージェントを自動停止します。")
        return True, f"Loop detected: {current_state.value} x{repetition_count}"

    if not actions:
        print("No actions to execute (NO_ACTION or no valid pyautogui commands).")
        return False, None

    print(f"Extracted {len(actions)} action(s):")
    for i, action in enumerate(actions, 1):
        print(f"  {i}. {action[0]}: {action[1:]}")

    # アクション実行
    execute_actions(actions, screen_w, screen_h, crop_offset_x, crop_offset_y)
    print("Actions executed successfully.\n")

    return False, None


def main():
    global CONFIG, USE_CROP, CROP_REGION

    # 設定ロード
    print("\n========== Claude Coding Autopilot Agent v0.3 ==========")
    print("Loading configuration...")
    CONFIG = load_config("config.yaml")

    if not validate_config(CONFIG):
        print("[ERROR] Configuration validation failed. Exiting.")
        return

    # グローバル変数に設定を反映
    USE_CROP = CONFIG["capture"]["use_crop"]
    CROP_REGION = CONFIG["capture"]["crop_region"]

    # FAILSAFE設定
    pyautogui.FAILSAFE = CONFIG["safety"]["enable_failsafe"]

    # ログディレクトリ作成
    log_dir = CONFIG["logging"]["log_dir"]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        print(f"[INFO] Created log directory: {log_dir}")

    # 設定表示
    print(f"[CONFIG] Loop protection: {CONFIG['loop_protection']['max_same_state_repetitions']} repetitions")
    print(f"[CONFIG] Crop enabled: {USE_CROP}")
    if USE_CROP:
        print(f"[CONFIG] Crop region: {CROP_REGION}")
    print(f"[CONFIG] JSON logging: {CONFIG['logging']['enable_json_log']}")
    print(f"[CONFIG] Dry-run mode: {CONFIG['safety']['dry_run']}")
    print()

    # モデルロード
    model_id = CONFIG["model"]["model_id"]
    print(f"Loading model: {model_id}...")
    model, tokenizer, image_processor = load_opencua_model(model_id)
    print("Model loaded successfully.\n")

    # 初回スクショで履歴を初期化
    print("Initializing screen capture history...")
    capture_screen_and_update_history()
    print(f"History initialized with {len(IMAGE_HISTORY)} frame(s).\n")

    # メインループ設定
    max_steps = CONFIG["safety"]["max_steps"]
    sleep_between_steps = CONFIG["model"]["sleep_between_steps"]

    print("==== Agent Started ====")
    print("Ctrl+C または マウスを画面左上(0,0)付近に動かすと停止できます。")
    print(f"ステップ間隔: {sleep_between_steps}秒")
    print(f"最大ステップ数: {max_steps}\n")

    step = 0
    stop_reason = None

    while step < max_steps:
        step += 1
        print(f"\n{'='*60}")
        print(f"STEP {step} / {max_steps}")
        print(f"{'='*60}")

        try:
            should_stop, reason = run_one_step(model, tokenizer, image_processor, step_num=step)

            if should_stop:
                stop_reason = reason
                print(f"\n[AUTO STOP] {reason}")
                break

        except pyautogui.FailSafeException:
            stop_reason = "FAILSAFE: マウスが画面左上(0,0)に移動"
            print(f"\n[FAILSAFE] マウスが画面左上(0,0)に移動したため停止しました。")
            break

        except KeyboardInterrupt:
            stop_reason = "KeyboardInterrupt: Ctrl+C"
            print("\n[KeyboardInterrupt] Ctrl+C が押されたため停止しました。")
            break

        except Exception as e:
            print(f"\n[ERROR] 予期しないエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            # エラーが起きても続行（必要なら break に変更）

        # 次ステップまで待機
        time.sleep(sleep_between_steps)

    print("\n==== Agent Stopped ====")
    if stop_reason:
        print(f"停止理由: {stop_reason}")
    print(f"総ステップ数: {step}")
    print(f"状態履歴サイズ: {len(STATE_HISTORY)}")


if __name__ == "__main__":
    main()
