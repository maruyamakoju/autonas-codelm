import base64
import re
import time

import torch
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from PIL import Image
import mss
import mss.tools
import pyautogui


# 安全用: 画面左上(0,0)にマウスを持っていくと PyAutoGUI が例外を出して止まる
pyautogui.FAILSAFE = True


# ========= 基本ユーティリティ =========

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        import base64 as _b64
        return _b64.b64encode(f.read()).decode()


def capture_screen(image_path: str = "screen.png"):
    """プライマリモニタのスクショを撮って保存、(path, width, height) を返す"""
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # 1: プライマリモニタ
        img = sct.grab(monitor)
        mss.tools.to_png(img.rgb, img.size, output=image_path)
        width = monitor["width"]
        height = monitor["height"]
    return image_path, width, height


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
        device_map="auto",      # VRAM に入りきらない分は CPU にオフロード
        trust_remote_code=True,
    )

    image_processor = AutoImageProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    return model, tokenizer, image_processor


def create_messages(image_path: str, instruction: str):
    """
    Claude Code の「Do you want to proceed?」確認ダイアログだけを処理するように指示する。
    """
    system_prompt = (
        "You are a GUI automation agent controlling my development environment. "
        "Your ONLY job is to handle confirmation prompts in Claude Code.\n\n"
        "On the screen, sometimes there is a confirmation box that looks like:\n"
        "\"Do you want to proceed?  ❯ 1. Yes    2. Type here to tell Claude what to do differently\"\n\n"
        "When this confirmation prompt is visible, you MUST click on the UI element "
        "that selects option '1. Yes' so that execution proceeds.\n"
        "If the prompt is NOT visible, you MUST do nothing.\n\n"
        "Your response format:\n"
        "- If you need to click, respond ONLY with one or more lines of Python code using pyautogui, "
        "for example: pyautogui.click(x=1234, y=567)\n"
        "- If there is nothing to do (no such confirmation prompt), respond with exactly one line: NO_ACTION\n"
        "Do NOT include explanations, markdown, or any other text."
    )

    # instruction は追加の補助指示として渡すが、system_prompt がメイン
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"data:image/png;base64,{encode_image(image_path)}",
                },
                {"type": "text", "text": instruction},
            ],
        },
    ]
    return messages


def run_inference(model, tokenizer, image_processor, messages, image_path):
    """OpenCUA に 1 ステップ問い合わせ"""
    device = next(model.parameters()).device
    print("Model main device:", device)

    # テキスト入力
    input_ids_list = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)

    # 画像入力
    image = Image.open(image_path).convert("RGB")
    image_info = image_processor.preprocess(images=[image])
    pixel_values = torch.tensor(
        image_info["pixel_values"],
        dtype=torch.bfloat16,
        device=device,
    )
    grid_thws = torch.tensor(
        image_info["image_grid_thw"],
        device=device,
    )

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            max_new_tokens=256,
        )

    prompt_len = input_ids.shape[1]
    generated_ids = generated_ids[:, prompt_len:]
    output_text = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text


# ========= 座標変換 & 実行 =========

def qwen25_smart_resize_to_absolute(model_x, model_y, original_width, original_height):
    """
    OpenCUA-32B (Qwen2.5 ベース) が出す座標は
    「smart_resize 後の画像上の絶対座標」なので、
    元のスクリーン座標に戻す。
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


CLICK_RE = re.compile(
    r"pyautogui\.click\(\s*x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)\s*\)"
)


def extract_clicks(text: str):
    """モデル出力テキストから click 座標だけ抜き出す"""
    clicks = []
    for mx, my in CLICK_RE.findall(text):
        clicks.append((int(mx), int(my)))
    return clicks


def run_one_step(model, tokenizer, image_processor, task: str):
    """
    1 ステップ:
      - 画面キャプチャ
      - モデルに投げる
      - click 座標をパース
      - 実画面座標に変換してクリック
      - 何もすることがなければ NO_ACTION としてスキップ
    """
    img_path, screen_w, screen_h = capture_screen("screen.png")
    print(f"Captured screen: {screen_w}x{screen_h}")

    messages = create_messages(img_path, task)
    output_text = run_inference(model, tokenizer, image_processor, messages, img_path)

    print("\n===== RAW MODEL OUTPUT =====")
    print(output_text)
    print("================================\n")

    # 何もする必要がない場合
    if "NO_ACTION" in output_text.strip():
        print("Model returned NO_ACTION. Skipping click this step.")
        return

    clicks = extract_clicks(output_text)
    if not clicks:
        print("No pyautogui.click(...) found in model output.")
        return

    print("Parsed click coordinates (model space):", clicks)

    for (mx, my) in clicks:
        sx, sy = qwen25_smart_resize_to_absolute(mx, my, screen_w, screen_h)
        print(f"Clicking: model({mx},{my}) -> screen({sx},{sy})")
        pyautogui.click(x=sx, y=sy)
        time.sleep(0.3)  # 連続クリックがあれば少し間をあける


# ========= メインループ =========

def main():
    model_id = "xlangai/OpenCUA-32B"

    # Claude Code の proceed ダイアログ専用タスク
    task = (
        "Your only task is to confirm the 'Do you want to proceed?' prompt in Claude Code "
        "by selecting option '1. Yes' when it is visible. "
        "If the prompt is not visible, do nothing and respond with NO_ACTION."
    )

    # まずモデルをロード（最初だけ重い）
    model, tokenizer, image_processor = load_opencua_model(model_id)

    # ほぼ無限ループで監視（安全のため上限ステップもつけておく）
    max_steps = 1000000
    step = 0

    print("\n==== Claude proceed agent started ====")
    print("Ctrl+C または マウスを画面左上(0,0)付近に動かすと停止できます。")

    while step < max_steps:
        step += 1
        print(f"\n==== STEP {step} / {max_steps} ====")

        try:
            run_one_step(model, tokenizer, image_processor, task)
        except pyautogui.FailSafeException:
            print("PyAutoGUI FAILSAFE: 左上 (0,0) にマウスが移動したので停止しました。")
            break
        except KeyboardInterrupt:
            print("KeyboardInterrupt: 手動で中断しました。")
            break

        # 次のチェックまで少し待つ（必要に応じて調整: 秒数を短くすると反応は速くなる）
        time.sleep(2.0)

    print("Agent finished / stopped.")


if __name__ == "__main__":
    main()
