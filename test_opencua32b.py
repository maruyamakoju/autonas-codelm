import base64
import torch
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from PIL import Image


def encode_image(image_path: str) -> str:
    """Encode image to base64 string for model input."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def load_opencua_model(model_id: str):
    """Load OpenCUA model, tokenizer, and image processor (GPU優先, 自動オフロード)."""
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

    # 32Bなので bf16 + device_map="auto" で GPU + CPU に自動オフロード
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


def create_grounding_messages(image_path: str, instruction: str):
    """Create chat messages for GUI grounding task."""
    system_prompt = (
        "You are a GUI agent. You are given a task and a screenshot of the screen. "
        "You need to perform a series of pyautogui actions to complete the task."
    )

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
    """Run inference on the model."""
    # モデルの最初のパラメータから device を取得
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

    # 生成（temperature は警告が出るだけなので付けない）
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            max_new_tokens=256,
        )

    # 出力デコード
    prompt_len = input_ids.shape[1]
    generated_ids = generated_ids[:, prompt_len:]
    output_text = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text


def main():
    model_id = "xlangai/OpenCUA-32B"
    image_path = "screenshot.png"
    instruction = "Click on the submit button"

    model, tokenizer, image_processor = load_opencua_model(model_id)

    messages = create_grounding_messages(image_path, instruction)
    result = run_inference(model, tokenizer, image_processor, messages, image_path)

    print("\n===== MODEL OUTPUT =====")
    print(result)
    print("========================\n")


if __name__ == "__main__":
    main()
