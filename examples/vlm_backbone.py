# phase1_siglip_gemma.py
# pip install mlx-vlm transformers opencv-python huggingface_hub

import mlx.core as mx
import numpy as np
from pathlib import Path
import json
import shutil
import urllib.request
import cv2


# ── Step 1: キーリマッピング ──────────────────────────────────────────────

def remap_keys_for_mlx_vlm(src_path: str, dst_dir: str):
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    print("Loading weights...")
    weights = mx.load(src_path)

    remapped = {}
    skipped = []
    PREFIX = "paligemma_with_expert.paligemma.model."

    for key, val in weights.items():
        if key.startswith(PREFIX):
            new_key = key[len(PREFIX):]
            # language_model.layers.* → language_model.model.layers.*
            if new_key.startswith("language_model."):
                new_key = "language_model.model." + new_key[len("language_model."):]
            remapped[new_key] = val
        else:
            skipped.append(key)

    print(f"Remapped: {len(remapped)} keys")
    print(f"Skipped: {len(skipped)} keys")

    # embed_tokens と lm_head は tied weights
    # どちらか一方しか保存されていない場合に補完する
    LM_HEAD_KEY = "paligemma_with_expert.paligemma.lm_head.weight"
    if "language_model.model.embed_tokens.weight" not in remapped \
            and LM_HEAD_KEY in weights:
        remapped["language_model.model.embed_tokens.weight"] = weights[LM_HEAD_KEY]

    elif "language_model.lm_head.weight" not in remapped \
            and "language_model.model.embed_tokens.weight" in remapped:
        remapped["language_model.lm_head.weight"] = \
            remapped["language_model.model.embed_tokens.weight"]

    out_path = str(dst_dir / "model.safetensors")
    mx.save_safetensors(out_path, remapped)
    print(f"Saved to {out_path}")
    return dst_dir


# ── Step 2: mlx-vlm 用の config.json を準備 ──────────────────────────────

def prepare_mlx_vlm_config(dst_dir: str):
    """
    mlx-vlm の paligemma2 モデルが読み込める config.json を作成する
    (google/paligemma-3b-pt-224 の構成に合わせる)
    """
    dst_dir = Path(dst_dir)

    config = {
        "model_type": "paligemma",
        "text_config": {
            "model_type": "gemma",
            "hidden_size": 2048,
            "intermediate_size": 16384,
            "num_hidden_layers": 18,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "vocab_size": 257152,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6,
        },
        "vision_config": {
            "model_type": "siglip_vision_model",
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "image_size": 224,
            "patch_size": 14,
        },
        "projection_dim": 2048,
        "ignore_index": -100,
        "image_token_index": 257152,
        "vocab_size": 257152,
    }

    with open(dst_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("config.json written.")


# ── Step 3: tokenizer を取得 ──────────────────────────────────────────────

def fetch_tokenizer(dst_dir: str):
    from huggingface_hub import snapshot_download

    dst_dir = Path(dst_dir)
    tmp = dst_dir / "_tokenizer_tmp"

    print("Downloading tokenizer from google/paligemma-3b-pt-224 ...")
    local = snapshot_download(
        repo_id="google/paligemma-3b-pt-224",
        allow_patterns=[
            "tokenizer*",
            "special_tokens_map.json",
            "preprocessor_config.json",
        ],
        local_dir=str(tmp),
    )
    for f in tmp.rglob("*"):
        if f.is_file():
            shutil.copy(f, dst_dir / f.name)
    shutil.rmtree(tmp, ignore_errors=True)
    print("Tokenizer ready.")


# ── Step 4: Web から画像を取得（OpenCV）─────────────────────────────────

def load_image_from_url(url: str, save_path: str = "/tmp/test_image.png") -> str:
    """
    URL から画像をダウンロードし、OpenCV で読み込んで
    SigLIP 入力用 (224×224, RGB) に変換して保存する。
    戻り値: 保存先パス
    """
    print(f"Downloading image from {url} ...")
    tmp_path = "/tmp/_raw_download"
    urllib.request.urlretrieve(url, tmp_path)

    img_bgr = cv2.imread(tmp_path)
    if img_bgr is None:
        raise RuntimeError(f"cv2.imread failed for {tmp_path}")

    # BGR → RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 224×224 にリサイズ（SigLIP の入力解像度）
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)

    # PNG として保存（mlx-vlm は PIL.Image.open 経由で読むためファイル渡し）
    cv2.imwrite(save_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
    print(f"Image saved to {save_path}  shape={img_resized.shape}")
    return save_path


# ── Step 5: 動作確認 ──────────────────────────────────────────────────────

def verify_phase1(model_dir: str, image_url: str):
    from mlx_vlm import load, generate
    from mlx_vlm.utils import load_config

    print(f"\nLoading model from {model_dir} ...")
    model, processor = load(model_dir)
    config = load_config(model_dir)

    # PaliGemma pt モデル向けプロンプト形式
    prompt = "caption en\n"
    images = [image_url]

    print("Running forward pass (SigLIP + Gemma 2B prefix embedding)...")
    output = generate(
        model,
        processor,
        prompt,
        images,
        max_tokens=20,
        verbose=True,
    )
    print(f"\nOutput: {output}")
    print("\nPhase 1 OK: SigLIP + Gemma 2B forward pass succeeded.")


# ── メイン ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SRC_WEIGHTS = "./models/FIwaki/pi05_base_mlx/model.safetensors"  # 変換済み MLX 重み
    PHASE1_DIR  = "./examples/outputs"

    # COCO のサンプル画像（変更自由）
    IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"

    # 1. キーリマップ & 保存
    remap_keys_for_mlx_vlm(SRC_WEIGHTS, PHASE1_DIR)

    # 2. config.json 作成
    prepare_mlx_vlm_config(PHASE1_DIR)

    # 3. tokenizer 取得
    fetch_tokenizer(PHASE1_DIR)

    # 4. 動作確認
    verify_phase1(PHASE1_DIR, IMAGE_URL)