# convert_pi05_to_mlx.py
# lerobot/pi05_base を MLX bf16 形式に変換し HuggingFace にアップロードする
#
# 必要:
#   pip install torch safetensors mlx huggingface_hub
#   huggingface-cli login

import shutil
from pathlib import Path

import mlx.core as mx
from safetensors.torch import load_file as torch_load_file
from huggingface_hub import HfApi, snapshot_download


# ═══════════════════════════════════════════════════════════════
# 設定
# ═══════════════════════════════════════════════════════════════

MODEL_DIR = "./models/lerobot/pi05_base"
OUTPUT_DIR = "./models/FIwaki/pi05_base_mlx_bf16"
HF_REPO_ID = "FIwaki/pi05_base_mlx_bf16"

# lerobot/pi05_base からコピーするファイル
COPY_FILES = [
    "config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "README.md",
]


# ═══════════════════════════════════════════════════════════════
# 変換
# ═══════════════════════════════════════════════════════════════


def convert_pi05_to_mlx_bf16(input_path: str, output_dir: str):
    """
    lerobot/pi05_base の model.safetensors を MLX bf16 形式に変換して保存する。

    変換内容:
      - float32 → bfloat16 (mx.save_safetensors でネイティブ bf16 保存)
      - Conv2d weight: PyTorch [O,I,H,W] → MLX [O,H,W,I]
      - その他のテンソルはキー名そのまま保持
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {input_path}  ({input_path.stat().st_size / 1e9:.1f} GB)")
    pt_tensors = torch_load_file(str(input_path), device="cpu")
    print(f"Total tensors: {len(pt_tensors)}")

    mlx_tensors = {}
    for i, (key, tensor) in enumerate(pt_tensors.items()):
        if i % 100 == 0:
            print(f"  [{i}/{len(pt_tensors)}] converting...")

        # float32 経由で numpy → MLX array に変換
        np_array = tensor.float().numpy()
        mlx_array = mx.array(np_array)

        # Conv2d weight: [O,I,H,W] → [O,H,W,I]
        if "patch_embedding.weight" in key and mlx_array.ndim == 4:
            mlx_array = mx.transpose(mlx_array, (0, 2, 3, 1))
            print(
                f"  Transposed Conv2d: {key}  {list(tensor.shape)} → {list(mlx_array.shape)}"
            )

        # bf16 にキャスト
        mlx_tensors[key] = mlx_array.astype(mx.bfloat16)

    # MLX ネイティブで bf16 safetensors として保存
    output_path = str(output_dir / "model.safetensors")
    print(f"\nSaving to {output_path} ...")
    mx.save_safetensors(output_path, mlx_tensors)

    size_gb = Path(output_path).stat().st_size / 1e9
    print(f"Saved. ({size_gb:.2f} GB)")


# ═══════════════════════════════════════════════════════════════
# 付属ファイルのコピー
# ═══════════════════════════════════════════════════════════════


def copy_metadata(src_dir: str, dst_dir: str):
    """lerobot/pi05_base の設定ファイル類をコピーする"""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    for fname in COPY_FILES:
        src = src_dir / fname
        if src.exists():
            shutil.copy(src, dst_dir / fname)
            print(f"Copied: {fname}")
        else:
            print(f"Skipped (not found): {fname}")


# ═══════════════════════════════════════════════════════════════
# mlx_vlmで読み込める形式に変換
# ═══════════════════════════════════════════════════════════════

# def build_mlx_vlm_compatible_repo(
#     mlx_weights_path: str,   # 変換済み model.safetensors
#     output_dir: str,
#     src_pt_tensors: dict,    # torch_load_file で読んだ元テンソル辞書
# ):
#     """
#     mlx-vlm の load() が直接受け付けられるディレクトリを構成する。
#     1. PaliGemma 重みをキーリマップして model.safetensors に保存
#     2. config.json を生成
#     3. tokenizer / preprocessor_config を取得
#     """
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # ── 1. PaliGemma 重みのキーリマップ ──────────────────────────
#     PREFIX = "paligemma_with_expert.paligemma.model."
#     remapped = {}

#     for key, tensor in src_pt_tensors.items():
#         if not key.startswith(PREFIX):
#             continue
#         new_key = key[len(PREFIX):]
#         # language_model.layers.* → language_model.model.layers.*
#         if new_key.startswith("language_model."):
#             new_key = "language_model.model." + new_key[len("language_model."):]
#         np_array = tensor.float().numpy()
#         mlx_array = mx.array(np_array)
#         if "patch_embedding.weight" in key and mlx_array.ndim == 4:
#             mlx_array = mx.transpose(mlx_array, (0, 2, 3, 1))
#         remapped[new_key] = mlx_array.astype(mx.bfloat16)

#     # embed_tokens の tied weights 補完
#     LM_HEAD = "paligemma_with_expert.paligemma.lm_head.weight"
#     if "language_model.model.embed_tokens.weight" not in remapped \
#             and LM_HEAD in src_pt_tensors:
#         t = src_pt_tensors[LM_HEAD]
#         remapped["language_model.model.embed_tokens.weight"] = \
#             mx.array(t.float().numpy()).astype(mx.bfloat16)

#     out_path = str(output_dir / "model.safetensors")
#     mx.save_safetensors(out_path, remapped)
#     print(f"Saved remapped PaliGemma weights: {len(remapped)} keys → {out_path}")

#     # ── 2. config.json を生成 ─────────────────────────────────────
#     import json
#     config = {
#         "model_type": "paligemma",
#         "text_config": {
#             "model_type": "gemma",
#             "hidden_size": 2048,
#             "intermediate_size": 16384,
#             "num_hidden_layers": 18,
#             "num_attention_heads": 8,
#             "num_key_value_heads": 1,
#             "vocab_size": 257152,
#             "rope_theta": 10000.0,
#             "rms_norm_eps": 1e-6,
#         },
#         "vision_config": {
#             "model_type": "siglip_vision_model",
#             "hidden_size": 1152,
#             "intermediate_size": 4304,
#             "num_hidden_layers": 27,
#             "num_attention_heads": 16,
#             "image_size": 224,
#             "patch_size": 14,
#         },
#         "projection_dim": 2048,
#         "ignore_index": -100,
#         "image_token_index": 257152,
#         "vocab_size": 257152,
#     }
#     with open(output_dir / "config.json", "w") as f:
#         json.dump(config, f, indent=2)
#     print("Saved config.json")

#     # ── 3. tokenizer / preprocessor_config を取得 ─────────────────
#     print("Downloading tokenizer from google/paligemma-3b-pt-224 ...")
#     tmp = output_dir / "_tmp_tokenizer"
#     snapshot_download(
#         repo_id="google/paligemma-3b-pt-224",
#         allow_patterns=["tokenizer*", "special_tokens_map.json", "preprocessor_config.json"],
#         local_dir=str(tmp),
#     )
#     for f in tmp.rglob("*"):
#         if f.is_file():
#             shutil.copy(f, output_dir / f.name)
#     shutil.rmtree(tmp, ignore_errors=True)
#     print("Tokenizer ready.")


# ═══════════════════════════════════════════════════════════════
# HuggingFace へのアップロード
# ═══════════════════════════════════════════════════════════════


def upload_to_huggingface(output_dir: str, repo_id: str):
    """変換済みモデルを HuggingFace Hub にアップロードする"""
    api = HfApi()

    print(f"\nCreating repo: {repo_id}")
    api.create_repo(repo_id, exist_ok=True)

    print(f"Uploading {output_dir} → {repo_id} ...")
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        commit_message="Add MLX bf16 converted pi05_base weights",
    )
    print(f"Done: https://huggingface.co/{repo_id}")


# ═══════════════════════════════════════════════════════════════
# メイン
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    src_dir = Path(MODEL_DIR)
    input_path = src_dir / "model.safetensors"

    # 1. PyTorch テンソルを一度だけ読み込む
    print(f"Loading: {input_path}")
    pt_tensors = torch_load_file(str(input_path), device="cpu")

    # 2. フル変換（全キー bf16）→ pi05_base_mlx_bf16 へ
    convert_pi05_to_mlx_bf16(str(input_path), OUTPUT_DIR)
    copy_metadata(src_dir, OUTPUT_DIR)

    # 3. PaliGemma リマップ済み → pi05_paligemma_mlx へ
    #    これが mlx-vlm の load() に直接渡せるディレクトリになる
    # PALIGEMMA_DIR = OUTPUT_DIR.replace("pi05_base_mlx_bf16", "pi05_paligemma_mlx")
    # build_mlx_vlm_compatible_repo(
    #     mlx_weights_path = OUTPUT_DIR + "/model.safetensors",
    #     output_dir       = PALIGEMMA_DIR,
    #     src_pt_tensors   = pt_tensors,
    # )

    # 4. 両方アップロード
    upload_to_huggingface(OUTPUT_DIR, "FIwaki/pi05_base_mlx_bf16")
    # upload_to_huggingface(PALIGEMMA_DIR, "FIwaki/pi05_paligemma_mlx")
