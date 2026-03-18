# pi05_mlx.py
# MLX版 π₀.₅ 推論
#
# 入力:
#   observation.state : np.ndarray (state_dim,)   現在の状態
#   observation.images: dict[str, np.ndarray]      カメラ画像 (H,W,3) uint8 BGR
#   task              : str                         言語指示
#
# 出力:
#   actions           : np.ndarray (action_horizon, action_dim)
#
# 使い方:
#   python pi05_mlx.py
#
# 前提:
#   - Phase 1 重み変換済み: PHASE1_DIR に config.json / model.safetensors 等
#   - 変換済み bf16 重み : SRC_WEIGHTS に model.safetensors
#   - phase2_gemma_expert.py が同ディレクトリに存在

import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import urllib.request

import cv2
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

from action_expert import (
    Pi05Phase2,
    load_expert_weights,
    ACTION_DIM,
    ACTION_HORIZON,
    PG_HIDDEN,
)

# ═══════════════════════════════════════════════════════════════
# 定数
# ═══════════════════════════════════════════════════════════════

NUM_INFERENCE_STEPS = 10
MAX_STATE_DIM       = 32
MAX_ACTION_DIM      = 32
STATE_BINS          = 256
IMAGE_SIZE          = 224
MAX_TOKEN_LEN       = 200

SIGLIP_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SIGLIP_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)

# カメラキーの優先順 (存在するものを最大3枚使用)
CAMERA_KEYS = [
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
    "exterior_image_1_left",
    "wrist_image_left",
]


# ═══════════════════════════════════════════════════════════════
# 前処理
# ═══════════════════════════════════════════════════════════════

class Preprocessor:

    def __init__(self, tokenizer_name: str = "google/paligemma-3b-pt-224"):
        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def preprocess_image(self, img_bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 → [-1, 1] float32 (H, W, C)"""
        img = cv2.resize(img_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - SIGLIP_MEAN) / SIGLIP_STD
        return img

    @staticmethod
    def discretize_state(state: np.ndarray, n_bins: int = STATE_BINS) -> str:
        """[-1,1] の正規化済み状態 → 'sXXX sXXX ...' トークン文字列"""
        bins = np.clip(((state + 1.0) / 2.0 * n_bins).astype(int), 0, n_bins - 1)
        return " ".join(f"s{b:03d}" for b in bins)

    def tokenize(
        self,
        task: str,
        norm_state: Optional[np.ndarray] = None,
        max_length: int = MAX_TOKEN_LEN,
    ) -> Dict[str, np.ndarray]:
        """言語指示 [+ 離散化状態] → token ids / attention_mask"""
        if norm_state is not None:
            state_str = self.discretize_state(norm_state)
            prompt = f"{task}\n{state_str}"
        else:
            prompt = task

        enc = self.tokenizer(
            prompt,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        return {
            "input_ids":      enc["input_ids"][0].astype(np.int32),
            "attention_mask": enc["attention_mask"][0].astype(np.int32),
        }

    @staticmethod
    def normalize_quantile(x, q01, q99, eps=1e-8):
        return np.clip(2.0 * (x - q01) / (q99 - q01 + eps) - 1.0, -1.0, 1.0)

    @staticmethod
    def unnormalize_quantile(x, q01, q99, eps=1e-8):
        return (x + 1.0) / 2.0 * (q99 - q01 + eps) + q01


# ═══════════════════════════════════════════════════════════════
# PaliGemma hidden states 抽出
# ═══════════════════════════════════════════════════════════════

def extract_paligemma_hidden(
    pg_model,
    images: List[np.ndarray],      # [(H,W,C) float32 in [-1,1], ...]
    token_ids: np.ndarray,         # (max_length,) int32
) -> mx.array:
    """
    SigLIP + Projector + Gemma layers を通して
    最終 hidden states (1, T_prefix, PG_HIDDEN) を返す。
    """
    vision_tower = pg_model.vision_tower
    projector    = pg_model.multi_modal_projector
    lm           = pg_model.language_model

    # ── 画像 embedding ──────────────────────────────────────────
    img_embeds_list = []
    for img in images:
        pv = mx.array(img[None])           # (1, H, W, C)

        vision_out = vision_tower(pv)
        if isinstance(vision_out, tuple):
            vision_out = vision_out[0]
        if vision_out.ndim == 2:
            vision_out = mx.expand_dims(vision_out, axis=0)

        img_embed = projector(vision_out)
        if isinstance(img_embed, tuple):
            img_embed = img_embed[0]
        if img_embed.ndim == 2:
            img_embed = mx.expand_dims(img_embed, axis=0)

        img_embeds_list.append(img_embed)   # (1, 256, 2048)

    img_embeds = mx.concatenate(img_embeds_list, axis=1)  # (1, N*256, 2048)

    # ── テキスト embedding ──────────────────────────────────────
    ids = mx.array(token_ids[None])                        # (1, T)
    text_embeds = lm.model.embed_tokens(ids)               # (1, T, 2048)

    # ── 結合 → Gemma layers ──────────────────────────────────────
    h = mx.concatenate([img_embeds, text_embeds], axis=1)  # (1, T_total, 2048)

    for layer in lm.model.layers:
        h = layer(h)

    if hasattr(lm.model, "norm"):
        h = lm.model.norm(h)

    return h   # (1, T_total, 2048)


# ═══════════════════════════════════════════════════════════════
# Flow Matching 推論ループ
# ═══════════════════════════════════════════════════════════════

def flow_matching_sample(
    expert_model: Pi05Phase2,
    pg_hidden: mx.array,
    num_steps: int = NUM_INFERENCE_STEPS,
) -> mx.array:
    """
    τ=1 (ノイズ) → τ=0 (クリーンアクション) を Euler 法で積分。
    戻り値: (1, action_horizon, action_dim)
    """
    B = pg_hidden.shape[0]
    x = mx.random.normal(shape=(B, ACTION_HORIZON, ACTION_DIM))

    timesteps = np.linspace(1.0, 0.0, num_steps + 1)

    for i in range(num_steps):
        t_curr = float(timesteps[i])
        t_next = float(timesteps[i + 1])
        dt     = t_next - t_curr          # 負値 (1→0)

        tau = mx.array([t_curr] * B)      # (B,)
        v   = expert_model(pg_hidden, x, tau)   # (B, T_action, action_dim)
        x   = x + dt * v
        mx.eval(x)

    return x   # (B, action_horizon, action_dim)


# ═══════════════════════════════════════════════════════════════
# Pi05Policy: メインクラス
# ═══════════════════════════════════════════════════════════════

class Pi05Policy:
    """
    MLX版 π₀.₅ 推論クラス。

    入力 (observation):
        {
            "state":  np.ndarray (state_dim,)          現在の状態
            "images": {
                "base_0_rgb":       np.ndarray (H,W,3) uint8 BGR
                "left_wrist_0_rgb": np.ndarray (H,W,3) uint8 BGR  # optional
                ...
            }
            "task":   str                               言語指示
        }

    出力:
        np.ndarray (action_horizon, action_dim)
    """

    def __init__(
        self,
        pi05_repo: str,
        norm_stats: Optional[Dict] = None,
        tokenizer_name: str = "google/paligemma-3b-pt-224",
    ):
        self.norm_stats = norm_stats
        self._load_paligemma(pi05_repo)
        self._load_expert(pi05_repo)
        self.preprocessor = Preprocessor(tokenizer_name)

    def _load_paligemma(self, pi05_repo: str):
        from mlx_vlm.models.paligemma import paligemma as pg_module
        from mlx_vlm.models.paligemma.paligemma import ModelConfig
        from mlx_vlm.utils import load_config
        import json

        repo_path = Path(pi05_repo)
        print(f"Loading PaliGemma from {pi05_repo} ...")

        # config 読み込み
        config_dict = {
            "model_type": "paligemma",
            "text_config": {
                "model_type": "gemma",
                "hidden_size": 2048,
                "intermediate_size": 16384,
                "num_hidden_layers": 18,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
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
        model_config = ModelConfig.from_dict(config_dict)

        # モデル構築
        self.pg_model = pg_module.Model(model_config)

        # pi05_base_mlx_bf16 の重みからPaliGemma部分を抽出してロード
        weights_path = str(repo_path / "model.safetensors")
        raw = mx.load(weights_path)

        PREFIX = "paligemma_with_expert.paligemma.model."
        remapped = {}
        for key, val in raw.items():
            if not key.startswith(PREFIX):
                continue
            new_key = key[len(PREFIX):]
            if new_key.startswith("language_model."):
                new_key = "language_model.model." + new_key[len("language_model."):]
            remapped[new_key] = val

        # embed_tokens の tied weights 補完
        LM_HEAD = "paligemma_with_expert.paligemma.lm_head.weight"
        if "language_model.model.embed_tokens.weight" not in remapped and LM_HEAD in raw:
            remapped["language_model.model.embed_tokens.weight"] = raw[LM_HEAD]

        self.pg_model.load_weights(list(remapped.items()), strict=False)
        mx.eval(self.pg_model.parameters())

        # processor はトークナイザーだけ必要
        self.pg_processor = None  # extract_paligemma_hidden では使っていない
        print("PaliGemma loaded.")

    def _load_expert(self, pi05_repo: str):
        print("Building GemmaExpert ...")
        self.expert_model = Pi05Phase2()
        load_expert_weights(self.expert_model, pi05_repo)
        mx.eval(self.expert_model.parameters())
        print("GemmaExpert loaded.")

    # ── 状態の正規化 + パディング ──────────────────────────────

    def _prepare_state(self, state: np.ndarray) -> np.ndarray:
        """state_dim → MAX_STATE_DIM にパディング後、QUANTILES 正規化"""
        padded = np.zeros(MAX_STATE_DIM, dtype=np.float32)
        dim = min(len(state), MAX_STATE_DIM)
        padded[:dim] = state[:dim]

        if self.norm_stats and "state" in self.norm_stats:
            q01 = np.array(self.norm_stats["state"]["q01"])
            q99 = np.array(self.norm_stats["state"]["q99"])
            padded = Preprocessor.normalize_quantile(padded, q01, q99)

        return padded

    # ── アクションの逆正規化 ────────────────────────────────────

    def _postprocess_action(self, actions: np.ndarray) -> np.ndarray:
        if self.norm_stats and "action" in self.norm_stats:
            q01 = np.array(self.norm_stats["action"]["q01"])
            q99 = np.array(self.norm_stats["action"]["q99"])
            actions = Preprocessor.unnormalize_quantile(actions, q01, q99)
        return actions

    # ── メイン推論 ───────────────────────────────────────────────

    def select_action(self, observation: Dict) -> np.ndarray:
        """
        observation → actions (action_horizon, action_dim)

        observation キー:
          "state"  : np.ndarray (state_dim,)
          "images" : dict[str, np.ndarray (H,W,3) uint8 BGR]
          "task"   : str
        """
        state  = observation.get("state", np.zeros(MAX_STATE_DIM, dtype=np.float32))
        images = observation.get("images", {})
        task   = observation.get("task", "")

        # 1. 状態の前処理
        norm_state = self._prepare_state(state)

        # 2. 画像の前処理 (CAMERA_KEYS の順に最大3枚)
        processed_images = []
        for key in CAMERA_KEYS:
            if key in images:
                processed_images.append(
                    self.preprocessor.preprocess_image(images[key])
                )
            if len(processed_images) == 3:
                break

        if not processed_images:
            raise ValueError(
                f"observation['images'] に有効なキーがありません。"
                f"利用可能なキー: {CAMERA_KEYS}"
            )

        # 3. テキスト + 離散化状態をトークン化
        tokens = self.preprocessor.tokenize(task, norm_state)

        # 4. PaliGemma で prefix hidden states を取得
        pg_hidden = extract_paligemma_hidden(
            pg_model  = self.pg_model,
            images    = processed_images,
            token_ids = tokens["input_ids"],
        )
        mx.eval(pg_hidden)

        # 5. Flow Matching denoising (10 ステップ)
        actions_mlx = flow_matching_sample(
            expert_model = self.expert_model,
            pg_hidden    = pg_hidden,
        )

        # 6. numpy 変換 + 逆正規化
        actions = np.array(actions_mlx[0])           # (action_horizon, action_dim)
        actions = self._postprocess_action(actions)

        # 7. action_dim をロボットの実際の次元にトリム
        actual_dim = len(state) if len(state) <= MAX_ACTION_DIM else MAX_ACTION_DIM
        actions = actions[:, :actual_dim]

        return actions   # (action_horizon, actual_action_dim)


# ═══════════════════════════════════════════════════════════════
# 動作確認
# ═══════════════════════════════════════════════════════════════

def verify(pi05_repo_or_path: str, image_url: str):
    print("\n=== π₀.₅ MLX: select_action テスト ===\n")

    policy = Pi05Policy(
        pi05_repo = pi05_repo_or_path,   # Action Expert 等フル重み
    )

    # Web から画像取得
    print(f"Downloading: {image_url}")
    tmp = "/tmp/pi05_test.jpg"
    urllib.request.urlretrieve(image_url, tmp)
    img_bgr = cv2.imread(tmp)
    assert img_bgr is not None

    # observation を構築
    STATE_DIM  = 14   # 例: シングルアーム 6 joint + gripper + ...
    observation = {
        "state": np.random.uniform(-0.5, 0.5, size=(STATE_DIM,)).astype(np.float32),
        "images": {
            "base_0_rgb": img_bgr,
        },
        "task": "pick up the object on the table",
    }

    print(f"\nObservation:")
    print(f"  state shape : {observation['state'].shape}")
    print(f"  image keys  : {list(observation['images'].keys())}")
    print(f"  task        : {observation['task']}")

    print("\nRunning select_action ...")
    actions = policy.select_action(observation)

    print(f"\n=== 結果 ===")
    print(f"actions.shape : {actions.shape}")
    print(f"actions[0]    : {np.round(actions[0], 4)}")
    print(f"actions[-1]   : {np.round(actions[-1], 4)}")
    assert not np.isnan(actions).any(), "NaN を検出"
    print("\nOK: select_action が正常に完了しました。")


if __name__ == "__main__":
    PI05_REPO       = "./models/FIwaki/pi05_base_mlx_bf16"
    IMAGE_URL       = "http://images.cocodataset.org/val2017/000000039769.jpg"

    verify(
        pi05_repo_or_path=PI05_REPO,
        image_url=IMAGE_URL,
    )