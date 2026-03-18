# phase3_select_action.py
# Phase 3: Flow Matching 推論ループ + 前後処理 → select_action 完成
#
# 使い方:
#   python phase3_select_action.py
#
# 前提:
#   - Phase 1 成功: ./pi05_phase1_paligemma/ が存在
#   - Phase 2 成功: phase2_gemma_expert.py が同ディレクトリに存在
#   - 変換済み bf16 重み: ./pi05_base_mlx_bf16/model.safetensors

import math
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import urllib.request
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

# Phase 2 のモジュールをインポート
from 02_action_expert import (
    Pi05Phase2,
    load_expert_weights,
    ACTION_DIM,
    ACTION_HORIZON,
    PG_HIDDEN,
)


# ═══════════════════════════════════════════════════════════════
# 定数
# ═══════════════════════════════════════════════════════════════

NUM_INFERENCE_STEPS  = 10      # Flow Matching denoising ステップ数
MAX_STATE_DIM        = 32      # config.json より
MAX_ACTION_DIM       = 32
STATE_BINS           = 256     # pi0.5: ロボット状態の離散化ビン数
IMAGE_SIZE           = 224
MAX_TOKEN_LEN        = 200

# SigLIP の正規化パラメータ (ImageNet 準拠)
SIGLIP_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SIGLIP_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# 1. 前処理
# ═══════════════════════════════════════════════════════════════

class Preprocessor:
    """
    カメラ画像・ロボット状態・言語指示 → モデル入力に変換する。

    pi0.5 の前処理:
      - 画像: リサイズ → SigLIP 正規化 → [B, H, W, C] (0~1 → -1~1)
      - 状態: QUANTILES 正規化 → 256 ビン離散化 → 言語トークンに付加
      - テキスト: Gemma tokenizer でトークン化
    """

    def __init__(self, tokenizer_name: str = "google/paligemma-3b-pt-224"):
        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # ── 画像前処理 ──────────────────────────────────────────────

    def preprocess_image(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        OpenCV BGR 画像 → SigLIP 入力
        戻り値: (H, W, C) float32 in [-1, 1]
        """
        # リサイズ
        img = cv2.resize(img_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        # BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # [0,255] → [0,1]
        img = img.astype(np.float32) / 255.0
        # SigLIP 正規化: (x - 0.5) / 0.5 → [-1, 1]
        img = (img - SIGLIP_MEAN) / SIGLIP_STD
        return img  # (H, W, C)

    # ── 状態の離散化 (pi0.5 固有) ────────────────────────────────

    @staticmethod
    def discretize_state(
        state: np.ndarray,          # (state_dim,) in [-1, 1] (正規化済み)
        n_bins: int = STATE_BINS,
    ) -> str:
        """
        連続状態値を STATE_BINS ビンの離散トークン列に変換して文字列で返す。
        例: state=[0.1, -0.5, ...] → "s023 s128 ..."

        pi0.5 論文: 状態を 256 ビンに離散化し言語プロンプトに組み込む。
        """
        # [-1, 1] → [0, n_bins-1]
        bins = np.clip(
            ((state + 1.0) / 2.0 * n_bins).astype(int),
            0, n_bins - 1
        )
        return " ".join(f"s{b:03d}" for b in bins)

    # ── テキスト + 状態トークン化 ────────────────────────────────

    def tokenize(
        self,
        task: str,
        state: Optional[np.ndarray] = None,  # (state_dim,) 正規化済み
        max_length: int = MAX_TOKEN_LEN,
    ) -> Dict[str, np.ndarray]:
        """
        言語指示 [+ 離散化状態] を Gemma tokenizer でトークン化する。
        戻り値: {"input_ids": (max_length,), "attention_mask": (max_length,)}
        """
        if state is not None:
            state_str = self.discretize_state(state)
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
            "input_ids":      enc["input_ids"][0],       # (max_length,)
            "attention_mask": enc["attention_mask"][0],  # (max_length,)
        }

    # ── QUANTILES 正規化 ─────────────────────────────────────────

    @staticmethod
    def normalize_quantile(
        x: np.ndarray,
        q01: np.ndarray,
        q99: np.ndarray,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        x を [q01, q99] の範囲で [-1, 1] に正規化する。
        policy_preprocessor.json: STATE / ACTION は QUANTILES 正規化。
        """
        return np.clip(2.0 * (x - q01) / (q99 - q01 + eps) - 1.0, -1.0, 1.0)

    @staticmethod
    def unnormalize_quantile(
        x: np.ndarray,
        q01: np.ndarray,
        q99: np.ndarray,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """normalize_quantile の逆変換。"""
        return (x + 1.0) / 2.0 * (q99 - q01 + eps) + q01


# ═══════════════════════════════════════════════════════════════
# 2. PaliGemma の hidden states 抽出
# ═══════════════════════════════════════════════════════════════

def extract_paligemma_hidden(
    pg_model,
    pg_processor,
    images: List[np.ndarray],
    token_ids: np.ndarray,
    attention_mask: np.ndarray,
) -> mx.array:
    vision_tower = pg_model.vision_tower
    projector    = pg_model.multi_modal_projector
    lm           = pg_model.language_model

    # 画像 embedding
    img_embeds_list = []
    for img in images:
        pv = mx.array(img[None])  # (1, H, W, C)
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
        img_embeds_list.append(img_embed)

    img_embeds = mx.concatenate(img_embeds_list, axis=1)  # (1, N*256, 2048)

    # テキスト embedding
    ids = mx.array(token_ids[None].astype(np.int32))  # (1, T)
    text_embeds = lm.model.embed_tokens(ids)           # (1, T, 2048)

    # 結合
    combined = mx.concatenate([img_embeds, text_embeds], axis=1)  # (1, T_total, 2048)

    # Gemmaのlayersを直接ループ
    h = combined
    for layer in lm.model.layers:
        h = layer(h)

    # 最終 norm
    if hasattr(lm.model, "norm"):
        h = lm.model.norm(h)

    return h  # (1, T_total, 2048)


# ═══════════════════════════════════════════════════════════════
# 3. Flow Matching 推論ループ
# ═══════════════════════════════════════════════════════════════

def flow_matching_sample(
    expert_model: Pi05Phase2,
    pg_hidden: mx.array,         # (1, T_prefix, 2048)
    num_steps: int = NUM_INFERENCE_STEPS,
    action_horizon: int = ACTION_HORIZON,
    action_dim: int = ACTION_DIM,
) -> mx.array:
    """
    Flow Matching の denoising ループ。
    τ=1 (純粋なノイズ) から τ=0 (クリーンなアクション) へ積分する。

    ODE: dx/dτ = -v_θ(x, τ)   ← τ を 1→0 に逆行するため符号反転
    Euler 法: x_{τ-Δτ} = x_τ - Δτ * v_θ(x_τ, τ)

    pi0.5 config より:
      time_sampling_beta_alpha = 1.5
      time_sampling_beta_beta  = 1.0
      → 推論時は等間隔タイムステップを使用
    """
    B = pg_hidden.shape[0]

    # τ=1 からノイズで初期化
    x = mx.random.normal(shape=(B, action_horizon, action_dim))

    # タイムステップ: 1.0 → 0.0 を num_steps 等分
    timesteps = np.linspace(1.0, 0.0, num_steps + 1)

    for i in range(num_steps):
        t_curr = float(timesteps[i])
        t_next = float(timesteps[i + 1])
        dt     = t_next - t_curr          # 負の値 (1→0 方向)

        tau = mx.array([t_curr] * B)      # (B,)

        # velocity field を予測
        v = expert_model(pg_hidden, x, tau)   # (B, T_action, action_dim)

        # Euler ステップ
        x = x + dt * v

        mx.eval(x)

    return x   # (B, action_horizon, action_dim)  クリーンなアクション


# ═══════════════════════════════════════════════════════════════
# 4. select_action: 全体パイプライン
# ═══════════════════════════════════════════════════════════════

class Pi05Policy:
    """
    MLX 版 π₀.₅ の推論クラス。

    select_action(observation) → actions (action_horizon, action_dim)

    observation の形式:
        {
            "images": {
                "base_0_rgb":        np.ndarray (H, W, 3) uint8 BGR,
                "left_wrist_0_rgb":  np.ndarray (H, W, 3) uint8 BGR,  # optional
                "right_wrist_0_rgb": np.ndarray (H, W, 3) uint8 BGR,  # optional
            },
            "state": np.ndarray (state_dim,)  生の関節角度など,
            "task":  str  自然言語タスク指示,
        }
    """

    def __init__(
        self,
        phase1_dir: str,
        src_weights: str,
        norm_stats: Optional[Dict] = None,
    ):
        self._load_paligemma(phase1_dir)
        self._load_expert(src_weights)
        self.preprocessor = Preprocessor()
        self.norm_stats    = norm_stats  # {"state": {"q01":..,"q99":..}, "action": {...}}

    # ── モデルロード ────────────────────────────────────────────

    def _load_paligemma(self, phase1_dir: str):
        from mlx_vlm import load
        print(f"Loading PaliGemma from {phase1_dir} ...")
        self.pg_model, self.pg_processor = load(str(Path(phase1_dir).resolve()))
        mx.eval(self.pg_model.parameters())
        print("PaliGemma loaded.")

    def _load_expert(self, src_weights: str):
        print("Building GemmaExpert ...")
        self.expert_model = Pi05Phase2()
        load_expert_weights(self.expert_model, src_weights)
        mx.eval(self.expert_model.parameters())
        print("GemmaExpert loaded.")

    # ── 前処理 ──────────────────────────────────────────────────

    def _preprocess(self, observation: Dict) -> Dict:
        images_dict = observation["images"]
        state       = observation.get("state", None)
        task        = observation.get("task", "")

        # 画像前処理
        processed_images = []
        for key in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
            if key in images_dict:
                img = self.preprocessor.preprocess_image(images_dict[key])
                processed_images.append(img)

        if not processed_images:
            raise ValueError("observation['images'] に有効な画像キーがありません")

        # 状態の正規化
        norm_state = None
        if state is not None:
            # パディング: state_dim → MAX_STATE_DIM
            padded = np.zeros(MAX_STATE_DIM, dtype=np.float32)
            padded[:min(len(state), MAX_STATE_DIM)] = state[:MAX_STATE_DIM]

            if self.norm_stats and "state" in self.norm_stats:
                q01 = np.array(self.norm_stats["state"]["q01"])
                q99 = np.array(self.norm_stats["state"]["q99"])
                padded = Preprocessor.normalize_quantile(padded, q01, q99)

            norm_state = padded

        # トークン化 (状態を言語プロンプトに組み込む)
        tokens = self.preprocessor.tokenize(task, norm_state)

        return {
            "images":   processed_images,
            "tokens":   tokens,
            "state":    norm_state,
        }

    # ── 後処理 ──────────────────────────────────────────────────

    def _postprocess(self, actions: np.ndarray) -> np.ndarray:
        """
        正規化されたアクション → 生のアクション値に変換。
        norm_stats がない場合はそのまま返す。
        """
        if self.norm_stats and "action" in self.norm_stats:
            q01 = np.array(self.norm_stats["action"]["q01"])
            q99 = np.array(self.norm_stats["action"]["q99"])
            # actions: (action_horizon, action_dim)
            actions = Preprocessor.unnormalize_quantile(actions, q01, q99)
        return actions

    # ── メインの推論 ─────────────────────────────────────────────

    def select_action(self, observation: Dict) -> np.ndarray:
        """
        observation → actions (action_horizon, action_dim)
        """
        # 1. 前処理
        processed = self._preprocess(observation)

        # 2. PaliGemma で prefix hidden states を取得
        pg_hidden = extract_paligemma_hidden(
            pg_model      = self.pg_model,
            pg_processor  = self.pg_processor,
            images        = processed["images"],
            token_ids     = processed["tokens"]["input_ids"],
            attention_mask= processed["tokens"]["attention_mask"],
        )
        mx.eval(pg_hidden)

        # 3. Flow Matching denoising
        actions_mlx = flow_matching_sample(
            expert_model   = self.expert_model,
            pg_hidden      = pg_hidden,
            num_steps      = NUM_INFERENCE_STEPS,
            action_horizon = ACTION_HORIZON,
            action_dim     = ACTION_DIM,
        )

        # 4. numpy に変換
        actions_np = np.array(actions_mlx[0])  # (action_horizon, action_dim)

        # 5. 後処理 (denormalize)
        actions_np = self._postprocess(actions_np)

        return actions_np   # (action_horizon, action_dim)


# ═══════════════════════════════════════════════════════════════
# 5. 動作確認
# ═══════════════════════════════════════════════════════════════

def verify_phase3(phase1_dir: str, src_weights: str, image_url: str):
    print("\n=== Phase 3: select_action エンドツーエンドテスト ===\n")

    # ── モデルロード ──
    policy = Pi05Policy(
        phase1_dir  = phase1_dir,
        src_weights = src_weights,
        norm_stats  = None,   # base モデルなので norm_stats なし
    )

    # ── テスト画像をWebから取得 ──
    print(f"Downloading test image from {image_url} ...")
    tmp = "/tmp/_phase3_test.jpg"
    urllib.request.urlretrieve(image_url, tmp)
    img_bgr = cv2.imread(tmp)
    assert img_bgr is not None, "画像の読み込みに失敗"
    print(f"Image shape: {img_bgr.shape}")

    # ── ダミー observation を構築 ──
    observation = {
        "images": {
            "base_0_rgb": img_bgr,
        },
        "state": np.random.uniform(-0.5, 0.5, size=(32,)).astype(np.float32),
        "task":  "pick up the object on the table",
    }

    # ── select_action を実行 ──
    print("\nRunning select_action ...")
    actions = policy.select_action(observation)

    print(f"\n=== 結果 ===")
    print(f"actions.shape: {actions.shape}")
    print(f"  期待値:      ({ACTION_HORIZON}, {ACTION_DIM})")
    print(f"actions[0]:   {actions[0]}")
    print(f"actions[-1]:  {actions[-1]}")

    assert actions.shape == (ACTION_HORIZON, ACTION_DIM), \
        f"Shape mismatch: {actions.shape}"
    assert not np.isnan(actions).any(), "NaN を検出"

    print("\nPhase 3 OK: select_action が正常に動作しました。")
    return policy


if __name__ == "__main__":
    PHASE1_DIR   = "./examples/outputs"
    SRC_WEIGHTS  = "./models/FIwaki/pi05_base_mlx/model.safetensors"
    IMAGE_URL    = "http://images.cocodataset.org/val2017/000000039769.jpg"

    verify_phase3(PHASE1_DIR, SRC_WEIGHTS, IMAGE_URL)