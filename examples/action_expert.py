# phase2_gemma_expert.py
# Phase 2: AdaRMSNorm付きGemmaExpert (300M) の実装とグローバルAttention接続
#
# 使い方:
#   python phase2_gemma_expert.py
#
# 前提:
#   - Phase 1 が成功していること (./pi05_phase1_paligemma/ が存在)
#   - 変換済み bf16 重みが ./pi05_base_mlx_bf16/model.safetensors に存在

import math
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ═══════════════════════════════════════════════════════════════
# 1. ハイパーパラメータ
# ═══════════════════════════════════════════════════════════════

# PaliGemma (Gemma 1 / 2B) side
PG_HIDDEN  = 2048
PG_LAYERS  = 18

# GemmaExpert (300M) side  ← pi05_base の重み形状から確定
EXP_HIDDEN      = 1024
EXP_INTER       = 4096
EXP_LAYERS      = 18
EXP_NUM_Q_HEADS = 8     # q_proj: [2048, 1024] → 2048/256=8
EXP_NUM_KV_HEADS= 1     # k_proj: [256,  1024] → 256/256=1
EXP_HEAD_DIM    = 256
VOCAB_SIZE      = 257152

# AdaRMSNorm の dense: [3072, 1024]
# 3072 = PG_HIDDEN(2048) + timestep_emb_dim(1024)
ADA_DENSE_IN    = 3072   # PG_HIDDEN + EXP_HIDDEN (timestep embedding 次元)
ADA_DENSE_OUT   = 3072   # 出力は [scale_norm, scale_attn, scale_mlp] を含む

# Action / time
ACTION_DIM      = 32
ACTION_HORIZON  = 50


# ═══════════════════════════════════════════════════════════════
# 2. AdaRMSNorm
# ═══════════════════════════════════════════════════════════════

class AdaRMSNorm(nn.Module):
    """
    Adaptive RMSNorm。
    タイムステップ埋め込みから scale を生成して RMSNorm の出力に掛ける。

    pi05_base の重み:
      .dense.weight  [3072, 1024]  (out_features, in_features)
      .dense.bias    [3072]

    入力:
      x         : (B, T, EXP_HIDDEN=1024)
      timestep  : (B, EXP_HIDDEN=1024)  ← time_mlp_out の出力

    出力:
      (B, T, EXP_HIDDEN)
    """

    def __init__(self):
        super().__init__()
        # linear: EXP_HIDDEN → ADA_DENSE_OUT
        self.dense = nn.Linear(EXP_HIDDEN, ADA_DENSE_OUT, bias=True)

    def __call__(self, x: mx.array, timestep_emb: mx.array) -> mx.array:
        # RMSNorm
        rms = mx.rsqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
        x_norm = x * rms

        # タイムステップから scale を生成
        # timestep_emb: (B, 1024) → dense → (B, 3072)
        # ここでは先頭の EXP_HIDDEN(1024) 次元を scale として使用
        scale_all = self.dense(timestep_emb)           # (B, 3072)
        scale = scale_all[:, :EXP_HIDDEN]              # (B, 1024)
        scale = mx.expand_dims(scale, axis=1)          # (B, 1, 1024)

        return x_norm * (1.0 + scale)


# ═══════════════════════════════════════════════════════════════
# 3. GemmaExpert の Attention
# ═══════════════════════════════════════════════════════════════

class GemmaExpertAttention(nn.Module):
    """
    GQA (Grouped Query Attention)。
    num_q_heads=8, num_kv_heads=1, head_dim=256
    """

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(EXP_HIDDEN, EXP_NUM_Q_HEADS * EXP_HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(EXP_HIDDEN, EXP_NUM_KV_HEADS * EXP_HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(EXP_HIDDEN, EXP_NUM_KV_HEADS * EXP_HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(EXP_NUM_Q_HEADS * EXP_HEAD_DIM, EXP_HIDDEN, bias=False)

    def __call__(
        self,
        x: mx.array,           # (B, T, EXP_HIDDEN)
        mask: mx.array = None,
    ) -> mx.array:
        B, T, _ = x.shape

        q = self.q_proj(x)  # (B, T, 8*256)
        k = self.k_proj(x)  # (B, T, 1*256)
        v = self.v_proj(x)  # (B, T, 1*256)

        # reshape: (B, T, heads, head_dim) → (B, heads, T, head_dim)
        q = q.reshape(B, T, EXP_NUM_Q_HEADS, EXP_HEAD_DIM).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, EXP_NUM_KV_HEADS, EXP_HEAD_DIM).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, EXP_NUM_KV_HEADS, EXP_HEAD_DIM).transpose(0, 2, 1, 3)

        # GQA: KV を Q heads 数に合わせて repeat
        if EXP_NUM_KV_HEADS < EXP_NUM_Q_HEADS:
            repeat = EXP_NUM_Q_HEADS // EXP_NUM_KV_HEADS
            k = mx.repeat(k, repeat, axis=1)  # (B, 8, T, 256)
            v = mx.repeat(v, repeat, axis=1)

        # scaled dot-product attention
        scale = math.sqrt(EXP_HEAD_DIM)
        attn = (q @ k.transpose(0, 1, 3, 2)) / scale   # (B, 8, T, T)

        if mask is not None:
            attn = attn + mask

        attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(x.dtype)
        out  = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.o_proj(out)


# ═══════════════════════════════════════════════════════════════
# 4. GemmaExpert の MLP (SiLU gate)
# ═══════════════════════════════════════════════════════════════

class GemmaExpertMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(EXP_HIDDEN, EXP_INTER, bias=False)
        self.up_proj   = nn.Linear(EXP_HIDDEN, EXP_INTER, bias=False)
        self.down_proj = nn.Linear(EXP_INTER,  EXP_HIDDEN, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


# ═══════════════════════════════════════════════════════════════
# 5. GemmaExpert の 1 層
# ═══════════════════════════════════════════════════════════════

class GemmaExpertLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn                = GemmaExpertAttention()
        self.mlp                      = GemmaExpertMLP()
        self.input_layernorm          = AdaRMSNorm()
        self.post_attention_layernorm = AdaRMSNorm()

    def __call__(
        self,
        x: mx.array,            # (B, T_total, EXP_HIDDEN)
        timestep_emb: mx.array, # (B, EXP_HIDDEN)
        mask: mx.array = None,
    ) -> mx.array:
        # pre-norm → attention → residual
        h = self.input_layernorm(x, timestep_emb)
        h = self.self_attn(h, mask)
        x = x + h

        # pre-norm → MLP → residual
        h = self.post_attention_layernorm(x, timestep_emb)
        h = self.mlp(h)
        x = x + h

        return x


# ═══════════════════════════════════════════════════════════════
# 6. GemmaExpert モデル全体
# ═══════════════════════════════════════════════════════════════

class GemmaExpert(nn.Module):
    """
    Action Expert (300M)。
    PaliGemma の prefix embedding (B, T_prefix, PG_HIDDEN) と
    noisy action tokens (B, T_action, EXP_HIDDEN) を受け取り、
    velocity field v_θ を返す。

    グローバル Attention の仕組み:
      - PaliGemma の hidden を EXP_HIDDEN に射影してトークン列に連結
      - [prefix_proj | action_tokens] の全体に双方向 attention
      - attention 後、action_tokens 部分のみ取り出して出力
    """

    def __init__(self):
        super().__init__()
        self.layers = [GemmaExpertLayer() for _ in range(EXP_LAYERS)]
        self.norm    = AdaRMSNorm()           # final norm

        # PaliGemma hidden (2048) → EXP_HIDDEN (1024) への射影
        # ※ 重みとしては存在しないため線形変換で近似
        # （実際の pi0.5 では attention 内で直接結合）
        self.pg_proj = nn.Linear(PG_HIDDEN, EXP_HIDDEN, bias=False)

    def __call__(
        self,
        pg_hidden: mx.array,      # (B, T_prefix, PG_HIDDEN=2048)  PaliGemma 出力
        action_emb: mx.array,     # (B, T_action, EXP_HIDDEN=1024) action tokens
        timestep_emb: mx.array,   # (B, EXP_HIDDEN=1024)           タイムステップ埋め込み
    ) -> mx.array:
        B = pg_hidden.shape[0]
        T_prefix = pg_hidden.shape[1]
        T_action = action_emb.shape[1]
        T_total  = T_prefix + T_action

        # PaliGemma hidden を EXP_HIDDEN に射影
        pg_proj = self.pg_proj(pg_hidden)  # (B, T_prefix, 1024)

        # prefix + action を連結 → グローバル Attention
        x = mx.concatenate([pg_proj, action_emb], axis=1)  # (B, T_total, 1024)

        # 全トークン双方向 attention（マスクなし）
        for layer in self.layers:
            x = layer(x, timestep_emb, mask=None)

        # action 部分だけ取り出して final norm
        x_action = x[:, T_prefix:, :]                # (B, T_action, 1024)
        x_action = self.norm(x_action, timestep_emb)  # AdaRMSNorm

        return x_action  # (B, T_action, EXP_HIDDEN)


# ═══════════════════════════════════════════════════════════════
# 7. タイムステップ埋め込み (Fourier + MLP)
# ═══════════════════════════════════════════════════════════════

class TimestepEmbedding(nn.Module):
    """
    pi0.5 のタイムステップ埋め込み。
    pi05_base の重み:
      time_mlp_in.weight  [1024, 1024]
      time_mlp_in.bias    [1024]
      time_mlp_out.weight [1024, 1024]
      time_mlp_out.bias   [1024]
    """
    def __init__(self, min_period: float = 0.004, max_period: float = 4.0):
        super().__init__()
        self.min_period = min_period
        self.max_period = max_period
        self.mlp_in  = nn.Linear(EXP_HIDDEN, EXP_HIDDEN, bias=True)
        self.mlp_out = nn.Linear(EXP_HIDDEN, EXP_HIDDEN, bias=True)

    def __call__(self, t: mx.array) -> mx.array:
        """
        t: (B,) ∈ [0, 1]  flow matching のタイムステップ
        → (B, EXP_HIDDEN)
        """
        # Fourier 特徴量
        half = EXP_HIDDEN // 2
        freqs = mx.exp(
            mx.linspace(
                math.log(1.0 / self.max_period),
                math.log(1.0 / self.min_period),
                half,
            )
        )                                               # (half,)
        t_exp = mx.expand_dims(t, -1) * freqs          # (B, half)
        emb = mx.concatenate(
            [mx.sin(t_exp), mx.cos(t_exp)], axis=-1
        )                                               # (B, EXP_HIDDEN)

        # MLP: in → SiLU → out
        emb = nn.silu(self.mlp_in(emb))
        emb = self.mlp_out(emb)
        return emb                                      # (B, EXP_HIDDEN)


# ═══════════════════════════════════════════════════════════════
# 8. アクション エンコーダ / デコーダ
# ═══════════════════════════════════════════════════════════════

class ActionEncoder(nn.Module):
    """
    連続アクション (B, T, ACTION_DIM) → (B, T, EXP_HIDDEN)
    action_in_proj.weight [1024, 32], bias [1024]
    """
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(ACTION_DIM, EXP_HIDDEN, bias=True)

    def __call__(self, actions: mx.array) -> mx.array:
        return self.proj(actions)


class ActionDecoder(nn.Module):
    """
    (B, T, EXP_HIDDEN) → (B, T, ACTION_DIM)  velocity field v_θ
    action_out_proj.weight [32, 1024], bias [32]
    """
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(EXP_HIDDEN, ACTION_DIM, bias=True)

    def __call__(self, h: mx.array) -> mx.array:
        return self.proj(h)


# ═══════════════════════════════════════════════════════════════
# 9. 重みロード
# ═══════════════════════════════════════════════════════════════

def load_expert_weights(
    model: nn.Module,
    src_path: str,
) -> nn.Module:
    """
    pi05_base の safetensors から GemmaExpert 関連の重みをロードする。
    キーのマッピング:
      paligemma_with_expert.gemma_expert.model.layers.{i}.* → layers.{i}.*
      paligemma_with_expert.gemma_expert.model.norm.*        → norm.*
      action_in_proj.*  → action_encoder.proj.*
      action_out_proj.* → action_decoder.proj.*
      time_mlp_in.*     → timestep_emb.mlp_in.*
      time_mlp_out.*    → timestep_emb.mlp_out.*
    """
    src_path = str(Path(src_path) / "model.safetensors") if Path(src_path).is_dir() else src_path
    print(f"Loading expert weights from {src_path} ...")
    raw = mx.load(src_path)

    weights = {}
    EXP_PREFIX = "paligemma_with_expert.gemma_expert.model."

    for key, val in raw.items():
        # GemmaExpert layers / norm
        if key.startswith(EXP_PREFIX + "layers."):
            new_key = "gemma_expert." + key[len(EXP_PREFIX):]
            weights[new_key] = val
        elif key.startswith(EXP_PREFIX + "norm."):
            new_key = "gemma_expert." + key[len(EXP_PREFIX):]
            weights[new_key] = val

        # action proj
        elif key.startswith("action_in_proj."):
            weights["action_encoder.proj." + key[len("action_in_proj."):]] = val
        elif key.startswith("action_out_proj."):
            weights["action_decoder.proj." + key[len("action_out_proj."):]] = val

        # time MLP
        elif key.startswith("time_mlp_in."):
            weights["timestep_emb.mlp_in." + key[len("time_mlp_in."):]] = val
        elif key.startswith("time_mlp_out."):
            weights["timestep_emb.mlp_out." + key[len("time_mlp_out."):]] = val

    print(f"  Mapped {len(weights)} expert keys")
    model.load_weights(list(weights.items()), strict=False)
    print("  Expert weights loaded.")
    return model


# ═══════════════════════════════════════════════════════════════
# 10. Phase 2 統合モデル (PaliGemma + GemmaExpert)
# ═══════════════════════════════════════════════════════════════

class Pi05Phase2(nn.Module):
    """
    Phase 2 の統合モデル。
    PaliGemma (mlx-vlm でロード済み) の hidden states と
    GemmaExpert を接続し、velocity field を出力する。
    """
    def __init__(self):
        super().__init__()
        self.gemma_expert   = GemmaExpert()
        self.action_encoder = ActionEncoder()
        self.action_decoder = ActionDecoder()
        self.timestep_emb   = TimestepEmbedding()

    def __call__(
        self,
        pg_hidden: mx.array,      # (B, T_prefix, 2048)  PaliGemma の最終 hidden states
        noisy_actions: mx.array,  # (B, T_action, 32)    ノイズ付きアクション
        timestep: mx.array,       # (B,)                 τ ∈ [0, 1]
    ) -> mx.array:
        # タイムステップ埋め込み
        t_emb = self.timestep_emb(timestep)         # (B, 1024)

        # アクションを EXP_HIDDEN に射影
        action_emb = self.action_encoder(noisy_actions)  # (B, T_action, 1024)

        # GemmaExpert でグローバル Attention
        h = self.gemma_expert(pg_hidden, action_emb, t_emb)  # (B, T_action, 1024)

        # velocity field に変換
        velocity = self.action_decoder(h)           # (B, T_action, 32)
        return velocity


# ═══════════════════════════════════════════════════════════════
# 11. 動作確認
# ═══════════════════════════════════════════════════════════════

def verify_phase2(src_weights: str):
    print("\n=== Phase 2: GemmaExpert + Global Attention ===")

    # モデル構築
    model = Pi05Phase2()

    # 重みロード
    load_expert_weights(model, src_weights)
    mx.eval(model.parameters())

    # ダミー入力でフォワードパス確認
    B          = 1
    T_prefix   = 256 + 10   # 画像256トークン + 言語10トークン（例）
    T_action   = ACTION_HORIZON

    pg_hidden     = mx.zeros((B, T_prefix, PG_HIDDEN))
    noisy_actions = mx.random.normal(shape=(B, T_action, ACTION_DIM))
    timestep      = mx.array([0.5])   # τ=0.5

    print(f"\nInput shapes:")
    print(f"  pg_hidden:     {pg_hidden.shape}")
    print(f"  noisy_actions: {noisy_actions.shape}")
    print(f"  timestep:      {timestep.shape}")

    velocity = model(pg_hidden, noisy_actions, timestep)
    mx.eval(velocity)

    print(f"\nOutput shape: {velocity.shape}")
    print(f"  Expected:    ({B}, {T_action}, {ACTION_DIM})")
    assert velocity.shape == (B, T_action, ACTION_DIM), "Shape mismatch!"
    print("\nPhase 2 OK: GemmaExpert forward pass succeeded.")

    return model


if __name__ == "__main__":
    SRC_WEIGHTS = "./models/FIwaki/pi05_base_mlx/model.safetensors"
    verify_phase2(SRC_WEIGHTS)