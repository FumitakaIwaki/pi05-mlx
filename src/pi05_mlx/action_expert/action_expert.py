import logging
import math
import numpy as np
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .ada_rms_norm import AdaRMSNorm
from .config import ActionExpertConfig
from .encoder_decoder import ActionEncoder, ActionDecoder
from .timtestep_embedding import TimestepEmbedding


class ActionExpertAttention(nn.Module):
    """GQA (Grouped Query Attention)
    num_q_heads=8, num_kv_heads=1, head_dim=256
    """

    def __init__(
        self,
        hidden: int,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
    ):
        super().__init__()
        self.hidden = hidden
        self.head_dim = head_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads

        self.q_proj = nn.Linear(
            hidden,
            num_q_heads * head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            hidden,
            num_kv_heads * head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            hidden,
            num_kv_heads * head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            num_q_heads * head_dim,
            hidden,
            bias=False,
        )
    
    def __call__(
        self,
        x: mx.array,    # (B, T, HIDDEN)
        mask: mx.array | None = None,
    ) -> mx.array:
        B, T, _ = x.shape

        q = self.q_proj(x)  # (B, T, 8*256)
        k = self.k_proj(x)  # (B, T, 1*256)
        v = self.v_proj(x)  # (B, T, 1*256)

        # reshape: (B, T, heads, head_dim) → (B, heads, T, head_dim)
        q = q.reshape(
                B, T, self.num_q_heads, self.head_dim
            ).transpose(0, 2, 1, 3)
        k = k.reshape(
                B, T, self.num_kv_heads, self.head_dim
            ).transpose(0, 2, 1, 3)
        v = v.reshape(
                B, T, self.num_kv_heads, self.head_dim
            ).transpose(0, 2, 1, 3)
        
        # GQA: KVを Q heads の数に合わせて repeat
        if self.num_kv_heads < self.num_q_heads:
            repeat = self.num_q_heads // self.num_kv_heads
            k = mx.repeat(k, repeat, axis=1)    # (B, 8, T, 256)
            v = mx.repeat(v, repeat, axis=1)
        
        # scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(0, 1, 3, 2)) / scale    # (B, 8, T, T)

        if mask is not None:
            attn = attn + mask
        
        attn = mx.softmax(
            attn.astype(mx.float32), axis=-1
        ).astype(x.dtype)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, -1)

        return self.o_proj(out)


class ActionExpertMLP(nn.Module):
    def __init__(
        self,
        hidden: int,
        inter: int,
    ):
        super().__init__()
        self.hidden = hidden
        self.inter = inter

        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(
            nn.silu(self.gate_proj(x) * self.up_proj(x))
        )


class ActionExpertLayer(nn.Module):
    def __init__(self, cfg: ActionExpertConfig):
        super().__init__()
        self.self_attn = ActionExpertAttention(
            hidden=cfg.hidden,
            head_dim=cfg.head_dim,
            num_q_heads=cfg.num_q_heads,
            num_kv_heads=cfg.num_kv_heads,
        )
        self.mlp = ActionExpertMLP(
            hidden=cfg.hidden,
            inter=cfg.inter,
        )
        self.input_layer_norm = AdaRMSNorm(
            hidden=cfg.hidden,
            ada_dense_out=cfg.ada_dense_out,
        )
        self.post_attn_layer_norm = AdaRMSNorm(
            hidden=cfg.hidden,
            ada_dense_out=cfg.ada_dense_out,
        )
    
    def __call__(
        self,
        x: mx.array,
        timestep_emb: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        # pre-norm → attention → residual
        h = self.input_layer_norm(x, timestep_emb)
        h = self.self_attn(h, mask)
        x = x + h

        # pre-norm → MLP → residual
        h = self.post_attn_layer_norm(x, timestep_emb)
        h = self.mlp(h)
        x = x + h

        return x


class ActionExpert(nn.Module):
    """Action Expert
    Inputs:
        - prefix embedding of PaliGemma (B, T_prefix, PG_HIDDEN)
        - noisy action tokens (B, T_action, EXP_HIDDEN)
    
    Outputs:
        - velocity field `v_0`
    
    Architecture:
        - PaliGemma の hidden を EXP_HIDDEN に射影してトークン列に連結
        - [prefix_proj | action_tokens] の全体に双方向 attention
        - action_tokens 部分のみ取り出して出力
    """

    def __init__(self, cfg: ActionExpertConfig):
        super().__init__()
        self.layers = [ActionExpertLayer(cfg) for _ in range(cfg.layers)]
        self.norm = AdaRMSNorm(
            hidden=cfg.hidden,
            ada_dense_out=cfg.ada_dense_out,
        )
        # PaliGemma hidden (2048) → EXP_HIDDEN (1024)
        self.pg_proj = nn.Linear(cfg.pg_hidden, cfg.hidden, bias=False)
    
    def __call__(
            self,
            pg_hidden: mx.array,    # (B, T_prefix, PG_HIDDEN=2048)
            action_emb: mx.array,   # (B, T_action, EXP_HIDDEN=1024)
            timestep_emb: mx.array, # (B, EXP_HIDDEN=1024)
    ) -> mx.array:
        T_prefix = pg_hidden.shape[1]

        pg_proj = self.pg_proj(pg_hidden)   # (B, T_prefix, 1024)

        x = mx.concatenate([pg_proj, action_emb], axis=1)   # (B, T_prefix + T_action, 1024)

        for layer in self.layers:
            x = layer(x, timestep_emb, mask=None)
        
        x_action = x[:, T_prefix:, :]   # (B, T_action, 1024)
        x_action = self.norm(x_action, timestep_emb)

        return x_action     # (B, T_action, EXP_HIDDEN)


class PI05ActionExpert(nn.Module):
    """ActionExpert totality model"""

    def __init__(
        self,
        cfg: ActionExpertConfig,
        logger: logging.Logger | None = logging.getLogger(__file__),
    ):
        super().__init__()
        self.logger = logger
        self.cfg = cfg
        self.expert = ActionExpert(cfg=cfg)
        self.action_encoder = ActionEncoder(
            action_dim=cfg.action_dim,
            hidden=cfg.hidden,
        )
        self.action_decoder = ActionDecoder(
            action_dim=cfg.action_dim,
            hidden=cfg.hidden,
        )
        self.timestep_emb = TimestepEmbedding(
            hidden=cfg.hidden,
            min_period=0.004,
            max_period=4.0,
        )
    
    def __call__(
        self,
        pg_hidden: mx.array,    # (N, T_prefix, 2048)
        noisy_actions: mx.array,    # (B, T_action, 32)
        timestep: mx.array,     # (B,) 
    ) -> mx.array:
        t_emb = self.timestep_emb(timestep)

        action_emb = self.action_encoder(noisy_actions)

        h = self.expert(pg_hidden, action_emb, t_emb)

        velocity = self.action_decoder(h)

        return velocity

    def _load_expert_weights(
        self,
        repo_id: str,
    ):
        """
        Load weights for action expert from safetensors
        key mapping:
            paligemma_with_expert.gemma_expert.model.layers.{i}.* → layers.{i}.*
            paligemma_with_expert.gemma_expert.model.norm.*        → norm.*
            action_in_proj.*  → action_encoder.proj.*
            action_out_proj.* → action_decoder.proj.*
            time_mlp_in.*     → timestep_emb.mlp_in.*
            time_mlp_out.*    → timestep_emb.mlp_out.*
        """
        src_path = str(Path(repo_id) / "model.safetensors") if Path(repo_id).is_dir() else repo_id
        
        self.logger.info(f"Loading expert weights from {src_path}...")
        raw = mx.load(src_path)

        weights = {}
        prefix = "paligemma_with_expert.gemma_expert.model."

        for key, val in raw.items():
            # GemmaExpert layers / norm
            if key.startswith(prefix + "layers."):
                new_key = "gemma_expert." + key[len(prefix):]
                weights[new_key] = val
            elif key.startswith(prefix + "norm."):
                new_key = "gemma_expert." + key[len(prefix):]
                weights[new_key] = val
            # action proj
            elif key.startswith("action_in_proj."):
                weights["action_encoder.proj." + key[len("action_in_proj."):]] = val
            elif key.startswith("action_out_proj."):
                weights["action_decoder.proj." + key[len("action_out_proj."):]] = val
            #time MLP
            elif key.startswith("time_mlp_in."):
                weights["timestep_emb.mlp_in." + key[len("time_mlp_in."):]] = val
            elif key.startswith("time_mlp_out."):
                weights["timestep_emb.mlp_out." + key[len("time_mlp_out."):]] = val
        
        self.logger.info(f"  Mapped {len(weights)} expert keys")
        self.load_weights(list(weights.items()), strict=False)
        self.logger.info("  Action expert weights loaded.")

    def flow_matching_sample(
        self,
        hidden: mx.array,
        num_steps: int,
    ) -> mx.array:
        """
        τ=1 (noise) → τ=0 (action) : integrate using Euler method.
        Return:
            (1, action_horizon, action_dim)
        """
        B = hidden.shape[0]
        x = mx.random.normal(shape=(
            B,
            self.cfg.action_horizon,
            self.cfg.action_dim
        ))

        timesteps = np.linspace(1.0, 0.0, num_steps + 1)

        for i in range(num_steps):
            t_curr = float(timesteps[i])
            t_next = float(timesteps[i + 1])
            dt = t_next - t_curr

            tau = mx.array([t_curr] * B)
            v = self(hidden, x, tau)
            x = x + dt * v
            mx.eval(x)
        
        return x
