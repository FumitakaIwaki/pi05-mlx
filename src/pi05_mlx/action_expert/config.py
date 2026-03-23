from dataclasses import dataclass


@dataclass
class ActionExpertConfig:
    # PaliGemma (Gemma 1 / 2B) side
    pg_hidden: int = 2048
    pg_layers: int = 18

    # GemmaExpert (300M)
    hidden: int = 1024
    inter: int = 4096
    layers: int = 18
    num_q_heads: int = 8  # q_proj: [2048, 1024] → 2048/256 = 8
    num_kv_heads: int = 1  # k_proj: [256, 1024] → 256/256 = 1
    head_dim: int = 256
    vocab_size: int = 257152

    # Dense of AdaRMSNorm: [3072, 1024]
    # 3072 = pg_hidden (2048) + timestep_emb_dim (1024)
    ada_dense_in: int = 3072  # pg_hidden + hidden (timestep embedding dimension)
    ada_dense_out: int = 3072  # Include [scale_norm, scale_attn, scale_mlp]

    # Action / time
    action_dim: int = 32
    action_horizon: int = 50

    @classmethod
    def from_dict(cls, params: dict):
        return cls(**{k: v for k, v in params.items() if k in cls.__dataclass_fields__})
