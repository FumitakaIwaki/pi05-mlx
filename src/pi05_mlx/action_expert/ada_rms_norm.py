import mlx.core as mx
import mlx.nn as nn


class AdaRMSNorm(nn.Module):
    """Adaptive RMSNorm
    タイムステップ埋め込みから scale を生成して RMSNormの出力に掛け合わせる

    Weights of pi05:
        .dense.weight [3072, 1024] (out_features, in_features)
        .dense.bias [3072]
    
    Inputs:
        x: (B, T, EXP_HIDDEN=1024)
        timestep: (B, EXP_HIDDEN=1024) ← time_mlp_outの出力
    
    Outputs:
        (B, T, EXP_HIDDEN)
    """

    def __init__(
            self,
            hidden: int,
            ada_dense_out: int
    ):
        super().__init__()
        self.hidden = hidden
        self.ada_dense_out = ada_dense_out
        # Linear: EXP_HIDDEN → ADA_DENSE_OUT
        self.dense = nn.Linear(
            hidden,
            ada_dense_out,
            bias=True,
        )
    
    def __call__(self, x: mx.array, timestep_emb: mx.array) -> mx.array:
        # RMSNorm
        rms = mx.rsqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
        x_norm = x * rms

        # generate scale
        # timestep_emb: (B, 1024) → dense → (B, 3072)
        # 先頭の EXP_HIDDEN (1024) 次元を scale として使用
        scale_all = self.dense(timestep_emb)            # (B, 3072)
        scale = scale_all[:, :self.hidden]     # (B, 1024)
        scale = mx.expand_dims(scale, axis=1)           # (B, 1, 1024)

        return x_norm * (1.0 + scale)
