import math

import mlx.core as mx
import mlx.nn as nn


class TimestepEmbedding(nn.Module):
    """
    Weights of pi05:
        - time_mlp_in.weight: [1024, 1024]
        - time_mlp_in.nias: [1024]
        - time_mlp_out.weight: [1024, 1024]
        - time_mlp_out.bias: [1024]
    """

    def __init__(
        self,
        hidden: int,
        min_period: float = 0.004,
        max_period: float = 4.0,
    ):
        super().__init__()
        self.hidden = hidden
        self.min_period = min_period
        self.max_period = max_period

        self.mlp_in = nn.Linear(hidden, hidden, bias=True)
        self.mlp_out = nn.Linear(hidden, hidden, bias=True)

    def __call__(self, t: mx.array) -> mx.array:
        """
        t: (B,) in [0, 1] timesteps of flow matching
            → (B, EXP_HIDDEN)
        """
        # Fourier divergence
        half = self.hidden // 2
        freqs = mx.exp(
            mx.linspace(
                math.log(1.0 / self.max_period),
                math.log(1.0 / self.min_period),
                half,
            )
        )  # (half,)
        t_exp = mx.expand_dims(t, -1) * freqs  # (B, half)
        emb = mx.concatenate([mx.sin(t_exp), mx.cos(t_exp)], axis=-1)  # (B, EXP_HIDDEN)

        # MLP: in → SiLU → out
        emb = nn.silu(self.mlp_in(emb))
        emb = self.mlp_out(emb)

        return emb
