import mlx.core as mx
import mlx.nn as nn


class ActionEncoder(nn.Module):
    """
    Continuous Action (B, T, ACTION_DIM) → (B, T, EXP_HIDDEN)
    action_in_proj_weight [1024, 32], bias [1024]
    """

    def __init__(
        self,
        action_dim: int,
        hidden: int,
    ):
        super().__init__()
        print(action_dim, hidden)
        self.proj = nn.Linear(action_dim, hidden, bias=True)

    def __call__(self, actions: mx.array) -> mx.array:
        return self.proj(actions)


class ActionDecoder(nn.Module):
    """
    (B, T, EXP_HIDDEN) → (B, T, ACTION_DIM) velocity field `v_0`
    action_out_proj.weight [32, 1024], bias [32]
    """

    def __init__(
        self,
        action_dim: int,
        hidden: int,
    ):
        super().__init__()
        self.proj = nn.Linear(hidden, action_dim, bias=True)

    def __call__(self, h: mx.array) -> mx.array:
        return self.proj(h)
