import numpy as np

from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class PI05Config:
    pi05_repo_id: str = "lerobot/pi05_base"
    tokenizer_repo_id: str = "google/paligemma-3b-pt-224"

    # Inference
    num_inference_steps: int = 10
    max_state_dim: int = 32
    max_action_dim: int = 32
    state_bins: int = 256
    image_size: int = 224
    max_token_len: int = 200
    norm_stats: Optional[Dict] = None

    # SigLIP
    siglip_mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    siglip_std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    # Cameras
    camera_keys = [
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    ]

    # PaliGemma
    paligemma_cfg = {
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
