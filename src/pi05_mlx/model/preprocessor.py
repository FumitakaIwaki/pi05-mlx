import cv2
import logging
import numpy as np
from transformers import AutoTokenizer
from typing import Optional, Dict


class Preprocessor:
    def __init__(
        self,
        tokenizer_name: str,
        state_bins: int,
        image_size: int,
        max_token_len: int,
        siglip_mean: np.ndarray,
        siglip_std: np.ndarray,
        logger: logging.Logger = logging.getLogger(__file__),
    ):
        self.state_bins = state_bins
        self.image_size = image_size
        self.max_token_len = max_token_len
        self.siglip_mean = siglip_mean
        self.siglip_std = siglip_std

        logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def preprocess_image(self, img_bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 → [-1, 1] float32 (H, W, C)"""
        img = cv2.resize(
            img_bgr, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - self.siglip_mean) / self.siglip_std

        return img

    @staticmethod
    def discretize_state(state: np.ndarray, n_bins: int) -> str:
        """[-1,1] normalized state → tokenized string"""
        bins = np.clip(((state + 1.0) / 2.0 * n_bins).astype(int), 0, n_bins - 1)

        return " ".join(f"s{b:03d}" for b in bins)

    def tokenize(
        self,
        task: str,
        norm_state: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Language instruction → token ids / attention_mask"""
        if norm_state is not None:
            state_str = self.discretize_state(
                state=norm_state,
                n_bins=self.state_bins,
            )
            prompt = f"{task}\n{state_str}"
        else:
            prompt = task

        enc = self.tokenizer(
            prompt,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        return {
            "input_ids": enc["input_ids"][0].astype(np.int32),
            "attention_mask": enc["attention_mask"][0].astype(np.int32),
        }

    @staticmethod
    def normalize_quantile(x, q01, q99, eps=1e-8):
        return np.clip(2.0 * (x - q01) / (q99 - q01 + eps) - 1.0, -1.0, 1.0)

    @staticmethod
    def unnormalize_quantile(x, q01, q99, eps=1e-8):
        return (x + 1.0) / 2.0 * (q99 - q01 + eps) + q01
