import logging
import numpy as np
from pathlib import Path
from typing import Dict, List

import mlx.core as mx
from mlx_vlm.models.paligemma import paligemma
from mlx_vlm.models.paligemma.paligemma import ModelConfig

from pi05_mlx.action_expert import PI05ActionExpert, ActionExpertConfig
from .config import PI05Config
from .preprocessor import Preprocessor


class PI05Policy:
    def __init__(
        self,
        cfg: PI05Config,
        action_expert_cfg: ActionExpertConfig,
        logger: logging.Logger = logging.getLogger(__file__)
    ):
        self.logger = logger
        self.cfg = cfg
        self.norm_stats = cfg.norm_stats

        self._load_paligemma(
            repo_id=cfg.pi05_repo_id,
            model_cfg=cfg.paligemma_cfg,
        )
        self._load_expert(
            repo_id=cfg.pi05_repo_id,
            cfg=action_expert_cfg,
            logger=logger,
        )

        self.preprocessor = Preprocessor(
            tokenizer_name=cfg.tokenizer_repo_id,
            state_bins=cfg.state_bins,
            image_size=cfg.image_size,
            max_token_len=cfg.max_token_len,
            siglip_mean=cfg.siglip_mean,
            siglip_std=cfg.siglip_std,
            logger=logger,
        )
    
    def _load_paligemma(
        self,
        repo_id: str,
        model_cfg: Dict,
    ):
        repo_path = Path(repo_id)
        self.logger.info(f"Loading PaliGemma from {repo_id}...")

        model_config = ModelConfig.from_dict(model_cfg)

        # Model construction
        self.vlm_backbone = paligemma.Model(model_config)
        raw = mx.load(str(repo_path / "model.safetensors"))

        prefix = "paligemma_with_expert.paligemma.model."
        remapped = {}
        for key, val in raw.items():
            if not key.startswith(prefix):
                continue
            new_key = key[len(prefix):]
            if new_key.startswith("language_model."):
                new_key = "language_model.model." + new_key[len("language_model."):]
            remapped[new_key] = val
        
        lm_head = "paligemma_with_expert.paligemma.lm_head.weight"
        if "language_model.model.embed_tokens.weight" not in remapped and lm_head in raw:
            remapped["language_model.model.embed_tokens.weight"] = raw[lm_head]
        
        self.vlm_backbone.load_weights(list(remapped.items()), strict=False)
        mx.eval(self.vlm_backbone.parameters())

        self.vlm_processor = None
        self.logger.info("PaliGemma loaded.")
    
    def _load_expert(
        self,
        repo_id: str,
        cfg: ActionExpertConfig,
        logger: logging.Logger | None = None,
    ):
        self.logger.info("Building ActionExpert...")
        self.action_expert = PI05ActionExpert(
            cfg=cfg,
            logger=logger,
        )
        self.action_expert._load_expert_weights(repo_id=repo_id)

        mx.eval(self.action_expert.parameters())
        self.logger.info("Action expert loaded.")
    
    def _prepare_state(self, state: np.ndarray) -> np.ndarray:
        """state_dim → padding to MAX_STATE_DIM → normalize QUANTILES"""
        padded = np.zeros(self.cfg.max_state_dim, dtype=np.float32)
        dim = min(len(state), self.cfg.max_state_dim)
        padded[:dim] = state[:dim]

        if self.norm_stats and "state" in self.norm_stats:
            q01 = np.array(self.norm_stats["state"]["q01"])
            q99 = np.array(self.norm_stats["state"]["q99"])
            padded = self.preprocessor.normalize_quantile(padded, q01, q99)
        
        return padded
    
    def _postprocess_action(self, actions: np.array) -> np.array:
        if self.norm_stats and "action" in self.norm_stats:
            q01 = np.array(self.norm_stats["action"]["q01"])
            q99 = np.array(self.norm_stats["action"]["q99"])
            actions = Preprocessor.unnormalize_quantile(actions, q01, q99)

        return actions

    def extract_vlm_backbone_hidden(
        self,
        images: List[np.ndarray],
        token_ids: np.ndarray,
    ) -> mx.array:
        """
        Through SigLIP + Projector + Gemma layers,
        return final hidden state (1, T_prefix, PG_HIDDEN)
        """
        vision_tower = self.vlm_backbone.vision_tower
        projector = self.vlm_backbone.multi_modal_projector
        lm = self.vlm_backbone.language_model

        # Images
        img_embeds_list = []
        for img in images:
            pv = mx.array(img[None])

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
        img_embeds = mx.concatenate(img_embeds_list, axis=1)

        # Text
        ids = mx.array(token_ids[None])
        text_embeds = lm.model.embed_tokens(ids)

        # Total
        h = mx.concatenate([img_embeds, text_embeds], axis=1)

        for layer in lm.model.layers:
            h = layer(h)
        
        if hasattr(lm.model, "norm"):
            h = lm.model.norm(h)
        
        return h

    def select_action(self, observation: Dict) -> np.ndarray:
        """observation → action (action_horizon, action_dim)
        keys of observation:
            "state" : np.ndarray (state_dim,)
            "images": dict[str, np.ndarray (H, W, 3) uint 8 BGR]
            "task" : str
        """
        state = observation.get("state", np.zeros(self.cfg.max_state_dim, dtype=np.float32))
        images = observation.get("images", {})
        task = observation.get("task", "")

        norm_state = self._prepare_state(state)

        processed_images = []
        for key in self.cfg.camera_keys:
            if key in images:
                processed_images.append(
                    self.preprocessor.preprocess_image(images[key])
                )
        
        if not processed_images:
            raise ValueError(
                f"There is no available key in observation['images']",
                f"Available keys: {self.cfg.camera_keys}"
            )

        tokens = self.preprocessor.tokenize(task=task, norm_state=norm_state)

        vlm_hidden = self.extract_vlm_backbone_hidden(
            images=processed_images,
            token_ids=tokens["input_ids"],
        )
        mx.eval(vlm_hidden)

        actions_mlx = self.action_expert.flow_matching_sample(
            hidden=vlm_hidden,
            num_steps=self.cfg.num_inference_steps,
        )

        actions = np.array(actions_mlx[0])
        actions = self._postprocess_action(actions)

        actual_dim = len(state) if len(state) <= self.cfg.max_action_dim else self.cfg.max_action_dim
        actions = actions[:, :actual_dim]

        return actions
