import argparse
import logging
import shutil
from pathlib import Path
from rich.logging import RichHandler

import mlx.core as mx
from safetensors.torch import load_file as torch_load_file
from huggingface_hub import HfApi


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

COPY_FILES = [
    "config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "README.md",
]

DTYPE_MAP = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    # Abbreviation
    "fp32": mx.float32,
    "fp16": mx.float16,
    "bf16": mx.bfloat16,
}


# ═══════════════════════════════════════════════════════════════
# Converter class
# ═══════════════════════════════════════════════════════════════


class PI05MLXConverter:
    def __init__(
        self,
        model_dir: str,
        output_dir: str,
        dtype: str = "bfloat16",
        logger: logging.Logger | None = None,
    ):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.dtype = dtype
        self.logger = logger or logging.getLogger(__name__)

    def convert(self):
        """
        Convert model.safetensors from lerobot/pi05_base to MLX format and save.

        Conversions applied:
          - float32 -> specified dtype (saved natively via mx.save_safetensors)
          - Conv2d weight: PyTorch [O,I,H,W] -> MLX [O,H,W,I]
          - All other tensors are kept as-is with their original key names
        """
        mx_dtype = DTYPE_MAP[self.dtype]
        input_path = self.model_dir / "model.safetensors"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Loading: {input_path}  ({input_path.stat().st_size / 1e9:.1f} GB)")
        pt_tensors = torch_load_file(str(input_path), device="cpu")
        self.logger.info(f"Total tensors: {len(pt_tensors)}")

        mlx_tensors = {}
        for i, (key, tensor) in enumerate(pt_tensors.items()):
            if i % 100 == 0:
                self.logger.info(f"  [{i}/{len(pt_tensors)}] converting...")

            # Convert to numpy via float32, then to MLX array
            np_array = tensor.float().numpy()
            mlx_array = mx.array(np_array)

            # Conv2d weight: [O,I,H,W] -> [O,H,W,I]
            if "patch_embedding.weight" in key and mlx_array.ndim == 4:
                mlx_array = mx.transpose(mlx_array, (0, 2, 3, 1))
                self.logger.info(
                    f"  Transposed Conv2d: {key}  {list(tensor.shape)} → {list(mlx_array.shape)}"
                )

            # Cast to the specified dtype
            mlx_tensors[key] = mlx_array.astype(mx_dtype)

        # Save natively as safetensors via MLX
        output_path = str(self.output_dir / "model.safetensors")
        self.logger.info(f"\nSaving to {output_path} ...")
        mx.save_safetensors(output_path, mlx_tensors)

        size_gb = Path(output_path).stat().st_size / 1e9
        self.logger.info(f"Saved. ({size_gb:.2f} GB)")

    def copy_metadata(self):
        """Copy config and metadata files from the source model directory."""
        for fname in COPY_FILES:
            src = self.model_dir / fname
            if src.exists():
                shutil.copy(src, self.output_dir / fname)
                self.logger.info(f"Copied: {fname}")
            else:
                self.logger.info(f"Skipped (not found): {fname}")

    def upload_to_huggingface(self, repo_id: str, private: bool = False):
        """Upload the converted model to HuggingFace Hub."""
        api = HfApi()

        self.logger.info(f"\nCreating repo: {repo_id} (private={private})")
        api.create_repo(repo_id, exist_ok=True, private=private)

        self.logger.info(f"Uploading {self.output_dir} → {repo_id} ...")
        api.upload_folder(
            folder_path=str(self.output_dir),
            repo_id=repo_id,
            commit_message="Add MLX converted pi05_base weights",
        )
        self.logger.info(f"Done: https://huggingface.co/{repo_id}")

    def run(
        self,
        push_to_hub: bool = False,
        hf_repo_id: str | None = None,
        private: bool = False,
    ):
        """Run conversion, metadata copy, and optionally upload to HuggingFace Hub."""
        self.convert()
        self.copy_metadata()

        if push_to_hub:
            if hf_repo_id is None:
                raise ValueError("--hf-repo-id is required when --push-to-hub is set")
            self.upload_to_huggingface(hf_repo_id, private=private)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(
    description="Covert PyTorch based pi05 to MLX based and push to HF"
)
parser.add_argument(
    "--model-dir",
    required=True,
    help="Path to the model to convert",
)
parser.add_argument(
    "--output-dir",
    default="./converted_model",
    help="Path to the directory for converted models (default: ./converted_model)",
)
parser.add_argument(
    "--hf-repo-id",
    default=None,
    help="Repo id for HF (default: None)",
)
parser.add_argument(
    "--dtype",
    default="bfloat16",
    choices=[
        "float32", "fp32",
        "float16", "fp16",
        "bfloat16", "bf16",
    ],
    help="MLX dtype for the converted model (default: bfloat16)",
)
parser.add_argument(
    "--push-to-hub",
    action="store_true",
    help="Wether push the converted model to HF or not",
)
parser.add_argument(
    "--private",
    action="store_true",
    help="Wether the repository make prvate or not",
)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )
    logger = logging.getLogger(__file__)

    args = parser.parse_args()

    converter = PI05MLXConverter(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        dtype=args.dtype,
        logger=logger,
    )
    converter.run(
        push_to_hub=args.push_to_hub,
        hf_repo_id=args.hf_repo_id,
        private=args.private,
    )
