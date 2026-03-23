# $\pi_{0.5}$ -MLX: Convert pytorch based pi05 to mlx based
Author: @FumitakaIwaki

## Setup and Run
1. Build the environment
    ```shell
    uv sync
    ```
1. Download the pi05 policy (following example is use `lerobot/pi05_base` )
    ```shell
    uv run hf download lerobot/pi05_base --repo-type model --local-dir ./models/lerobot/pi05_base
    ```
1. Convert the policy to mlx and push to HF
    ```shell
    uv run python -m src.pi05_mlx.mlx_converter.convert \
		--model-dir ./models/lerobot/pi05_base \
		--output-dir ./models/<Your-HF-Name>/pi05_base_mlx_bf16 \
		--hf-repo-id <Your-HF-Name>/pi05_base_mlx_bf16 \
		--dtype bf16 \
		--push-to-hub \
		--private
    ```
1. Test inference using the converted model
    ```shell
    uv run ./tests/test_select_action.py \
		--pi05-repo-or-path ./models/<Your-HF-Name>/pi05_base_mlx_bf16 \
		--image-url-or-path http://images.cocodataset.org/val2017/000000039769.jpg \
		--state-dim 14
    ```
    The used sample image is [THIS](http://images.cocodataset.org/val2017/000000039769.jpg) .

## Usage

### `convert.py`
```
usage: convert.py [-h] --model-dir MODEL_DIR [--output-dir OUTPUT_DIR] [--hf-repo-id HF_REPO_ID] [--dtype {float32,fp32,float16,fp16,bfloat16,bf16}] [--push-to-hub] [--private]

Covert PyTorch based pi05 to MLX based and push to HF

options:
  -h, --help            show this help message and exit
  --model-dir MODEL_DIR
                        Path to the model to convert
  --output-dir OUTPUT_DIR
                        Path to the directory for converted models (default: ./converted_model)
  --hf-repo-id HF_REPO_ID
                        Repo id for HF (default: None)
  --dtype {float32,fp32,float16,fp16,bfloat16,bf16}
                        MLX dtype for the converted model (default: bfloat16)
  --push-to-hub         Wether push the converted model to HF or not
  --private             Wether the repository make prvate or not
```

### `test_select_action.py`
```
usage: test_select_action.py [-h] [--pi05-repo-or-path PI05_REPO_OR_PATH] [--image-url-or-path IMAGE_URL_OR_PATH] [--state-dim STATE_DIM]

Test π₀.₅ MLX select_action

options:
  -h, --help            show this help message and exit
  --pi05-repo-or-path PI05_REPO_OR_PATH
                        Path or HuggingFace repo ID for the pi05 model (default: ./models/FIwaki/pi05_base_mlx_bf16)
  --image-url-or-path IMAGE_URL_OR_PATH
                        URL or local path to the input image
  --state-dim STATE_DIM
                        A number of a state dimension and action dimension
```