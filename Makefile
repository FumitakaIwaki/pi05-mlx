.PHONY:
	help \
	convert-h \
	convert \
	test

.DEFAULT_GOAL := help

help:
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "; section = ""} \
		/^### .* ###$$/ { \
			section = substr($$0, 5, length($$0) - 8); \
			printf "\n\033[1;33m%s\033[0m\n", section; \
			next; \
		} \
		/^[a-zA-Z0-9_-]+:.*?## / { \
			printf "  \033[36m%-28s\033[0m %s\n", $$1, $$2; \
		}' $(MAKEFILE_LIST)
	@echo ""

include .env


### Examples ###

convert-h:
	uv run python -m src.pi05_mlx.mlx_converter.convert -h

convert:
	uv run python -m src.pi05_mlx.mlx_converter.convert \
		--model-dir ./models/lerobot/pi05_base \
		--output-dir ./models/FIwaki/pi05_base_mlx_bf16 \
		--hf-repo-id FIwaki/pi05_base_mlx_bf16 \
		--dtype bf16 \
		--push-to-hub \
		--private

test:
	uv run ./tests/test_select_action.py \
		--pi05-repo-or-path ./models/FIwaki/pi05_base_mlx_bf16 \
		--image-url-or-path http://images.cocodataset.org/val2017/000000039769.jpg \
		--state-dim 14