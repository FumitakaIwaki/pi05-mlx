.PHONY:
	help

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

convert:
	uv run python -m src.pi05_mlx.mlx_converter.convert
