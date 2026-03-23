import argparse
import cv2
import logging
import numpy as np
import os
from rich.logging import RichHandler
import urllib

from pi05_mlx.action_expert import ActionExpertConfig
from pi05_mlx.model import PI05Config, PI05Policy


parser = argparse.ArgumentParser(description="Test π₀.₅ MLX select_action")
parser.add_argument(
    "--pi05-repo-or-path",
    default="./models/FIwaki/pi05_base_mlx_bf16",
    help="Path or HuggingFace repo ID for the pi05 model (default: ./models/FIwaki/pi05_base_mlx_bf16)",
)
parser.add_argument(
    "--image-url-or-path",
    default="http://images.cocodataset.org/val2017/000000039769.jpg",
    help="URL or local path to the input image",
)
parser.add_argument(
    "--state-dim",
    type=int,
    default=14,
    help="A number of a state dimension and action dimension",
)


def main(
    pi05_repo_or_path: str,
    image_url_or_path: str,
    state_dim: str = 14,
    logger: logging.Logger = logging.getLogger(__file__),
):
    logger.info("\n=== Testing π₀.₅ MLX: select_action ===\n")

    policy = PI05Policy(
        cfg=PI05Config(
            pi05_repo_id=pi05_repo_or_path,
        ),
        action_expert_cfg=ActionExpertConfig(),
        logger=logger,
    )

    logger.info(f"Loading: {image_url_or_path}")
    try:
        if os.path.isfile(image_url_or_path):
            img_bgr = cv2.imread(image_url_or_path)
        else:
            tmp = "./tests/pi05_test.jpg"
            urllib.request.urlretrieve(image_url_or_path, tmp)
            img_bgr = cv2.imread(tmp)
    except Exception as e:
        raise e
    assert img_bgr is not None

    observation = {
        "state": np.random.uniform(-0.5, 0.5, size=(state_dim,)).astype(np.float32),
        "images": {
            "base_0_rgb": img_bgr,
        },
        "task": "pick up the object on the table",
    }

    logger.info(f"\nObservation:")
    logger.info(f"  state shape : {observation['state'].shape}")
    logger.info(f"  image keys  : {list(observation['images'].keys())}")
    logger.info(f"  task        : {observation['task']}")

    logger.info("\nRunning select_action ...")
    actions = policy.select_action(observation)

    logger.info(f"\n=== Results ===")
    logger.info(f"actions.shape : {actions.shape}")
    logger.info(f"actions[0]    : {np.round(actions[0], 4)}")
    logger.info(f"actions[-1]   : {np.round(actions[-1], 4)}")
    assert not np.isnan(actions).any(), "NaN detected."
    logger.info("\nOK: select_action is completed successfully!")

    if os.path.exists(tmp):
        os.remove(tmp)
        logger.info(f"Removed {tmp}")


if __name__ == "__main__":
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )
    logger = logging.getLogger(__file__)

    main(
        pi05_repo_or_path=args.pi05_repo_or_path,
        image_url_or_path=args.image_url_or_path,
        state_dim=args.state_dim,
        logger=logger,
    )
