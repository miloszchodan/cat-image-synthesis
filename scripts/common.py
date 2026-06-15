import random
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denormalize(images: torch.Tensor) -> torch.Tensor:
    return ((images + 1.0) / 2.0).clamp(0.0, 1.0)


def save_sample_grid(images: torch.Tensor, out_path: Path, nrow: int = 8) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(denormalize(images.detach().cpu()), out_path, nrow=nrow)
