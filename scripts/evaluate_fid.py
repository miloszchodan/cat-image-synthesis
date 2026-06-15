import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import Inception_V3_Weights, inception_v3
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from datasets.numpy_dataset import NumpyDataset
from models.ddpm import DDPM
from models.gan import Generator
from models.unet import UNet
from models.vae import VAE
from scripts.common import denormalize, get_device, save_sample_grid, set_seed


def load_state_dict(checkpoint_path: Path, key: str | None, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if key is None:
        return checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    return checkpoint[key]


def load_generator(model_name: str, checkpoint_path: Path, device: torch.device) -> nn.Module:
    if model_name == "vae":
        model = VAE(latent_dim=128).to(device)
        model.load_state_dict(load_state_dict(checkpoint_path, "model_state_dict", device))
        return model.eval()

    if model_name == "gan":
        model = Generator(z_dim=100, features_g=64, img_channels=3).to(device)
        model.load_state_dict(load_state_dict(checkpoint_path, "generator_state_dict", device))
        return model.eval()

    if model_name == "ddpm":
        network = UNet(image_channels=3)
        model = DDPM(network, num_timesteps=1000, device=str(device)).to(device)
        model.load_state_dict(load_state_dict(checkpoint_path, "model_state_dict", device))
        return model.eval()

    raise ValueError(f"Unsupported model: {model_name}")


@torch.no_grad()
def generate_batch(model_name: str, model: nn.Module, batch_size: int, device: torch.device) -> torch.Tensor:
    if model_name == "vae":
        z = torch.randn(batch_size, 128, device=device)
        return model.decoder(z)

    if model_name == "gan":
        z = torch.randn(batch_size, 100, 1, 1, device=device)
        return model(z)

    if model_name == "ddpm":
        return model.sample(image_size=64, batch_size=batch_size, channels=3)

    raise ValueError(f"Unsupported model: {model_name}")


def build_inception(device: torch.device) -> nn.Module:
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights, aux_logits=True, transform_input=False)
    model.fc = nn.Identity()
    return model.to(device).eval()


def preprocess_for_inception(images: torch.Tensor) -> torch.Tensor:
    images = denormalize(images)
    images = torch.nn.functional.interpolate(
        images,
        size=(299, 299),
        mode="bilinear",
        align_corners=False,
    )
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


@torch.no_grad()
def collect_real_features(
    inception: nn.Module,
    data_path: Path,
    num_samples: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dataset = NumpyDataset(data_path, mmap_mode="r")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    features = []
    seen = 0
    total_batches = math.ceil(min(num_samples, len(dataset)) / batch_size)
    for batch in tqdm(loader, desc="Real features", unit="batch", total=total_batches):
        batch = batch.to(device, non_blocking=True).float()
        if seen + batch.size(0) > num_samples:
            batch = batch[: num_samples - seen]
        feats = inception(preprocess_for_inception(batch))
        features.append(feats.cpu().numpy())
        seen += batch.size(0)
        if seen >= num_samples:
            break
    return np.concatenate(features, axis=0)


@torch.no_grad()
def collect_generated_features(
    model_name: str,
    model: nn.Module,
    inception: nn.Module,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    sample_dir: Path | None,
) -> np.ndarray:
    features = []
    remaining = num_samples
    saved_preview = False
    total_batches = math.ceil(num_samples / batch_size)
    for _ in tqdm(range(total_batches), desc=f"{model_name} features", unit="batch"):
        cur_batch = min(batch_size, remaining)
        images = generate_batch(model_name, model, cur_batch, device)
        if sample_dir is not None and not saved_preview:
            save_sample_grid(images[: min(64, cur_batch)], sample_dir / f"{model_name}_fid_samples.png")
            saved_preview = True
        feats = inception(preprocess_for_inception(images))
        features.append(feats.cpu().numpy())
        remaining -= cur_batch
    return np.concatenate(features, axis=0)


def covariance(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.mean(features, axis=0), np.cov(features, rowvar=False)


def trace_sqrt_product(sigma1: np.ndarray, sigma2: np.ndarray, backend: str) -> float:
    if backend == "scipy":
        from scipy import linalg

        covmean = linalg.sqrtm(sigma1 @ sigma2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(np.trace(covmean))

    product = torch.from_numpy(sigma1 @ sigma2).double()
    eigvals = torch.linalg.eigvals(product).real.clamp_min(0.0)
    return float(torch.sqrt(eigvals).sum().item())


def fid_score(real_features: np.ndarray, generated_features: np.ndarray, backend: str) -> float:
    mu1, sigma1 = covariance(real_features)
    mu2, sigma2 = covariance(generated_features)
    diff = mu1 - mu2
    return float(
        diff @ diff
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2.0 * trace_sqrt_product(sigma1, sigma2, backend)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute FID for trained cat generators")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "cats_64.npy")
    parser.add_argument("--models", nargs="+", default=["vae", "gan", "ddpm"], choices=["vae", "gan", "ddpm"])
    parser.add_argument("--vae-checkpoint", type=Path, default=ROOT / "checkpoints" / "vae" / "vae_best.pth")
    parser.add_argument("--gan-checkpoint", type=Path, default=ROOT / "checkpoints" / "gan" / "wgan_gp_latest.pth")
    parser.add_argument("--ddpm-checkpoint", type=Path, default=ROOT / "checkpoints" / "ddpm" / "ddpm_latest.pth")
    parser.add_argument("--num-samples", type=int, default=512, help="FID sample count. Use more for final reporting.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ddpm-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "metrics" / "fid_results.json")
    parser.add_argument("--save-samples", action="store_true", help="Save one generated grid per evaluated model")
    parser.add_argument(
        "--fid-backend",
        choices=["eig", "scipy"],
        default="eig",
        help="Matrix square-root backend. 'eig' avoids SciPy/OpenMP conflicts on Windows.",
    )
    args = parser.parse_args()
    if not args.data.is_absolute():
        args.data = ROOT / args.data
    if not args.output.is_absolute():
        args.output = ROOT / args.output

    if not args.data.exists():
        raise FileNotFoundError(f"Processed dataset not found: {args.data}. Run scripts/preprocess_data.py first.")

    set_seed(args.seed)
    device = get_device()
    inception = build_inception(device)
    real_features = collect_real_features(inception, args.data, args.num_samples, args.batch_size, device)

    checkpoints = {
        "vae": args.vae_checkpoint,
        "gan": args.gan_checkpoint,
        "ddpm": args.ddpm_checkpoint,
    }
    results = {
        "data": str(args.data.relative_to(ROOT) if args.data.is_relative_to(ROOT) else args.data),
        "num_samples": args.num_samples,
        "seed": args.seed,
        "fid": {},
    }
    sample_dir = ROOT / "outputs" / "metrics" / "fid_samples" if args.save_samples else None

    for model_name in args.models:
        checkpoint_path = checkpoints[model_name]
        if not checkpoint_path.exists():
            print(f"[WARN] Skipping {model_name}: missing checkpoint {checkpoint_path}")
            continue
        model = load_generator(model_name, checkpoint_path, device)
        batch_size = args.ddpm_batch_size if model_name == "ddpm" else args.batch_size
        generated_features = collect_generated_features(
            model_name,
            model,
            inception,
            args.num_samples,
            batch_size,
            device,
            sample_dir,
        )
        results["fid"][model_name] = {
            "checkpoint": str(checkpoint_path.relative_to(ROOT) if checkpoint_path.is_relative_to(ROOT) else checkpoint_path),
            "score": fid_score(real_features, generated_features, args.fid_backend),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nSaved: {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
