import argparse
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from models.gan import Generator
from models.vae import VAE
from scripts.common import get_device, save_sample_grid, set_seed


def load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def load_model(model_name: str, checkpoint_path: Path, device: torch.device):
    checkpoint = load_checkpoint(checkpoint_path, device)
    if model_name == "vae":
        model = VAE(latent_dim=128).to(device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        return model.eval()

    if model_name == "gan":
        model = Generator(z_dim=100, features_g=64, img_channels=3).to(device)
        model.load_state_dict(checkpoint["generator_state_dict"])
        return model.eval()

    raise ValueError(f"Unsupported model: {model_name}")


def make_latents(model_name: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if model_name == "vae":
        return torch.randn(128, device=device), torch.randn(128, device=device)
    if model_name == "gan":
        return torch.randn(100, 1, 1, device=device), torch.randn(100, 1, 1, device=device)
    raise ValueError(f"Unsupported model: {model_name}")


def interpolate(z0: torch.Tensor, z1: torch.Tensor, steps: int) -> torch.Tensor:
    weights = torch.linspace(0.0, 1.0, steps, device=z0.device)
    return torch.stack([(1.0 - w) * z0 + w * z1 for w in weights], dim=0)


@torch.no_grad()
def decode(model_name: str, model, latents: torch.Tensor) -> torch.Tensor:
    if model_name == "vae":
        return model.decoder(latents)
    if model_name == "gan":
        return model(latents)
    raise ValueError(f"Unsupported model: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a 10-step latent interpolation grid")
    parser.add_argument("--model", choices=["vae", "gan"], default="vae")
    parser.add_argument("--vae-checkpoint", type=Path, default=ROOT / "checkpoints" / "vae" / "vae_best.pth")
    parser.add_argument("--gan-checkpoint", type=Path, default=ROOT / "checkpoints" / "gan" / "wgan_gp_latest.pth")
    parser.add_argument("--steps", type=int, default=10, help="Total images including both endpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "interpolation")
    args = parser.parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir

    if args.steps < 2:
        raise ValueError("--steps must be at least 2")

    set_seed(args.seed)
    device = get_device()
    checkpoint_path = args.vae_checkpoint if args.model == "vae" else args.gan_checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_model(args.model, checkpoint_path, device)
    z0, z1 = make_latents(args.model, device)
    latents = interpolate(z0, z1, args.steps)
    images = decode(args.model, model, latents)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = args.output_dir / f"{args.model}_latent_interpolation_seed_{args.seed}.png"
    latent_path = args.output_dir / f"{args.model}_latent_interpolation_seed_{args.seed}.npz"
    save_sample_grid(images, grid_path, nrow=args.steps)
    np.savez(
        latent_path,
        z0=z0.detach().cpu().numpy(),
        z1=z1.detach().cpu().numpy(),
        interpolated=latents.detach().cpu().numpy(),
        checkpoint=str(checkpoint_path),
        seed=args.seed,
    )

    print(f"Saved grid: {grid_path.relative_to(ROOT)}")
    print(f"Saved latent matrices: {latent_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
