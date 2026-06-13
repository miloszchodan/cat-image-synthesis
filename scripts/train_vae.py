import argparse
import sys
from pathlib import Path
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from datasets.numpy_dataset import NumpyDataset
from models.vae import VAE
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    mse = nn.functional.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return mse + beta * kld, mse, kld
def main():
    parser = argparse.ArgumentParser(description="Train Baseline VAE")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "cats_64.npy",
                        help="Path to the processed .npy dataset")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume from vae_latest.pth if exists")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension size")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta weight for KL Divergence")
    args = parser.parse_args()
    checkpoints_dir = ROOT / "checkpoints" / "vae"
    samples_dir = ROOT / "outputs" / "vae" / "samples"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if not args.data.exists():
        print(f"[ERROR] Data file not found: {args.data}")
        print("Please run preprocess_data.py first.")
        sys.exit(1)
    print("Loading dataset...")
    dataset = NumpyDataset(args.data, mmap_mode='r')
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True
    )
    print(f"Dataset loaded: {len(dataset)} images.")
    torch.backends.cudnn.benchmark = True
    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    fixed_noise = torch.randn(64, args.latent_dim).to(device)
    scaler = torch.cuda.amp.GradScaler()
    start_epoch = 1
    best_loss = float('inf')
    if args.resume and (checkpoints_dir / "vae_latest.pth").exists():
        print(f"Resuming from {checkpoints_dir / 'vae_latest.pth'}")
        checkpoint = torch.load(checkpoints_dir / "vae_latest.pth", map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            start_epoch = 201
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_kld = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, data in enumerate(pbar):
            data = data.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                recon_batch, mu, logvar = model(data)
                loss, mse, kld = loss_function(recon_batch, data, mu, logvar, beta=args.beta)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            train_mse += mse.item()
            train_kld += kld.item()
            pbar.set_postfix({"Loss": loss.item(), "MSE": mse.item(), "KLD": kld.item()})
        avg_loss = train_loss / len(dataloader)
        avg_mse = train_mse / len(dataloader)
        avg_kld = train_kld / len(dataloader)
        print(f"====> Epoch: {epoch} Average loss: {avg_loss:.4f} (MSE: {avg_mse:.4f}, KLD: {avg_kld:.4f})")
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': avg_loss,
        }
        torch.save(save_dict, checkpoints_dir / "vae_latest.pth")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(save_dict, checkpoints_dir / "vae_best.pth")
        model.eval()
        with torch.no_grad():
            sample = model.decoder(fixed_noise)
            sample_norm = (sample + 1.0) / 2.0
            save_image(sample_norm, samples_dir / f"sample_epoch_{epoch:03d}.png", nrow=8)
    print("Training finished.")
if __name__ == "__main__":
    main()
