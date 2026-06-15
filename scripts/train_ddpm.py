import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.numpy_dataset import NumpyDataset
from models.unet import UNet
from models.ddpm import DDPM
from common import set_seed
import os
ROOT = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
IMAGE_SIZE = 64
CHANNELS_IMG = 3
NUM_EPOCHS = 100
NUM_TIMESTEPS = 1000
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "cats_64.npy")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints" / "ddpm")
    parser.add_argument("--samples-dir", type=Path, default=ROOT / "outputs" / "ddpm" / "samples")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic cuDNN settings")
    args = parser.parse_args()
    set_seed(args.seed, deterministic=args.deterministic)
    global BATCH_SIZE, NUM_EPOCHS
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    args.samples_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print("Loading dataset...")
    dataset = NumpyDataset(args.data)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True
    )
    network = UNet(image_channels=CHANNELS_IMG)
    model = DDPM(network, num_timesteps=NUM_TIMESTEPS, device=DEVICE)
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"=> Loaded checkpoint (resuming from epoch {start_epoch})")
    print(f"Starting training on device: {DEVICE}")
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, real in enumerate(loader):
            real = real.to(DEVICE)
            optimizer.zero_grad()
            loss = model(real)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] Average Loss: {avg_loss:.4f}")
        if epoch % 5 == 0 or epoch == NUM_EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                samples = model.sample(image_size=IMAGE_SIZE, batch_size=16, channels=CHANNELS_IMG)
                img_grid = torchvision.utils.make_grid(
                    samples, normalize=True, value_range=(-1, 1), nrow=4
                )
                torchvision.utils.save_image(
                    img_grid, args.samples_dir / f"epoch_{epoch}.png"
                )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, args.checkpoint_dir / "ddpm_latest.pth")
if __name__ == "__main__":
    main()
