import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.numpy_dataset import NumpyDataset
from models.gan import Generator, Critic, initialize_weights
import os
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 50
FEATURES_CRITIC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)
    interpolated_images.requires_grad_(True)
    mixed_scores = critic(interpolated_images)
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    args = parser.parse_args()
    global BATCH_SIZE, NUM_EPOCHS
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    os.makedirs("outputs/gan/samples", exist_ok=True)
    os.makedirs("checkpoints/gan", exist_ok=True)
    print("Loading dataset...")
    dataset = NumpyDataset("data/processed/cats_64.npy")
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True
    )
    torch.backends.cudnn.benchmark = True
    gen = Generator(Z_DIM, FEATURES_GEN, CHANNELS_IMG).to(DEVICE)
    critic = Critic(CHANNELS_IMG, FEATURES_CRITIC).to(DEVICE)
    initialize_weights(gen)
    initialize_weights(critic)
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        start_epoch = checkpoint['epoch'] + 1
        gen.load_state_dict(checkpoint['generator_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
        opt_critic.load_state_dict(checkpoint['opt_critic_state_dict'])
        print(f"=> Loaded checkpoint (resuming from epoch {start_epoch})")
    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(DEVICE)
    step = start_epoch * len(loader)
    print(f"Starting training on device: {DEVICE}")
    for epoch in range(start_epoch, NUM_EPOCHS):
        for batch_idx, real in enumerate(loader):
            real = real.to(DEVICE)
            cur_batch_size = real.shape[0]
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(DEVICE)
                with torch.no_grad():
                    fake = gen(noise)
                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                gp = gradient_penalty(critic, real, fake, device=DEVICE)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                )
                critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(DEVICE)
            fake = gen(noise)
            gen_fake = critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
            if batch_idx % 50 == 0:
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} "
                    f"Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
                )
                with torch.no_grad():
                    fake = gen(fixed_noise)
                    img_grid_real = torchvision.utils.make_grid(
                        real[:32], normalize=True, value_range=(-1, 1)
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:32], normalize=True, value_range=(-1, 1)
                    )
                    torchvision.utils.save_image(
                        img_grid_fake, f"outputs/gan/samples/epoch_{epoch}_step_{step}.png"
                    )
                step += 1
        torch.save({
            'epoch': epoch,
            'generator_state_dict': gen.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'opt_gen_state_dict': opt_gen.state_dict(),
            'opt_critic_state_dict': opt_critic.state_dict(),
        }, f"checkpoints/gan/wgan_gp_latest.pth")
if __name__ == "__main__":
    main()
