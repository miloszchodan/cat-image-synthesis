# Cat Image Synthesis

This project explores generating images of cats using three different generative models:
- **DDPM** (Denoising Diffusion Probabilistic Models)
- **GAN** (Generative Adversarial Networks / WGAN-GP)
- **VAE** (Variational Autoencoder)

## Training Progress Samples

Below are generated samples from each model taken at various epochs during training, demonstrating how the generation quality improves over time.

### DDPM (Diffusion)

| Epoch 0 | Epoch 200 | Epoch 400 |
|:---:|:---:|:---:|
| <img src="samples/ddpm_epoch_0.png" width="250"/> | <img src="samples/ddpm_epoch_200.png" width="250"/> | <img src="samples/ddpm_epoch_400.png" width="250"/> |

### GAN

| Epoch 0 | Epoch 200 | Epoch 400 |
|:---:|:---:|:---:|
| <img src="samples/gan_epoch_0.png" width="250"/> | <img src="samples/gan_epoch_200.png" width="250"/> | <img src="samples/gan_epoch_400.png" width="250"/> |

### VAE

| Epoch 0 | Epoch 200 | Epoch 400 |
|:---:|:---:|:---:|
| <img src="samples/vae_epoch_0.png" width="250"/> | <img src="samples/vae_epoch_200.png" width="250"/> | <img src="samples/vae_epoch_400.png" width="250"/> |

## Data and Checkpoints

*Note: The raw dataset and the trained model checkpoints (.pth) are not included directly in this repository due to GitHub's file size limits.*
