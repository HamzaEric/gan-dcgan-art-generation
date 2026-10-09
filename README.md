
# GAN DCGAN Art Generation

A PyTorch-based Deep Convolutional Generative Adversarial Network (DCGAN) trained on the WikiArt dataset to synthesize fine art. This project is designed to leverage Google TPU v5e-1 accelerators through PyTorch-XLA, addressing hardware bottlenecks through dataset RAM caching and precision casting.

The training pipeline explores GAN stabilization techniques, including the Two-Time-Scale Update Rule (TTUR), one-sided label smoothing, and Gaussian noise injection, to help balance the Generator and Discriminator during adversarial training.

## Project Structure

```text
gan-dcgan-art-generation/
├── Neural_Networks_Checkpoints/
│   ├── ..nn
│   ├── D_dcgan_epoch_50.pth
│   ├── D_dcgan_epoch_100.pth
│   ├── D_dcgan_epoch_150.pth
│   ├── D_dcgan_epoch_200.pth
│   ├── G_dcgan_epoch_50.pth
│   ├── G_dcgan_epoch_100.pth
│   ├── G_dcgan_epoch_150.pth
│   ├── G_dcgan_epoch_200.pth
│   ├── gan_min_discriminator_final.pth
│   └── gan_min_generator_final.pth
├── Notebooks/
│   ├── DCGAN_Evaluation.ipynb
│   ├── GAN.ipynb
│   ├── Latent_Exploration.ipynb
│   ├── Training_DCGAN.ipynb
│   └── wikiart_TPU_v5e_1_Compatible_training.ipynb
└── README.md
```

## Notebooks Overview

- **`wikiart_TPU_v5e_1_Compatible_training.ipynb`**: TPU-oriented training pipeline designed for Google TPU v5e-1 using PyTorch-XLA. Explores XLA device mapping, reduced-precision training, and RAM-cached dataset loading to improve training throughput.

- **`Training_DCGAN.ipynb`**: Implements the DCGAN architecture and standard CPU/CUDA-compatible training loop, including `BCEWithLogitsLoss`, `ConvTranspose2d`, and `BatchNorm2d`.

- **`GAN.ipynb`**: Implements the initial, simpler GAN architecture using multilayer perceptrons as a baseline before transitioning to deep convolutional networks.

- **`DCGAN_Evaluation.ipynb`**: Evaluates generated images and model behavior on CPU. Includes real-versus-generated image grids and diversity checks using pixel standard deviation and mean cosine similarity to investigate potential mode collapse.

- **`Latent_Exploration.ipynb`**: Explores the 100-dimensional latent space through one-dimensional and two-dimensional traversals, along with linear interpolation between latent vectors to investigate changes in generated images.

## Checkpoints and Inference

The `Neural_Networks_Checkpoints` directory contains saved model checkpoints from multiple training stages. The DCGAN checkpoints include separate Generator (`G_dcgan`) and Discriminator (`D_dcgan`) weights at epochs 50, 100, 150, and 200. The directory also contains final checkpoints for the initial GAN's Generator and Discriminator.

For CPU inference, instantiate the appropriate model architecture and load its corresponding checkpoint. If the saved weights use `bfloat16` precision, convert the model to `float32` before inference.

Example for loading a DCGAN Generator checkpoint:

```python
import torch

device = torch.device("cpu")
model = GeneratorDCGAN(z_dim=100, feature_g=32).to(device)

ckpt_path = "Neural_Networks_Checkpoints/G_dcgan_epoch_200.pth"

state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)

model = model.float()
model.eval()
```

**Note:** Ensure that `GeneratorDCGAN` and its configuration match the architecture used during training. The example assumes the checkpoint contains a compatible Generator state dictionary. The Discriminator checkpoint (`D_dcgan_epoch_200.pth`) must be loaded into the corresponding Discriminator architecture, not the Generator.

## Training Stabilization Experiments

The project explores techniques intended to reduce training instability and prevent the Discriminator from overpowering the Generator.

- **Learning rate adjustment:** Experiments with a lower Discriminator learning rate (`5e-5`) than the Generator learning rate (`2e-4`).
- **One-sided label smoothing:** Uses `0.9` as the real-image target instead of `1.0`.
- **Gaussian noise injection:** Adds Gaussian noise to Discriminator inputs to investigate whether it improves adversarial training stability.

These methods are experimental mitigations rather than guarantees of convergence. Their effectiveness depends on the dataset, architecture, loss function, and training dynamics.