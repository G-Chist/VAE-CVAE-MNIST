import os
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting

from models import VAE

IMG_SIZE = 512
img_path = r"C:\Users\79140\PycharmProjects\VAE-CVAE-MNIST\examples\terrainmask.png"


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(args.weight_path, map_location=device)
    hyperparams = checkpoint["hyperparams"]

    # Override args if needed
    for key, value in hyperparams.items():
        setattr(args, key, value)

    # Initialize hardcoded CNN VAE
    vae = VAE(latent_size=args.latent_size).to(device)
    vae.load_state_dict(checkpoint["state_dict"])
    vae.eval()
    print("Weights loaded successfully.")

    # Load and preprocess image
    image = Image.open(img_path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dim

    # Forward pass
    with torch.no_grad():
        recon_tensor, _, _, _ = vae(img_tensor)

    # Convert tensors to numpy arrays
    orig_np = img_tensor.squeeze(0).squeeze(0).cpu().numpy()
    recon_np = recon_tensor.squeeze(0).squeeze(0).cpu().numpy()

    # Side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(orig_np, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(recon_np, cmap='gray')
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")
    plt.show()

    # 3D surface plot of reconstructed terrain
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    X = np.linspace(0, IMG_SIZE - 1, IMG_SIZE)
    Y = np.linspace(0, IMG_SIZE - 1, IMG_SIZE)
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, recon_np, cmap='terrain')
    ax.set_title("3D Reconstruction")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--weight_path", type=str, required=True)
    args = parser.parse_args()

    main(args)
