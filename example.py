import os
import time
import torch
import argparse
from PIL import Image
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from collections import defaultdict

from models import VAE

img_path = r"C:\Users\79140\PycharmProjects\VAE-CVAE-MNIST\examples\2.png"


def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels=10 if args.conditional else 0).to(device)

    weight_file = "weights/vae_e10_bs64_lr0.001_enc784-256_dec256-784_z2.pth"

    vae.load_state_dict(torch.load(weight_file, map_location="cpu"))
    vae.eval()

    print("Weights loaded successfully.")

    # Read a PIL image
    image = Image.open(img_path)

    # Define a transform to convert PIL
    # image to a Torch tensor compatible with MNIST images
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # convert to b&w
        transforms.Resize((28, 28)),  # force correct size
        transforms.ToTensor()
    ])

    # Convert the PIL image to Torch tensor
    img_tensor = transform(image)

    print(img_tensor.size())

    reconstructed_img, _, _, z = vae.forward(img_tensor)

    plt.imshow(reconstructed_img.view(28, 28).cpu().data.numpy())
    plt.axis("off")
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[784, 256])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 784])
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')

    args = parser.parse_args()

    main(args)