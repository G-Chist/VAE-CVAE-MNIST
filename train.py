import os
import time
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from terrain_dataset import TerrainDataset

from models import VAE

IMG_SIZE = 512


def main(args):
    ts = int(time.time())
    fig_dir = os.path.join(args.fig_root, str(ts))
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs("weights", exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Dataset
    dataset = TerrainDataset(root_dir=r"C:\Users\79140\PycharmProjects\procedural-terrain-generation\data\datapoints_png")
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Loss function
    def loss_fn(recon_x, x, mu, logvar):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        recon_flat = recon_x.view(batch_size, -1)
        recon_flat = torch.clamp(recon_flat, 1e-7, 1-1e-7)

        BCE = torch.nn.functional.binary_cross_entropy(recon_flat, x_flat, reduction='mean')
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    # Initialize VAE
    vae = VAE(latent_size=args.latent_size).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    ts = int(time.time())
    os.makedirs(os.path.join(args.fig_root, str(ts)), exist_ok=True)
    os.makedirs("weights", exist_ok=True)

    for epoch in range(args.epochs):
        for iteration, x in enumerate(data_loader):
            x = x.to(device)

            recon_x, mu, logvar, _ = vae(x)
            loss = loss_fn(recon_x, x, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print
            if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
                print(f"Epoch {epoch+1}/{args.epochs} Iter {iteration}/{len(data_loader)-1} Loss {loss.item():.4f}")

            # Save reconstructed images every args.print_every*10 iterations
            if iteration % (args.print_every*10) == 0:
                recon_np = recon_x[0].squeeze(0).detach().cpu().numpy()
                plt.imshow(recon_np, cmap='gray')
                plt.axis("off")
                plt.savefig(os.path.join(args.fig_root, str(ts), f"E{epoch}_I{iteration}.png"), dpi=150)
                plt.close()

            # Save weights every args.print_every*10 iterations
            if iteration % (args.print_every*10) == 0:
                torch.save({
                    "state_dict": vae.state_dict(),
                    "hyperparams": {"latent_size": args.latent_size}
                }, os.path.join("weights", f"vae_cnn_e{args.epochs}_z{args.latent_size}.pth"))

    # Save final model
    torch.save({
        "state_dict": vae.state_dict(),
        "hyperparams": {"latent_size": args.latent_size}
    }, os.path.join("weights", "vae_cnn_final.pth"))
    print("Training complete, final weights saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--fig_root", type=str, default='figs')
    args = parser.parse_args()

    main(args)
