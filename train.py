import os
import time
import torch
import argparse
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from collections import defaultdict
from terrain_dataset import TerrainDataset

from models import VAE

IMG_SIZE = 512


def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")

    ts = time.time()

    dataset = TerrainDataset(root_dir=r"C:\Users\79140\PycharmProjects\procedural-terrain-generation\data\datapoints_png")

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    print(f"Dataset length: {len(data_loader)}")

    def loss_fn(recon_x, x, mean, log_var):
        batch_size = x.size(0)
        img_flat_size = x[0].numel()

        # prevent BCE NaNs
        epsilon = 1e-7
        recon_x = torch.clamp(recon_x, epsilon, 1.0 - epsilon)

        BCE = torch.nn.functional.binary_cross_entropy(
            recon_x.view(batch_size, img_flat_size),
            x.view(batch_size, img_flat_size),
            reduction='mean'
        )

        log_var = torch.clamp(log_var, -10, 10)
        KLD = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())

        return BCE + KLD

    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels=10 if args.conditional else 0).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    logs = defaultdict(list)

    for epoch in range(args.epochs):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration, x in enumerate(data_loader):

            y = torch.zeros(args.batch_size, dtype=torch.long, device=device)  # dummy labels
            x, y = x.to(device), y.to(device)

            if args.conditional:
                recon_x, mean, log_var, z = vae(x, y)
            else:
                recon_x, mean, log_var, z = vae(x)
                # print(recon_x.min().item(), recon_x.max().item())

            for i, yi in enumerate(y):
                id = len(tracker_epoch)
                tracker_epoch[id]['x'] = z[i, 0].item()
                tracker_epoch[id]['y'] = z[i, 1].item()
                tracker_epoch[id]['label'] = yi.item()

            loss = loss_fn(recon_x, x, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

            if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(data_loader)-1, loss.item()))

                if args.conditional:
                    c = torch.arange(0, 10).long().unsqueeze(1).to(device)
                    z = torch.randn([c.size(0), args.latent_size]).to(device)
                    x = vae.inference(z, c=c)
                else:
                    z = torch.randn([10, args.latent_size]).to(device)
                    x = vae.inference(z)

                plt.figure()
                plt.figure(figsize=(5, 10))
                for p in range(10):
                    plt.subplot(5, 2, p+1)
                    if args.conditional:
                        plt.text(
                            0, 0, "c={:d}".format(c[p].item()), color='black',
                            backgroundcolor='white', fontsize=8)
                    plt.imshow(x[p].view(IMG_SIZE, IMG_SIZE).cpu().data.numpy())
                    plt.axis('off')

                if not os.path.exists(os.path.join(args.fig_root, str(ts))):
                    if not(os.path.exists(os.path.join(args.fig_root))):
                        os.mkdir(os.path.join(args.fig_root))
                    os.mkdir(os.path.join(args.fig_root, str(ts)))

                plt.savefig(
                    os.path.join(args.fig_root, str(ts),
                                 "E{:d}I{:d}.png".format(epoch, iteration)),
                    dpi=300)
                plt.clf()
                plt.close('all')

        df = pd.DataFrame.from_dict(tracker_epoch, orient='index')

        plt.figure(figsize=(6, 6))

        df_limited = df.groupby('label').head(100)
        # Create a color map for labels
        labels = sorted(df_limited['label'].unique())
        colors = cm.tab10(np.linspace(0, 1, len(labels)))

        for label, color in zip(labels, colors):
            subset = df_limited[df_limited['label'] == label]
            plt.scatter(subset['x'], subset['y'], label=label, color=color, s=10, alpha=0.7)

        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(title="label", markerscale=2, fontsize=8)

        plt.savefig(
            os.path.join(args.fig_root, str(ts), "E{:d}-Dist.png".format(epoch)),
            dpi=300
        )
        plt.clf()
        plt.close('all')

        # Save weights
        if not os.path.exists("weights"):
            os.mkdir("weights")

        weight_filename = (
            f"vae_e{args.epochs}_bs{args.batch_size}_lr{args.learning_rate}_"
            f"enc{'-'.join(map(str, args.encoder_layer_sizes))}_"
            f"dec{'-'.join(map(str, args.decoder_layer_sizes))}_"
            f"z{args.latent_size}.pth"
        )

        os.makedirs("weights", exist_ok=True)

        torch.save({
            "state_dict": vae.state_dict(),
            "hyperparams": {
                "encoder_layer_sizes": args.encoder_layer_sizes,
                "decoder_layer_sizes": args.decoder_layer_sizes,
                "latent_size": args.latent_size,
                "conditional": args.conditional
            }
        }, os.path.join("weights", weight_filename))

        print(f"Saved weights and hyperparameters to weights/{weight_filename}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--encoder_layer_sizes", type=int, nargs='+', default=[IMG_SIZE*IMG_SIZE, 512, 256])
    parser.add_argument("--decoder_layer_sizes", type=int, nargs='+', default=[256, 512, IMG_SIZE*IMG_SIZE])
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--print_every", type=int, default=32)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')

    args = parser.parse_args()

    main(args)
