from diffusion.data import Dataset_3d
from tqdm import tqdm
import os
import numpy as np
import sys

sys.path.append('/root/3D-Volume-Generation/dataset')

from voxel_data import VoxelDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
# sys.path is a list of absolute path strings

# device = 'cuda'


class Lambda(nn.Module):
    def __init__(self, function):
        super(Lambda, self).__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


class VAE3D128_diff_8(nn.Module):
    def __init__(self):
        super(VAE3D128_diff_8, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=12,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(12),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels=12, out_channels=24,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels=24, out_channels=24,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels=24, out_channels=48,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels=48, out_channels=48,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels=48, out_channels=16,
                      kernel_size=3, stride=1, padding=1),

        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose3d(in_channels=16, out_channels=48,
                               kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels=48, out_channels=48,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels=48, out_channels=24,
                               kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels=24, out_channels=24,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels=24, out_channels=12,
                               kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(12),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels=12, out_channels=1,
                               kernel_size=4, stride=2, padding=1),

            nn.Sigmoid()
        )

    def forward(self, x):

        x = x.reshape((-1, 1, 64, 64, 64))

        z = self.encoder(x)

        out = self.decoder(z)
        return out.squeeze()


# Training function
def train_model(model, optimizer, criterion, num_epochs, device, train_loader, path):
    model.to(device)

    for epoch in range(num_epochs):

        model.train()
        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), os.path.join(
                path, f"model_{epoch}.pt"))
            torch.save(optimizer.state_dict(),
                       os.path.join(path, f"optim_{epoch}.pt"))
            # eval(model, epoch)

        # running_loss = 0.0
        for i, (batch, _) in enumerate(tqdm(train_loader)):

            # Move data to device
            batch = batch.to(device)

            inputs = batch.reshape((-1, 1, 64, 64, 64))
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            loss = criterion(outputs, batch)

            # # # Backward pass and optimization

            loss.backward()
            optimizer.step()
            print(loss)

        # Print epoch loss
    print("Training complete!")


def main(args):

    criterion = nn.BCELoss()
    # ---------------------------------------------
    # Set path for saving autoencoder ckpts

    # TODO take path from terminal
    path = args.ckpt_path
    os.makedirs(path, exist_ok=True)

    # ---------------------------------------------
    # set data path to your data path
    # TODO take path from terminal
    data_path = args.dataset_path
    plane_path = f'{data_path}/airplane_voxels_train.npy'
    chair_path = f'{data_path}/chair_voxels_train.npy'
    table_path = f'{data_path}/table_voxels_train.npy'

    # ---------------------------------------------
    # Set Parameters
    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.num_epochs
    # ---------------------------------------------

    data = Dataset_3d(chair_path, plane_path, table_path)
    train_loader = DataLoader(data, shuffle=True, batch_size=batch_size)

    # Instantiate the model, optimizer, and dataloader
    model = VAE3D128_diff_8()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, optimizer, criterion,
                num_epochs, device, train_loader, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str,
                        default="root/3D-Volume-Generation")
    parser.add_argument("--dataset_path", type=str, default='/data/hdf5_data')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()
    main(args)
