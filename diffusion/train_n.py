import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
# from dataset import AFHQDataModule, get_data_iterator, tensor_to_pil_image
from dotmap import DotMap
from model import DiffusionModule, VAE3D128_diff_8
from network3d_n import UNet3D
from pytorch_lightning import seed_everything
from scheduler import DDPMScheduler
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from data import Dataset_3d
from torch.utils.data import DataLoader, Dataset
import numpy as np


matplotlib.use("Agg")


def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now


def main(args):
    """config"""
    config = DotMap()
    config.update(vars(args))
    config.device = f"cuda:{args.gpu}"

    now = get_current_time()
    if args.use_cfg:
        save_dir = Path(f"results/cfg_diffusion-{args.sample_method}-{now}")
    else:
        save_dir = Path(f"results/diffusion-{args.sample_method}-{now}")
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"save_dir: {save_dir}")

    seed_everything(config.seed)

    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    """######"""

    # ----------------------------------------------

    # Set directories here
    # TODO take path from terminal
    plane_path = f'{args.dataset_path}/airplane_voxels_train.npy'
    chair_path = f'{args.dataset_path}/chair_voxels_train.npy'
    table_path = f'{args.dataset_path}/table_voxels_train.npy'

    # ----------------------------------------------

    # ----------------------------------------------

    # set batch size here
    # TODO take batch size from terminal
    batch_size = args.batch_size

    # ----------------------------------------------

    data = Dataset_3d(chair_path, plane_path, table_path)
    train_loader = DataLoader(data, shuffle=True, batch_size=batch_size)

    if args.sample_method == "ddpm":
        var_scheduler = DDPMScheduler(
            config.num_diffusion_train_timesteps,
            beta_1=config.beta_1,
            beta_T=config.beta_T,
            mode="linear",
        )
    else:
        raise ValueError(f"Invalid sample method {args.sample_method}")

    # ----------------------------------------------

    # set autoencoder ckpt path
    # TODO take path from terminal
    autoencoder = VAE3D128_diff_8().to('cuda')
    autoencoder.load_state_dict(torch.load(
        args.ae_ckpt_path, map_location="cuda"))
    # ----------------------------------------------

    network = UNet3D(
        T=config.num_diffusion_train_timesteps,
        data_resolution=8,
        ch=192,
        ch_mult=[1, 2, 2],
        attn=[1],
        num_res_blocks=3,
        dropout=0.1,
        use_cfg=args.use_cfg,
        cfg_dropout=args.cfg_dropout,
        num_classes=3,
    )

    ddpm = DiffusionModule(network, var_scheduler)
    ddpm = ddpm.to(config.device)

    optimizer = torch.optim.Adam(ddpm.network.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min((t + 1) / config.warmup_steps, 1.0)
    )

    # TODO take epochs from terminal
    step = 0
    losses = []
    epochs = args.num_diffusion_train_timesteps

    for i in range(epochs):

        label = []
        data = []
        for data, label in tqdm(train_loader):

            label, data = label.to(config.device), data.to(config.device)

            autoencoder.eval()
            data = data.reshape((-1, 1, 64, 64, 64)).to('cuda')
            output = autoencoder.encoder(data.to("cuda"))

            if args.use_cfg:  # Conditional, CFG training
                loss = ddpm.get_loss(output, class_label=label)
            else:  # Unconditional training
                loss = ddpm.get_loss(output)

            optimizer.zero_grad()
            loss.backward()
            print(loss)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

            step += 1
            # pbar.update(1)
        if i % 1 == 0:
            ddpm.eval()
            autoencoder.eval()
            print(f'Epoch {i} is done')
            samples = ddpm.sample(4, return_traj=False)
            outputs = autoencoder.decoder(samples).squeeze()
            outputs = outputs.detach().cpu().numpy()
            np.save(f"{save_dir}/last.npy", outputs)

            ddpm.save(f"{save_dir}/last.ckpt")
            # ddpm.save(f"{save_dir}/{i}.ckpt")
            ddpm.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument(
        "--num_diffusion_train_timesteps",
        type=int,
        default=1000,
        help="diffusion Markov chain num steps",
    )
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--sample_method", type=str, default="ddpm")
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--cfg_dropout", type=float, default=0.1)
    parser.add_argument("--dataset_path", type=str, default='/data/hdf5_data')
    parser.add_argument("--ae_ckpt_path", type=str,
                        default='/root/3D-Volume-Generation/model_2.pt')
    args = parser.parse_args()
    main(args)
