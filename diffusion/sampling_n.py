import argparse

import numpy as np
import torch
# from dataset import tensor_to_pil_image
# from model import DiffusionModule
from scheduler import DDPMScheduler
from pathlib import Path
from model import DiffusionModule, VAE3D128_diff_8


def main(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    device = f"cuda:{args.gpu}"

    ddpm = DiffusionModule(None, None)
    ddpm.load(args.ldm_ckpt_path)
    ddpm.eval()
    ddpm = ddpm.to(device)

    num_train_timesteps = ddpm.var_scheduler.num_train_timesteps
    ddpm.var_scheduler = DDPMScheduler(
        num_train_timesteps,
        beta_1=1e-4,
        beta_T=0.02,
        mode="linear",
    ).to(device)

    total_num_samples = 500
    num_batches = 3

    # ----------------------------------------------
    # set autoencoder ckpt path
    # TODO take path from terminal
    autoencoder = VAE3D128_diff_8().to('cuda')
    autoencoder.load_state_dict(torch.load(
       args.ae_ckpt_path, map_location="cuda"))

    # ----------------------------------------------

    for i in range(num_batches):
        sidx = i * args.batch_size
        eidx = min(sidx + args.batch_size, total_num_samples)
        # B = eidx - sidx
        B = 3

        if args.use_cfg:  # Enable CFG sampling
            assert ddpm.network.use_cfg, f"The model was not trained to support CFG."
            samples = ddpm.sample(
                B,
                class_label=torch.tensor([i + 1]*B),
                guidance_scale=args.cfg_scale,
            )
        else:
            samples = ddpm.sample(B)

        outputs = autoencoder.decoder(samples).squeeze()

        samples = outputs.detach().cpu().numpy()

        np.save(f"{save_dir}/{i + 1}.npy", samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ldm_ckpt_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--sample_method", type=str, default="ddpm")
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--ae_ckpt_path", type=str,
                        default='/root/3D-Volume-Generation/model_2.pt')

    args = parser.parse_args()
    main(args)
