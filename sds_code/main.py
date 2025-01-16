import os
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import math
import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import time
from copy import deepcopy
from network3d_n import UNet3D
from scheduler import DDPMScheduler
from model import DiffusionModule, VAE3D128_diff_8


def seed_everything(seed=2000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(2292)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, -1)


def init_model(args):

    var_scheduler = DDPMScheduler(
        1000,
        beta_1=0.0001,
        beta_T=0.02,
        mode="linear",)

    ddpm = DiffusionModule(None, None)
    ddpm.load(args.ldm_ckpt_path)

    return ddpm


def run(args):
    args.precision = torch.float16 if args.precision == "fp16" else torch.float32

    autoencoder = VAE3D128_diff_8().to("cuda")
    # TODO take path from terminal
    autoencoder.load_state_dict(torch.load(
        args.ae_ckpt_path, map_location="cuda"))
    model = init_model(args)
    # model.use_lora()
    model.alphas = model.var_scheduler.alphas

    # device = model.device
    device = "cuda"
    guidance_scale = args.guidance_scale
    steps = args.step

    class_label = torch.tensor([0, 2]).to("cuda")
    latents = nn.Parameter(
        torch.randn(
            1, 16, 8, 8, 8,
            device=device,
            dtype=args.precision,
        )
    )
    print(device)

    optimizer = torch.optim.AdamW([latents], lr=1e-1, weight_decay=0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int(steps*1.5))

    for a in range(1000):
        # Run optimization
        b = random.randint(1, 2000)
        seed_everything(b)
        latents = nn.Parameter(
            torch.randn(
                1, 16, 8, 8, 8,
                device=device,
                dtype=args.precision,
            )
        )
        optimizer = torch.optim.AdamW([latents], lr=1e-1, weight_decay=0)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 100, int(steps*1.5))

        for step in tqdm(range(steps)):
            optimizer.zero_grad()

            loss = model.get_sds_loss(
                latents=latents,
                text_embeddings=class_label,
                guidance_scale=guidance_scale,)

            (2000 * loss).backward()

            optimizer.step()
            scheduler.step()
            if step in [290, 390, 490]:
                res = autoencoder.decoder(latents).squeeze()
                outputs = res.detach().cpu().numpy()

                np.save(
                    f"./sds_code/outputs/{a}_last_{step}.npy", outputs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--negative_prompt", type=str, default="low quality")
    parser.add_argument("--edit_prompt", type=str, default=None)
    parser.add_argument("--src_img_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="./outputs")

    parser.add_argument("--loss_type", type=str, default="sds")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--step", type=int, default=500)

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--log_step", type=int, default=25)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--ae_ckpt_path", type=str,
                        default="/root/Diffusion-Project-3DVolume-full/diffusion/model_199.pt")
    parser.add_argument("--ldm_ckpt_path", type=str,
                        default="/root/Diffusion-Project-3DVolume-full/diffusion/ckpts/best_2.ckpt")

    return parser.parse_args()


def main():
    args = parse_args()
    assert args.loss_type in ["vsd", "sds", "pds"], "Invalid loss type"
    if args.loss_type in ["pds"]:
        assert args.edit_prompt is not None, f"edit_prompt is required for {args.loss_type}"
        assert args.src_img_path is not None, f"src_img_path is required for {args.loss_type}"

    if os.path.exists(args.save_dir):
        print("[*] Save directory already exists. Overwriting...")
    else:
        os.makedirs(args.save_dir)

    log_opt = vars(args)
    config_path = os.path.join(args.save_dir, "run_config.yaml")
    with open(config_path, "w") as f:
        json.dump(log_opt, f, indent=4)

    print(f"[*] Running {args.loss_type}")
    run(args)


if __name__ == "__main__":
    main()
