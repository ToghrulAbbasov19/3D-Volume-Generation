from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
# from datasets import VoxelDataset
from torch.utils.data import DataLoader
from diffusion.model import VAE3D128_diff_prev as VAE3D128_diff
from base import VAE3D128
import random
random.seed(0)
torch.manual_seed(0)

import numpy as np
import sys
import time
from tqdm import tqdm

# from model.autoencoder import Autoencoder
from collections import deque
# from util import create_text_slice, device
device="cuda"

table_data = np.load('/data/hdf5_data/airplane_voxels_test.npy')
print(table_data.shape)
# data=torch.cat([torch.tensor(table_data[210, :, :, :]),torch.tensor(table_data[210, :, :, :]),torch.tensor(table_data[210, :, :, :]),torch.tensor(table_data[210, :, :, :]),torch.tensor(table_data[210, :, :, :])], axis=0)
# data=(data-0.5)/0.5
data=torch.tensor(table_data[:150])
# autoencoder = Autoencoder(is_variational=IS_VARIATIONAL)
autoencoder=VAE3D128_diff().to("cuda")
# autoencoder=VAE3D128().to("cuda")
autoencoder.load_state_dict(torch.load("/data/ae_ckpts_new_shape_2/model_200.pt", map_location="cuda"))
# autoencoder.load()
autoencoder.eval()
output=autoencoder(data.to("cuda"))
output=output.cpu().detach().numpy()

np.save("/root/Diffusion-Project-3DVolume/models/reconstruct/data_gan.npy", output)
print("done")