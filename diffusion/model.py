from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class DiffusionModule(nn.Module):
    def __init__(self, network, var_scheduler, **kwargs):
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler

    def get_loss(self, x0, class_label=None, noise=None):
        ######## TODO ########
        # DO NOT change the code outside this part.
        # compute noise matching loss.
        B = x0.shape[0]
        timestep = self.var_scheduler.uniform_sample_t(B, self.device)

        if noise is None:
            noise = torch.randn_like(x0).to(self.device)

        xT, eps = self.var_scheduler.add_noise(x0, timestep, noise)
        eps_theta = self.network(xT, timestep, class_label)

        loss = F.mse_loss(eps_theta, eps)
        ######################
        return loss
    
    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def data_resolution(self):
        return self.network.data_resolution

    @torch.no_grad()
    def sample(
        self,
        batch_size,
        return_traj=False,
        class_label: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 0.0,
    ):
        x_T = torch.randn([batch_size, 16, self.data_resolution, self.data_resolution, self.data_resolution]).to(self.device)
        do_classifier_free_guidance = guidance_scale > 0.0

        if do_classifier_free_guidance:


            ######## TODO ########
            # Assignment 2-3. Implement the classifier-free guidance.
            # Specifically, given a tensor of shape (batch_size,) containing class labels,
            # create a tensor of shape (2*batch_size,) where the first half is filled with zeros (i.e., null condition).
            assert class_label is not None
            assert len(class_label) == batch_size, f"len(class_label) != batch_size. {len(class_label)} != {batch_size}"
            
            class_label_ext = torch.cat([torch.zeros(len(class_label)), class_label]).to(self.device)
            class_label_ext = class_label_ext.to(torch.int)
            #######################
            


        traj = [x_T]
        for t in tqdm(self.var_scheduler.timesteps):
            x_t = traj[-1]
            if do_classifier_free_guidance:
                ######## TODO ########
                # Assignment 2. Implement the classifier-free guidance.
                
                # timestep_cp = t.repeat(batch_size).to(self.device)


                # print(class_label_ext[:len(class_label)])

                noise_pred = (1 + guidance_scale)*self.network(x_t, timestep=t.to(self.device), class_label=class_label_ext[len(class_label):]) - \
                    guidance_scale*self.network(x_t, timestep=t.to(self.device), class_label=class_label_ext[:len(class_label)])
               
                #######################
            else:
                
                noise_pred = self.network(
                    x_t,
                    timestep=t.to(self.device),
                    class_label=class_label,
                )

            x_t_prev = self.var_scheduler.step(x_t, t, noise_pred)

            traj[-1] = traj[-1].cpu()
            traj.append(x_t_prev.detach())

        if return_traj:
            return traj
        else:
            return traj[-1]

    def save(self, file_path):
        hparams = {
            "network": self.network,
            "var_scheduler": self.var_scheduler,
            } 
        state_dict = self.state_dict()

        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path)

    def load(self, file_path):
        dic = torch.load(file_path, map_location="cpu")
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]

        self.network = hparams["network"]
        self.var_scheduler = hparams["var_scheduler"]

        self.load_state_dict(state_dict)

class VAE3D128_diff(nn.Module):
    def __init__(self):
        super(VAE3D128_diff, self).__init__()

        #Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels = 1, out_channels = 12, kernel_size = 4, stride = 2, padding = 1), 
            nn.BatchNorm3d(12),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv3d(in_channels = 12, out_channels = 24, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv3d(in_channels = 24, out_channels = 24, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels = 24, out_channels = 48, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # nn.Conv3d(in_channels = 48, out_channels = 16, kernel_size = 3, stride = 1, padding = 1),
            
            
            nn.Conv3d(in_channels = 48, out_channels = 48, kernel_size = 4, stride = 2, padding=1),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels = 48, out_channels = 96, kernel_size = 3, stride = 1, padding=1),
            nn.BatchNorm3d(96),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels = 96, out_channels = 96, kernel_size = 4, stride = 2, padding=1),
            nn.BatchNorm3d(96),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels = 96, out_channels = 16, kernel_size = 3, stride = 1, padding=1),
            # nn.BatchNorm3d(96),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            # Lambda(lambda x: x.reshape(x.shape[0], -1)),

            # nn.Linear(in_features = 24, out_features=16)
        )

        # self.bottleneck=nn.Sequential(            
        #     nn.Linear(in_features = 64, out_features=128),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # )

        self.decoder = nn.Sequential(       
            # nn.Linear(in_features = 16, out_features=48),
            # nn.BatchNorm1d(48),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Lambda(lambda x: x.reshape(-1, 48, 1, 1, 1)),
            nn.ConvTranspose3d(in_channels = 16, out_channels = 96, kernel_size = 3, stride = 1, padding=1),
            nn.BatchNorm3d(96),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels = 96, out_channels = 96, kernel_size = 4, stride = 2, padding=1),
            nn.BatchNorm3d(96),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),


            nn.ConvTranspose3d(in_channels = 96, out_channels = 48, kernel_size = 3, stride = 1, padding=1),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels = 48, out_channels = 48, kernel_size = 4, stride = 2, padding=1),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels = 48, out_channels = 24, kernel_size = 3, stride = 1, padding=1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels = 24, out_channels = 24, kernel_size = 4, stride = 2, padding=1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels = 24, out_channels = 12, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm3d(12),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels = 12, out_channels = 1, kernel_size = 4, stride = 2, padding = 1),
            # nn.BatchNorm3d(24),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # nn.ConvTranspose3d(in_channels = 24, out_channels = 1, kernel_size = 4, stride = 2, padding = 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.reshape((-1, 1, 64, 64, 64))
        # print(x.shape)
        x = x.reshape((-1, 1, 64, 64, 64))
        z = self.encoder(x)
        # print(z.shape)
        # print(z.shape)
        # z=self.bottleneck(z)
        # print(z.shape)
        # z=z.reshape(-1, 128, 1, 1, 1)
        # print(z.shape)

        out = self.decoder(z)
        return out.squeeze()

class VAE3D128_diff_prev(nn.Module):
    def __init__(self):
        super(VAE3D128_diff_prev, self).__init__()

        #Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels = 1, out_channels = 12, kernel_size = 4, stride = 2, padding = 1), 
            nn.BatchNorm3d(12),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv3d(in_channels = 12, out_channels = 24, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv3d(in_channels = 24, out_channels = 24, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels = 24, out_channels = 48, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels = 48, out_channels = 16, kernel_size = 3, stride = 1, padding = 1),
            
            
            # nn.Conv3d(in_channels = 48, out_channels = 48, kernel_size = 4, stride = 2, padding=1),
            # nn.BatchNorm3d(48),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # nn.Conv3d(in_channels = 48, out_channels = 96, kernel_size = 3, stride = 1, padding=1),
            # nn.BatchNorm3d(96),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # nn.Conv3d(in_channels = 96, out_channels = 96, kernel_size = 4, stride = 2, padding=1),
            # nn.BatchNorm3d(96),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # nn.Conv3d(in_channels = 96, out_channels = 16, kernel_size = 3, stride = 1, padding=1),
            # nn.BatchNorm3d(96),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            # Lambda(lambda x: x.reshape(x.shape[0], -1)),

            # nn.Linear(in_features = 24, out_features=16)
        )

        # self.bottleneck=nn.Sequential(            
        #     nn.Linear(in_features = 64, out_features=128),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # )

        self.decoder = nn.Sequential(       
            # nn.Linear(in_features = 16, out_features=48),
            # nn.BatchNorm1d(48),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Lambda(lambda x: x.reshape(-1, 48, 1, 1, 1)),
            nn.ConvTranspose3d(in_channels = 16, out_channels = 48, kernel_size = 3, stride = 1, padding=1),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # nn.ConvTranspose3d(in_channels = 96, out_channels = 96, kernel_size = 4, stride = 2, padding=1),
            # nn.BatchNorm3d(96),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),


            # nn.ConvTranspose3d(in_channels = 96, out_channels = 48, kernel_size = 3, stride = 1, padding=1),
            # nn.BatchNorm3d(48),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # nn.ConvTranspose3d(in_channels = 48, out_channels = 48, kernel_size = 4, stride = 2, padding=1),
            # nn.BatchNorm3d(48),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels = 48, out_channels = 24, kernel_size = 3, stride = 1, padding=1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels = 24, out_channels = 24, kernel_size = 4, stride = 2, padding=1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels = 24, out_channels = 12, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm3d(12),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels = 12, out_channels = 1, kernel_size = 4, stride = 2, padding = 1),
            # nn.BatchNorm3d(24),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # nn.ConvTranspose3d(in_channels = 24, out_channels = 1, kernel_size = 4, stride = 2, padding = 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.reshape((-1, 1, 64, 64, 64))
        # print(x.shape)
        x = x.reshape((-1, 1, 64, 64, 64))
        z = self.encoder(x)
        # print(z.shape)
        # print(z.shape)
        # z=self.bottleneck(z)
        # print(z.shape)
        # z=z.reshape(-1, 128, 1, 1, 1)
        # print(z.shape)

        out = self.decoder(z)
        return out.squeeze()



class VAE3D128_diff_8(nn.Module):
    def __init__(self):
        super(VAE3D128_diff_8, self).__init__()

        #Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels = 1, out_channels = 12, kernel_size = 4, stride = 2, padding = 1), 
            nn.BatchNorm3d(12),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv3d(in_channels = 12, out_channels = 24, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv3d(in_channels = 24, out_channels = 24, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels = 24, out_channels = 48, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.Conv3d(in_channels = 48, out_channels = 16, kernel_size = 3, stride = 1, padding = 1),
            
            nn.Conv3d(in_channels = 48, out_channels = 48, kernel_size = 4, stride = 2, padding=1),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # nn.Conv3d(in_channels = 48, out_channels = 96, kernel_size = 3, stride = 1, padding=1),
            # nn.BatchNorm3d(96),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # nn.Conv3d(in_channels = 96, out_channels = 96, kernel_size = 4, stride = 2, padding=1),
            # nn.BatchNorm3d(96),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv3d(in_channels = 48, out_channels = 16, kernel_size = 3, stride = 1, padding=1),
            # nn.BatchNorm3d(96),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            # Lambda(lambda x: x.reshape(x.shape[0], -1)),

            # nn.Linear(in_features = 24, out_features=16)
        )

        # self.bottleneck=nn.Sequential(            
        #     nn.Linear(in_features = 64, out_features=128),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # )

        self.decoder = nn.Sequential(       
            # nn.Linear(in_features = 16, out_features=48),
            # nn.BatchNorm1d(48),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Lambda(lambda x: x.reshape(-1, 48, 1, 1, 1)),
            nn.ConvTranspose3d(in_channels = 16, out_channels = 48, kernel_size = 3, stride = 1, padding=1),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels = 48, out_channels = 48, kernel_size = 4, stride = 2, padding=1),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),


            # nn.ConvTranspose3d(in_channels = 96, out_channels = 48, kernel_size = 3, stride = 1, padding=1),
            # nn.BatchNorm3d(48),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels = 48, out_channels = 24, kernel_size = 3, stride = 1, padding=1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels = 24, out_channels = 24, kernel_size = 4, stride = 2, padding=1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # nn.ConvTranspose3d(in_channels = 24, out_channels = 12, kernel_size = 4, stride = 2, padding=1),
            # nn.BatchNorm3d(24),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels = 24, out_channels = 12, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm3d(12),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose3d(in_channels = 12, out_channels = 1, kernel_size = 4, stride = 2, padding = 1),
            # nn.BatchNorm3d(24),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # nn.ConvTranspose3d(in_channels = 24, out_channels = 1, kernel_size = 4, stride = 2, padding = 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.reshape((-1, 1, 64, 64, 64))
        # print(x.shape)
        x = x.reshape((-1, 1, 64, 64, 64))
        z = self.encoder(x)
        # print(z.shape)
        # print(z.shape)
        # z=self.bottleneck(z)
        # print(z.shape)
        # z=z.reshape(-1, 128, 1, 1, 1)
        # print(z.shape)

        out = self.decoder(z)
        return out.squeeze()