import numpy as np

res = np.load('/root/Diffusion-Project-3DVolume/diffusion/try/1.npy', allow_pickle=True)
res[res > 0.15] = 1
res[res <= 0.15] = 0
np.save('/root/Diffusion-Project-3DVolume/diffusion/try/1_final.npy', res)