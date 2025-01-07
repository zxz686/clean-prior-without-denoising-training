import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
import os
import torch

class DatasetCAttDenoising(data.Dataset):
    """
    Dataset class for loading paired clean, noisy images, and MAE images.
    """
    def __init__(self, opt):
        super(DatasetCAttDenoising, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if 'n_channels' in opt else 3
        self.sf = opt['scale'] if 'scale' in opt else 1
        self.lq_patchsize = opt['lq_patchsize'] if 'lq_patchsize' in opt else 64
        self.patch_size = opt['H_size'] if 'H_size' in opt else self.lq_patchsize * self.sf

        self.paths_H = util.get_image_paths(opt['dataroot_H'])  # Clean images
        self.paths_L = util.get_image_paths(opt['dataroot_L'])  # Noisy images
        self.paths_M = util.get_image_paths(opt['dataroot_mae'])  # MAE images
        assert self.paths_H, 'Error: H path is empty.'
        assert len(self.paths_H) == len(self.paths_L) == len(self.paths_M), 'Mismatch between clean, noisy, and MAE image counts.'

    def __getitem__(self, index):
        # ------------------------------------
        # get H, L, and M images
        # ------------------------------------
        H_path = self.paths_H[index]
        L_path = self.paths_L[index]
        M_path = self.paths_M[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        img_L = util.imread_uint(L_path, self.n_channels)
        img_M = util.imread_uint(M_path, self.n_channels)  # Assume img_M has the same channels as L and H

        if self.opt['phase'] == 'train':
            # If train, get L/H/M patch trio
            H, W, _ = img_H.shape
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            img_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            img_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            img_M = img_M[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]  # Extract matching patch from M

            # Data augmentation
            mode = random.randint(0, 7)
            img_H = util.augment_img(img_H, mode=mode)
            img_L = util.augment_img(img_L, mode=mode)
            img_M = util.augment_img(img_M, mode=mode)  # Augment M image in the same way

        # Convert H, L, and M images to tensors
        img_H, img_L = util.uint2single(img_H), util.uint2single(img_L)
        img_M = util.uint2single(img_M)  # Convert M to single
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
        img_M = util.single2tensor3(img_M)  # Convert M to tensor

        # Combine L and M to form a 6-channel input tensor
        img_LM = torch.cat([img_L, img_M], dim=0)  
        return {'LM': img_LM, 'H': img_H, 'L_path': L_path, 'H_path': H_path, 'M_path': M_path}

    def __len__(self):
        return len(self.paths_H)