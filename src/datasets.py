import os
from torch.utils.data import Dataset
from skimage import io, transform, color
import numpy as np

class DIV2KSubset(Dataset):
    def __init__(self, hr_dir, lr_dir, patch_size=16):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_files = sorted(os.listdir(hr_dir))
        self.lr_files = sorted(os.listdir(lr_dir))
        self.patch_size = patch_size

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_img = io.imread(hr_path).astype(np.float32)/255.
        lr_img = io.imread(lr_path).astype(np.float32)/255.
        
        # optional: convert to grayscale
        if hr_img.ndim == 3 and hr_img.shape[2] == 3:
            hr_img = color.rgb2gray(hr_img)
            lr_img = color.rgb2gray(lr_img)

        # simple patch extraction
        H, W = lr_img.shape
        h, w = self.patch_size, self.patch_size
        lr_patches = []
        hr_patches = []
        for i in range(0, H-h+1, h):
            for j in range(0, W-w+1, w):
                lr_patches.append(lr_img[i:i+h,j:j+w])
                hr_patches.append(hr_img[i:i+h,j:j+w])
        lr_patches = np.stack(lr_patches)
        hr_patches = np.stack(hr_patches)

        return lr_patches, hr_patches