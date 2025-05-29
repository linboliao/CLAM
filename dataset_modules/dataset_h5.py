import numpy as np
import pandas as pd
from loguru import logger

from torch.utils.data import Dataset

from PIL import Image
import h5py


class WholeSlideBag(Dataset):
    def __init__(self,
                 file_path,
                 img_transforms=None):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            roi_transforms (callable, optional): Optional transform to be applied on a sample
        """
        self.roi_transforms = img_transforms
        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['imgs']
            self.length = len(dset)

        self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        with h5py.File(self.file_path, "r") as hdf5_file:
            dset = hdf5_file['imgs']
            for name, value in dset.attrs.items():
                logger.info(name, value)

        logger.info('transformations:', self.roi_transforms)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as hdf5_file:
            img = hdf5_file['imgs'][idx]
            coord = hdf5_file['coords'][idx]

        img = Image.fromarray(img)
        img = self.roi_transforms(img)
        return {'img': img, 'coord': coord}


def is_background(img, threshold=15):
    img_array = np.asarray(img, dtype=np.uint8)
    pixel_diff = np.ptp(img_array, axis=2)  # 使用峰值函数替代max-min[4,5](@ref)
    exceed_count = np.count_nonzero(pixel_diff > threshold)  # 比sum更快[4](@ref)
    total_pixels = img_array.shape[0] * img_array.shape[1]
    return exceed_count / total_pixels < 0.3


class WholeSlideBagFp(Dataset):
    def __init__(self, file_path, wsi, img_transforms=None):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            img_transforms (callable, optional): Optional transform to be applied on a sample
        """
        self.wsi = wsi
        self.roi_transforms = img_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(dset)

        self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['coords']
        for name, value in dset.attrs.items():
            logger.info(name, value)

        logger.info('feature extraction settings')
        logger.info('transformations: ', self.roi_transforms)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as hdf5_file:
            coord = hdf5_file['coords'][idx]
        (w, h) = self.wsi.level_dimensions[self.patch_level]
        if 0 <= coord[0] <= w and 0 <= coord[1] <= h:
            img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size))
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img = img.convert('RGB')
            if not is_background(img):
                img = self.roi_transforms(img)
                return {'img': img, 'coord': coord}
            else:
                return None
        else:
            return None


step = 500
it = 1


class DatasetAllBags(Dataset):

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path).iloc[step * it: step * (it + 1)].reset_index()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df['slide_id'][idx]
