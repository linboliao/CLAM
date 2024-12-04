import os
import sys
import time

import h5py
import numpy as np
import openslide
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_modules.dataset_h5 import DatasetAllBags, WholeSlideBagFp
from models import get_encoder
from options.base_options import BaseOptions
from utils.file_utils import save_hdf5

sys.path.append('/data2/yhhu/LLB/Code/aslide/')
from aslide import Aslide


class ExtractFeaturesFP:
    def __init__(self, opt):
        self.slide_dir = opt.slide_dir if opt.slide_dir else os.path.join(opt.data_root, 'slides')
        self.coord_dir = opt.coord_dir if opt.coord_dir else os.path.join(opt.data_root, f'patch/{opt.patch_size}/coord')
        self.feat_dir = opt.feat_dir if opt.feat_dir else os.path.join(opt.data_root, f'features/{opt.patch_size}/{opt.feat_model}/')
        self.count_dir = opt.count_dir if opt.count_dir else os.path.join(opt.data_root, f'patch/{opt.patch_size}/')
        self.count_path = os.path.join(self.count_dir, f'count.csv')

        self.model_name = opt.feat_model
        self.batch_size = opt.batch_size
        self.device = torch.device(f'cuda:{opt.gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
        self.skip_done = opt.skip_done

        os.makedirs(self.feat_dir, exist_ok=True)
        os.makedirs(os.path.join(self.feat_dir, 'pt_files'), exist_ok=True)
        os.makedirs(os.path.join(self.feat_dir, 'h5_files'), exist_ok=True)

    def compute_with_dataloader(self, output_path, loader, model, verbose=0):
        """
        Process a data loader and compute features using the provided model.

        Args:
            output_path (str): Directory to save computed features (.h5 file).
            loader (DataLoader): PyTorch DataLoader.
            model (nn.Module): PyTorch model.
            verbose (int): Level of feedback.
        """
        if verbose > 0:
            logger.info(f'Processing {len(loader)} batches')

        mode = 'w'
        for count, data in enumerate(tqdm(loader)):
            with torch.inference_mode():
                batch = data['img'].to(self.device, non_blocking=True)
                features = model(batch)
                features = features.cpu().numpy()

                asset_dict = {'features': features, 'coords': data['coord'].numpy().astype(np.int32)}
                save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
                mode = 'a'

        return output_path

    def extract(self):
        bags_dataset = DatasetAllBags(self.count_path)
        model, img_transforms = get_encoder(self.model_name)
        model.eval()
        model = model.to(self.device)
        total = len(bags_dataset)

        loader_kwargs = {'num_workers': 8, 'pin_memory': True} if self.device.type == "cuda" else {}
        dest_files = os.listdir(os.path.join(self.feat_dir, 'pt_files'))

        for idx in tqdm(range(total)):
            slide_id, slide_ext = os.path.splitext(bags_dataset[idx])
            bag_name = slide_id + '.h5'
            h5_file_path = os.path.join(self.coord_dir, bag_name)
            slide_file_path = os.path.join(self.slide_dir, slide_id + slide_ext)
            logger.info(f'progress: {idx}/{total},{slide_id}')

            if self.skip_done and slide_id + '.pt' in dest_files:
                logger.info(f'skipped {slide_id}')
                continue

            output_path = os.path.join(self.feat_dir, 'h5_files', bag_name)
            time_start = time.time()
            wsi = openslide.open_slide(slide_file_path) if slide_ext != '.kfb' else Aslide(slide_file_path)
            dataset = WholeSlideBagFp(file_path=h5_file_path, wsi=wsi, img_transforms=img_transforms)

            loader = DataLoader(dataset=dataset, batch_size=self.batch_size, **loader_kwargs)
            output_file_path = self.compute_with_dataloader(output_path, loader=loader, model=model, verbose=1)

            time_elapsed = time.time() - time_start
            logger.info('computing features for {} took {} s'.format(output_file_path, time_elapsed))

            with h5py.File(output_file_path, "r") as file:
                features = file['features'][:]
                logger.info(f'features size: {features.shape}, coordinates size: {file["coords"].shape}')

            features = torch.from_numpy(features)
            bag_base, _ = os.path.splitext(bag_name)
            torch.save(features, os.path.join(self.feat_dir, 'pt_files', bag_base + '.pt'))


parser = BaseOptions().parse()
parser.add_argument('--gpu_id', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--skip_done', type=bool, default=True)
parser.add_argument('--feat_dir', type=str)
parser.add_argument('--feat_model', type=str, default='uni_v1', help="{resnet50_trunc、uni_v1、conch_v1}")

if __name__ == '__main__':
    args = parser.parse_args()
    ExtractFeaturesFP(args).extract()
