import os
import sys
import time

import h5py
import numpy as np
import openslide
import torch
import torch.nn as nn
import torchvision
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_modules.dataset_h5 import DatasetAllBags, WholeSlideBagFp
from models import get_encoder
from models.resnet_custom_dep import resnet18_baseline, resnet50_baseline
from options.base_options import BaseOptions
from options.train_options import TrainOptions
from utils.constants import MODEL2CONSTANTS
from utils.file_utils import save_hdf5
from utils.transform_utils import get_eval_transforms
from utils.utils import param_log

sys.path.append('/data2/lbliao/Code/aslide/')
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
        self.device = torch.device(f'cuda:{opt.gpus}') if torch.cuda.is_available() else torch.device('cpu')
        self.skip_done = opt.skip_done

        param_log(self)
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

    def load_simclr_pretrained_model(self, model, simclr_save_path):
        # add mlp projection head
        dim_mlp = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)
        # load simclr pretrained model parameters
        simclr_saved = torch.load(simclr_save_path)
        state_dict = {}
        for key, value in simclr_saved['state_dict'].items():
            new_key = key.replace("backbone.", "")
            state_dict[new_key] = value
        model.load_state_dict(state_dict)
        print('load simclr pretrained model successfully.')
        model.fc = nn.Identity()

        return model

    def get_model(self):
        if self.model_name == 'resnet18_256':
            model = resnet18_baseline(pretrained=True)
        elif self.model_name == 'resnet50_1024':
            model = resnet50_baseline(pretrained=True)
        elif self.model_name == 'resnet18_512':
            model = torchvision.models.resnet18(pretrained=True)
            model.fc = nn.Identity()
        elif self.model_name == 'resnet50_2048':
            model = torchvision.models.resnet50(pretrained=True)
            model.fc = nn.Identity()
        elif self.model_name == 'simclr_resnet18_512':
            model = torchvision.models.resnet18(pretrained=False, num_classes=128)
            self.load_simclr_pretrained_model(model, args.simclr_save_path)
        elif self.model_name == 'simclr_resnet50_1024':
            model = resnet50_baseline(pretrained=False)
            model.fc = nn.Linear(1024, 128)
            self.load_simclr_pretrained_model(model, args.simclr_save_path)
        elif self.model_name == 'simclr_resnet50_2048':
            model = torchvision.models.resnet50(pretrained=False, num_classes=128)
            self.load_simclr_pretrained_model(model, args.simclr_save_path)
            model.fc = nn.Identity()
        return model.to(self.device)

    def extract(self):
        bags_dataset = DatasetAllBags(self.count_path)
        # constants = MODEL2CONSTANTS['resnet50_trunc']
        # img_transforms = get_eval_transforms(mean=constants['mean'], std=constants['std'], target_img_size=500)
        model, img_transforms = get_encoder(self.model_name)
        # model = self.get_model()
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


parser = TrainOptions().parse()
parser.add_argument('--skip_done', type=bool, default=True)
parser.add_argument('--feat_dir', type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    ExtractFeaturesFP(args).extract()
