import importlib
import os
import random
import sys
import traceback
from asyncio import as_completed
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import openslide
import torch
from PIL import Image
from loguru import logger
from matplotlib import cm, pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_modules.dataset_generic import Generic_MIL_Dataset
from options.train_options import TrainOptions
from utils.utils import param_log

sys.path.append('/data2/lbliao/Code/aslide/')
from aslide import Aslide


class Heatmaps:
    def __init__(self, opt):
        self.slide_dir = opt.slide_dir if opt.slide_dir else os.path.join(opt.data_root, 'slides')
        self.coord_dir = opt.coord_dir if opt.coord_dir else os.path.join(opt.data_root, f'patch/{opt.patch_size}/coord')
        self.feat_dir = opt.feat_dir if opt.feat_dir else os.path.join(opt.data_root, f'features/{opt.patch_size}/{opt.feat_model}/')
        self.output_dir = opt.output_dir if opt.output_dir else os.path.join(opt.data_root, f'results/heatmaps/{opt.cls_model}/')
        self.label_path = opt.label_path if opt.label_path else os.path.join(opt.data_root, f'labels', 'label.csv')

        self.tmp_coord_dir = os.path.join(opt.data_root, f'patch/{opt.patch_size}/tmp_coord')

        self.patch_size = opt.patch_size
        self.checkpoints = opt.checkpoints
        self.feat_model = opt.feat_model
        self.cls_model = opt.cls_model
        self.slide_ext = opt.slide_ext
        self.seed = opt.seed
        self.device = torch.device(f'cuda:{opt.gpus}') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self.get_model(opt.num_classes, opt.feat_size)
        self.model.load_state_dict(torch.load(self.checkpoints, map_location=self.device))

        param_log(self)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.tmp_coord_dir, exist_ok=True)

    def set_random_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    def save_hdf5(self, slide_name, coords, attr):
        coord_path = os.path.join(self.tmp_coord_dir, f'{slide_name}.h5')

        with h5py.File(coord_path, 'w') as file:
            data_shape = coords.shape
            if 'coords' not in file:
                chunk_shape = (1,) + data_shape[1:]
                max_shape = (None,) + data_shape[1:]
                dset = file.create_dataset('coords', shape=data_shape, maxshape=max_shape, chunks=chunk_shape,
                                           dtype=coords.dtype)
                dset[:] = coords
                if attr is not None and attr:
                    dset.attrs.update(attr)
            else:
                dset = file['coords']
                dset.resize((dset.shape[0] + data_shape[0],) + dset.shape[1:], axis=0)
                dset[-data_shape[0]:] = coords

    def get_model(self, num_classes, feat_size):
        module = "models.model"
        module = importlib.import_module(module)
        clazz = getattr(module, self.cls_model)
        return clazz(n_classes=num_classes, feat_size=feat_size).to(self.device)

    def heatmap(self):
        sample_level = 1
        loader_kwargs = {'num_workers': 8, 'pin_memory': True} if self.device.type == "cuda" else {}
        dataset = Generic_MIL_Dataset(csv_path=self.label_path,
                                      data_dir=self.feat_dir,
                                      shuffle=False,
                                      print_info=True,
                                      label_dict={0: 0, 1: 1},
                                      patient_strat=False,
                                      ignore=[])
        dataset.load_from_h5(True)
        loader = DataLoader(dataset=dataset, batch_size=1, **loader_kwargs)  # 每张wsi的特征数量不一致，batch_size只能为1
        for count, data in enumerate(tqdm(loader)):
            patch_coords = []
            (features, _, coords, slide_id) = data
            features, coords, slide_id = features.squeeze(0), coords.squeeze(0), slide_id[0]
            if slide_id != '1547583.10':
                logger.info(f'skip slide {slide_id}')
                continue
            features = features.to(self.device)
            scores = self.model.get_attention_scores(features)

            scores = [value.item() for value in scores[0]]
            scores = [np.log10(value) for value in scores]
            min_val = min(scores)
            max_val = max(scores)
            scores = [(value - min_val) / (max_val - min_val) for value in scores]
            wsi_path = os.path.join(self.slide_dir, slide_id + self.slide_ext)
            wsi = Aslide(wsi_path) if self.slide_ext == '.kfb' else openslide.OpenSlide(wsi_path)
            w, h = wsi.level_dimensions[sample_level]
            thumb = wsi.get_thumbnail((w, h))
            thumb = Image.fromarray(thumb).convert('RGBA')
            canvas = np.zeros((h, w), dtype=np.float32)
            down_patch_size = int(self.patch_size // wsi.level_downsamples[sample_level])

            # sorted_lst = sorted(scores, reverse=True)
            # index = int(len(sorted_lst) * 0.1)
            # top10 = sorted_lst[index - 1]

            for i, score in enumerate(scores):
                p_w, p_h = coords[i] // wsi.level_downsamples[sample_level]
                if score > 0.7:
                    patch_coords.append(coords[i].tolist())

                p_w, p_h = int(p_w), int(p_h)
                if 0 <= p_w < thumb.width and 0 <= p_h < thumb.height:
                    canvas[p_h:p_h + down_patch_size, p_w:p_w + down_patch_size] = score
                else:
                    logger.info(f"Warning: Coordinates ({p_w}, {p_h}) out of image range.")
            canvas = np.ma.masked_where(~(canvas != 0), canvas)
            fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
            ax[0].imshow(thumb)
            ax[1].imshow(canvas, cmap='coolwarm')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{slide_id}.png'), dpi=1000)

            if len(patch_coords) > 0:
                patch_coords = np.array(patch_coords)
                logger.info(f'Extracted {len(patch_coords)} coordinates')

                attr = {
                    'patch_size': self.patch_size,
                    'patch_level': 0,
                    'level_dim': (w, h),
                    'name': slide_id,
                    'save_path': ''
                }
                self.save_hdf5(slide_id, patch_coords, attr)


parser = TrainOptions().parse()
parser.add_argument('--checkpoints', type=str, default=None, help='Checkpoints path')
parser.add_argument('--slide_ext', type=str, default='.kfb', help='Slide extension')
parser.add_argument('--feat_dir', type=str, default=None, help='Features directory')
parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
parser.add_argument('--label_path', type=str, default=None, help='Label path')
if __name__ == '__main__':
    args = parser.parse_args()
    Heatmaps(args).heatmap()
