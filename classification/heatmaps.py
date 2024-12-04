import importlib
import os
import random
import sys

import numpy as np
import openslide
import torch
from PIL import Image
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_modules.dataset_generic import Generic_MIL_Dataset

sys.path.append('/data2/yhhu/LLB/Code/aslide/')
from aslide import Aslide


class Heatmaps:
    def __init__(self, opt):
        self.slide_dir = getattr(opt, 'slide_dir', os.path.join(opt.data_root, 'slides'))
        self.coord_dir = getattr(opt, 'coord_dir', os.path.join(opt.data_root, f'patch/{opt.patch_size}/coord'))
        self.pt_dir = getattr(opt, 'pt_dir', os.path.join(opt.data_root, f'features/{opt.patch_size}/{opt.feat_model}/'))
        self.output_dir = getattr(opt, 'output_dir', os.path.join(opt.data_root, f'results/heatmaps/{opt.feat_model}/'))
        self.label_path = getattr(opt, 'label_path', os.path.join(opt.data_root, f'labels', 'label.csv'))
        self.patch_size = getattr(opt, 'patch_size', 256)

        self.checkpoints = opt.checkpoints
        self.model_name = opt.feat_model
        self.batch_size = opt.batch_size
        self.slide_ext = opt.slide_ext
        self.seed = opt.seed
        self.device = torch.device(f'cuda:{opt.gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
        self.skip_done = opt.skip_done
        self.model = self.get_model(opt.num_classes, opt.feat_size)

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

    def get_model(self, num_classes, feat_size):
        module = "models.model"
        module = importlib.import_module(module)
        clazz = getattr(module, self.model_name)
        return clazz(num_classes=num_classes, feat_size=feat_size)

    def heatmap(self):
        # TODO 以原图为底色，注意力分数为蒙版
        loader_kwargs = {'num_workers': 8, 'pin_memory': True} if self.device.type == "cuda" else {}
        dataset = Generic_MIL_Dataset(csv_path=self.label_path,
                                      data_dir=self.pt_dir,
                                      shuffle=False,
                                      print_info=True,
                                      label_dict={0: 0, 1: 1},
                                      patient_strat=False,
                                      ignore=[])
        dataset.load_from_h5(True)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, **loader_kwargs)
        for count, data in enumerate(tqdm(loader)):
            slide_id = data.get('slide_id')

            scores = self.model.get_attention_scores(data.get('features'))
            scores = [value.item() for value in scores[0]]
            scores = [np.log10(value) for value in scores]
            min_val = min(scores)
            max_val = max(scores)
            scores = [(value - min_val) / (max_val - min_val) for value in scores]
            wsi_path = os.path.join(self.slide_dir, slide_id + self.slide_ext)
            wsi = Aslide(wsi_path) if self.slide_ext == '.kfb' else openslide.OpenSlide(wsi_path)
            w, h = wsi.level_dimensions[-1]
            thumb = wsi.get_thumbnail((w, h))
            canvas = Image.new("RGBA", (w, h))
            down_patch_size = self.patch_size // wsi.level_downsamples[-1]
            for i, score in enumerate(scores):
                w, h = data['coords'][i] // wsi.level_downsamples[-1]
                if 0 <= w < thumb.size[0] and 0 <= h < thumb.size[1]:
                    canvas[h:h + down_patch_size, w:w + down_patch_size] = scores
                else:
                    logger.info(f"Warning: Coordinates ({w}, {h}) out of image range.")
            canvas.putalpha(125)
            thumb.paste(canvas, (0, 0), canvas)
            thumb.save(os.path.join(self.output_dir, f'{slide_id}.png'))
