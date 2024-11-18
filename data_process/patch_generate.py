import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import h5py
from PIL import Image
from openslide import OpenSlide
from loguru import logger

from options.base_options import BaseOptions

sys.path.append('/data2/yhhu/LLB/Code/aslide/')
from aslide import Aslide


class PatchGenerate:
    def __init__(self, opt):
        self.slide_dir = opt.slide_dir if opt.slide_dir else os.path.join(opt.data_root, 'slides')
        self.coord_dir = opt.coord_dir if opt.coord_dir else os.path.join(opt.data_root, f'patch/{opt.patch_size}/coord')
        self.count_dir = opt.count_dir if opt.count_dir else os.path.join(opt.data_root, f'patch/{opt.patch_size}/')
        self.patch_dir = opt.patch_dir if opt.patch_dir else os.path.join(opt.data_root, f'patch/{opt.patch_size}/image')

        self.random_sample = opt.random_sample
        self.patch_size = opt.patch_size
        self.patch_level = opt.patch_level
        self.slide_list = opt.slide_list

        for directory in [self.slide_dir, self.coord_dir, self.patch_dir, self.coord_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

    def generate(self, slide):
        logger.info(f"processing: {slide}")
        slide_name = os.path.splitext(os.path.basename(slide))[0]
        slide_path = os.path.join(self.slide_dir, slide)
        wsi = Aslide(slide_path) if '.kfb' in slide else OpenSlide(slide_path)

        file = h5py.File(os.path.join(self.coord_dir, f'{slide_name}.h5'), mode='r')
        coords_list = list(file['coords'][:])
        if 0 < self.random_sample < len(coords_list):
            coords_list = random.sample(coords_list, self.random_sample)
        logger.info(f'start to save patches of {slide_name}, coords:{len(coords_list)}')

        patch_size = self.patch_size
        for i in range(len(coords_list)):
            [x, y] = coords_list[i]

            patch = wsi.read_region((x, y), self.patch_level, (patch_size, patch_size))
            image_save_path = os.path.join(self.patch_dir, f'{slide_name}_{x}_{y}.jpg')
            patch = Image.fromarray(patch) if '.kfb' in slide else patch.convert('RGB')
            patch.save(image_save_path)

        logger.info(f'Process {slide} Success!')

    @property
    def slides(self):
        if self.slide_list:
            return self.slide_list
        else:
            return [f for f in os.listdir(self.slide_dir) if os.path.isfile(os.path.join(self.slide_dir, f))]

    def run(self):
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(self.generate, slide) for slide in self.slides]
            for future in futures:
                future.result()


parser = BaseOptions().parse()
parser.add_argument('--random_sample', type=int, default=-1, help='')
if __name__ == '__main__':
    args = parser.parse_args()
    PatchGenerate(args).run()
