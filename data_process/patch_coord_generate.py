import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import h5py
import numpy as np
import pandas as pd
from loguru import logger
from openslide import OpenSlide

from options.base_options import BaseOptions
from utils.utils import param_log

sys.path.insert(0, r'/data2/lbliao/Code/aslide')
from aslide import Aslide


class PatchCoordsGenerator:
    def __init__(self, opt):
        self.slide_dir = opt.slide_dir if opt.slide_dir else os.path.join(opt.data_root, 'slides')
        self.coord_dir = opt.coord_dir if opt.coord_dir else os.path.join(opt.data_root, f'patch/{opt.patch_size}/coord')
        self.mask_dir = opt.mask_dir if opt.mask_dir else os.path.join(opt.data_root, f'patch/{opt.patch_size}/mask')
        self.stitch_dir = opt.stitch_dir if opt.stitch_dir else os.path.join(opt.data_root, f'patch/{opt.patch_size}/stitch')
        self.count_dir = opt.count_dir if opt.count_dir else os.path.join(opt.data_root, f'patch/{opt.patch_size}/')

        self.patch_size = opt.patch_size
        self.patch_level = opt.patch_level
        self.min_RGB = opt.min_RGB
        self.min_RGB_diffs = opt.min_RGB_diffs
        self.max_RGB_diffs = opt.max_RGB_diffs
        self.tissue_ratio = opt.tissue_ratio
        self.skip_done = opt.skip_done
        self.slide_list = opt.slide_list
        self.overlap = opt.overlap

        param_log(self)
        for directory in [self.slide_dir, self.coord_dir, self.mask_dir, self.stitch_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

    def save_hdf5(self, slide_name, coords, attr):
        coord_path = os.path.join(self.coord_dir, f'{slide_name}.h5')

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

    def get_coords(self, wsi, slide_name):
        (w, h) = wsi.level_dimensions[self.patch_level]
        mask = np.zeros((h, w), dtype=np.uint8)
        stitch = np.zeros((h, w, 3), dtype=np.uint8)
        coords = []
        step = int(self.patch_size * (1 - self.overlap))
        for h_index in range(0, h - self.patch_size, step):
            for w_index in range(0, w - self.patch_size, step):
                patch = wsi.read_region((w_index, h_index), self.patch_level, (self.patch_size, self.patch_size))
                patch = np.array(patch)

                difference = np.max(patch, axis=2) - np.min(patch, axis=2)

                index = (np.min(patch, axis=2) < self.min_RGB) & \
                        (difference > self.min_RGB_diffs) & \
                        (difference < self.max_RGB_diffs)

                # 检查组织占比
                if np.sum(index) / (self.patch_size ** 2) >= self.tissue_ratio:
                    coords.append([w_index, h_index])

                    cv2.rectangle(patch, (0, 0), (self.patch_size, self.patch_size), (0, 0, 0), 20)
                    line_spacing = self.patch_size // (5 + 1)
                    for i in range(5):
                        cv2.line(patch, (0, i * line_spacing), (self.patch_size, self.patch_size + i * line_spacing),
                                 (0, 0, 0), 5)
                        cv2.line(patch, (0, -i * line_spacing), (self.patch_size, self.patch_size - i * line_spacing),
                                 (0, 0, 0), 5)
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

                stitch[h_index:h_index + self.patch_size, w_index:w_index + self.patch_size] = patch
                mask[h_index:h_index + self.patch_size, w_index:w_index + self.patch_size][index] = 255

        mask_path = os.path.join(self.mask_dir, f'{slide_name}.png')
        mask = cv2.resize(mask, (w // 50, h // 50), interpolation=cv2.INTER_AREA)
        stitch_path = os.path.join(self.stitch_dir, f'{slide_name}.png')
        stitch = cv2.resize(stitch, (w // 50, h // 50), interpolation=cv2.INTER_AREA)

        cv2.imwrite(mask_path, mask)
        cv2.imwrite(stitch_path, stitch)

        if len(coords) > 0:
            coords = np.array(coords)
            logger.info(f'Extracted {len(coords)} coordinates')

            attr = {
                'patch_size': self.patch_size,
                'patch_level': self.patch_level,
                'level_dim': (w, h),
                'name': slide_name,
                'save_path': mask_path
            }
            self.save_hdf5(slide_name, coords, attr)

        return len(coords)

    def process_slide(self, slide, df):
        slide_name, slide_ext = os.path.splitext(os.path.basename(slide))
        logger.info(f"start to process {slide}")
        slide_path = os.path.join(self.slide_dir, slide)
        wsi = Aslide(slide_path) if '.kfb' in slide else OpenSlide(slide_path)
        coord_num = self.get_coords(wsi, slide_name)
        df.loc[len(df)] = {'slide_id': slide, 'coord_num': coord_num}
        logger.info(f"Finished processing: {slide}")

    @property
    def slides(self):
        if self.slide_list:
            return self.slide_list
        slides = [f for f in os.listdir(self.slide_dir) if os.path.isfile(os.path.join(self.slide_dir, f))]
        if self.skip_done:
            total_count = len(slides)
            slides_done = set(os.listdir(self.stitch_dir))
            logger.info(f'{len(slides_done)}/{total_count} slides already done')
            slides = [slide for slide in slides if slide.replace('svs', 'png') not in slides_done]
        return slides

    def run(self):
        csv_path = os.path.join(self.count_dir, f'count.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame(columns=['slide_id', 'coord_num'])

        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(self.process_slide, slide, df) for slide in self.slides]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    traceback.print_exc()
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV file created at {csv_path}")


parser = BaseOptions().parse()
parser.add_argument('--skip_done', type=bool, default=True)
parser.add_argument('--min_RGB', type=int, default=230, help='')
parser.add_argument('--min_RGB_diffs', type=int, default=30, help='组织部分 rgb 最小差异值')
parser.add_argument('--max_RGB_diffs', type=int, default=256, help='组织部分 rgb 最小差异值')
parser.add_argument('--overlap', type=float, default=0, help='patch 重叠率')
parser.add_argument('--tissue_ratio', type=float, default=0.3, help='组织占 patch 比重')
if __name__ == '__main__':
    args = parser.parse_args()
    PatchCoordsGenerator(args).run()
