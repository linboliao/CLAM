import json
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from loguru import logger

from image_registration import affine_transform
from options.base_options import BaseOptions


class Contour:
    def __init__(self, opt):
        self.slide_dir = os.path.join(opt.data_root, 'slides')
        self.patch_dir = os.path.join(opt.data_root, f'patch/{opt.patch_size}/image')
        self.contour_dir = os.path.join(opt.data_root, f'patch/{opt.patch_size}/contour')
        self.transform_dir = os.path.join(opt.data_root, f'transform')
        self.transform = opt.transform
        self.transform_ori = opt.transform_ori
        self.patch_size = opt.patch_size
        self.ihc = opt.ihc

        for directory in [self.slide_dir, self.patch_dir, self.contour_dir, self.transform_dir]:
            os.makedirs(directory, exist_ok=True)

    def contour(self, slide):
        logger.info(f'start to process {slide}')
        patches = os.listdir(self.patch_dir)
        features = []
        affine_features = []
        for patch in patches:
            if not patch.startswith(slide):
                continue
            cnt_info = self.get_contours(patch)
            f, af = self.get_features(cnt_info, patch, slide)
            if f:
                features.extend(f)
            if af:
                affine_features.extend(af)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        with open(os.path.join(self.contour_dir, f'{slide}.geojson'), 'w') as f:
            json.dump(geojson, f, indent=2)
            logger.info(f'generated {slide} contour json!!!')
        if self.transform:
            affine_geojson = {
                "type": "FeatureCollection",
                "features": affine_features
            }
            with open(os.path.join(self.contour_dir, f"{slide}-{self.transform_ori}.geojson"), 'w') as f:
                json.dump(affine_geojson, f, indent=2)

            logger.info(f'generated {slide}-{self.transform_ori} contour json!!!')

    def get_contours(self, patch: str):
        lower_bound = np.array([20, 20, 30])
        upper_bound = np.array([130, 130, 160])

        image_path = os.path.join(self.patch_dir, patch)
        image = cv2.imread(image_path)
        image = cv2.copyMakeBorder(image, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # 分离暗区和亮区
        mask = cv2.inRange(image, lower_bound, upper_bound)
        dark_region = cv2.bitwise_not(mask)

        # 寻找暗区中的轮廓
        cnt_info = cv2.findContours(dark_region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return cnt_info

    def get_features(self, cnt_info, patch, slide_name):
        match = re.search(r'_(\d+)_(\d+)\.jpg', patch)
        x_str, y_str = match.groups()
        x, y = int(x_str), int(y_str)
        contours, hierarchy = cnt_info
        features = []
        affine_features = []

        for cnt, h in zip(contours, hierarchy[0]):

            area = cv2.contourArea(cnt)

            if self.patch_size // 15 < area < (self.patch_size - 1) ** 2:
                if h[3] != -1 and cv2.contourArea(contours[h[3]]) < (
                        self.patch_size - 1) ** 2 and area < cv2.contourArea(contours[h[3]]) // 7 and area < (
                        self.patch_size - 1) ** 2 // 500:
                    continue
                cnt += np.array([x - 3, y - 3])
                cnt = cnt.reshape((-1, 2))
                cnt = np.vstack((cnt, cnt[0]))  # 闭环

                if self.transform:
                    with open(os.path.join(self.transform_dir, f'{slide_name}-{self.transform_ori}.json'), 'r') as f:
                        reg_params = json.load(f)
                    # 后缀要改，一个文件存多个参数的情况
                    a, b, c, d, e, f = reg_params[f'{slide_name}.svs']
                    affine_cnt = affine_transform(cnt, a, b, c, d, e, f)
                    affine_cnt = np.reshape(affine_cnt, (len(affine_cnt) // 2, 2))
                    affine_cnt = np.round(affine_cnt)
                    affine_feature = {
                        "type": "Feature",
                        "id": str(uuid.uuid4()),
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [affine_cnt.tolist()]
                        }
                    }
                    affine_features.append(affine_feature)

                feature = {
                    "type": "Feature",
                    "id": str(uuid.uuid4()),
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [cnt.tolist()]
                    }
                }
                features.append(feature)
        return features, affine_features

    def run(self):
        slides = [os.path.splitext(os.path.basename(file))[0] for file in os.listdir(self.slide_dir) if
                  self.ihc in file]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.contour, slide) for slide in slides]
            for future in futures:
                future.result()


parser = BaseOptions().parse()
parser.add_argument('--transform', type=bool, default=True)
parser.add_argument('--transform_ori', type=str, default='IHC2HE')
parser.add_argument('--ihc', type=str, default='CKpan')
if __name__ == '__main__':
    args = parser.parse_args()
    Contour(args).run()
