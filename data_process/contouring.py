import json
import math
import os
import re
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import h5py
import numpy as np
from loguru import logger

from image_registration import affine_transform
from options.base_options import BaseOptions
from utils.utils import param_log


def get_coords():
    full_path = os.path.join('/data2/lbliao/Data/前列腺癌数据/CKPan/patch/1024/tmp_coord/', '1547583.10.h5')
    with h5py.File(full_path, 'r') as hdf5_file:
        coords = hdf5_file['coords'][:]
    return coords


class Contour:
    def __init__(self, opt):
        self.slide_dir = opt.slide_dir if opt.slide_dir else os.path.join(opt.data_root, 'slides')
        self.patch_dir = opt.patch_dir if opt.patch_dir else os.path.join(opt.data_root, f'patch/{opt.patch_size}/image')
        self.contour_dir = opt.contour_dir if opt.contour_dir else os.path.join(opt.data_root, f'patch/{opt.patch_size}/contour')
        self.transform_dir = opt.transform_dir if opt.transform_dir else os.path.join(opt.data_root, f'transform')
        self.transform = opt.transform
        self.transform_ori = opt.transform_ori
        self.patch_size = opt.patch_size
        self.ihc_ext = opt.ihc_ext
        self.slide_list = opt.slide_list

        param_log(self)
        for directory in [self.patch_dir, self.contour_dir, self.transform_dir]:
            os.makedirs(directory, exist_ok=True)

    def contour(self, slide):
        slide_name = os.path.splitext(slide)[0]
        logger.info(f'start to process {slide_name}')
        # patches = os.listdir(self.patch_dir)
        coords = get_coords()
        patches = [f'{slide_name}_{coord[0]}_{coord[1]}.jpg' for coord in coords]
        features = []
        affine_features = []
        for patch in patches:
            if not patch.startswith(f'{slide_name}') or not os.path.exists(os.path.join(self.patch_dir, patch)):
                continue
            cnt_info = self.get_contours(patch)
            f, af = self.get_features(cnt_info, patch, slide_name, {"name": "cancer", "color": [255, 0, 0]})
            if f:
                features.extend(f)
            if af:
                affine_features.extend(af)
        new_patches = os.listdir(self.patch_dir)
        new_patches = [patch for patch in new_patches if patch not in patches]
        for patch in new_patches:
            if not patch.startswith(f'{slide_name}') or not os.path.exists(os.path.join(self.patch_dir, patch)):
                continue
            cnt_info = self.get_contours(patch)
            f, af = self.get_features(cnt_info, patch, slide_name, {"name": "non-cancer", "color": [0, 255, 0]})
            if f:
                features.extend(f)
            if af:
                affine_features.extend(af)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        with open(os.path.join(self.contour_dir, f'{slide_name}.geojson'), 'w') as f:
            json.dump(geojson, f, indent=2)
            logger.info(f'generated {slide_name}.geojson contour json!!!')
        if self.transform:
            affine_geojson = {
                "type": "FeatureCollection",
                "features": affine_features
            }
            with open(os.path.join(self.contour_dir, f"{slide_name.replace(f'-{self.ihc_ext}', '')}.geojson"), 'w') as f:
                json.dump(affine_geojson, f, indent=2)

            logger.info(f'generated {slide_name.replace(f"-{self.ihc_ext}", "")}.geojson contour json!!!')

    def get_contours(self, patch: str):
        lower_bound = np.array([15, 15, 30])
        upper_bound = np.array([140, 140, 160])

        image_path = os.path.join(self.patch_dir, patch)
        image = cv2.imread(image_path)
        image = cv2.copyMakeBorder(image, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # 分离暗区和亮区
        mask = cv2.inRange(image, lower_bound, upper_bound)
        dark_region = cv2.bitwise_not(mask)

        # 寻找暗区中的轮廓
        cnt_info = cv2.findContours(dark_region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return cnt_info

    def get_features(self, cnt_info, patch, slide_name, clazz):
        match = re.search(r'_(\d+)_(\d+)\.jpg', patch)
        x_str, y_str = match.groups()
        x, y = int(x_str), int(y_str)
        contours, hierarchy = cnt_info
        features = []
        affine_features = []
        for cnt, h in zip(contours, hierarchy[0]):
            epsilon = 0.0005 * cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, epsilon, True)
            area = cv2.contourArea(cnt)
            patch_area = (self.patch_size - 1) ** 2
            parent_area = cv2.contourArea(contours[h[3]]) if h[3] != -1 else float('inf')

            # 存在父contour 且 父contour不为整张图的  且 父contour面积远大于子contour 且 子contour面积很小
            if patch_area // 200 < area < patch_area and not (h[3] != -1 and parent_area < patch_area and area < parent_area // 4):
                # hull = cv2.convexHull(cnt, returnPoints=False)
                # defects = cv2.convexityDefects(cnt, hull)
                # for i in range(defects.shape[0]):
                #     start, end, far, depth = defects[i, 0]
                #     cnt = np.vstack((cnt[:start], cnt[end + 1:]))
                for _ in range(3):
                    start = 0
                    while start < len(cnt):
                        start_point = cnt[start][0]
                        for i, end_point in enumerate(cnt[start + 3: start + 25]):
                            end_point = end_point[0]
                            if math.sqrt((start_point[0] - end_point[0]) ** 2 + (start_point[1] - end_point[1]) ** 2) < 4:
                                cnt = np.vstack((cnt[:start], cnt[start + i:]))
                                break
                        start += 1

                cnt += np.array([x - 3, y - 3])
                cnt = cnt.reshape((-1, 2))
                cnt = cnt.tolist()
                cnt.append(cnt[0])

                if self.transform:
                    with open(os.path.join(self.transform_dir, f'{slide_name}.json'), 'r') as f:
                        reg_params = json.load(f)
                    # TODO 一个 HE 对应多个 IHC
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
                if len(cnt) < 3:
                    continue
                feature = {
                    "type": "Feature",
                    "id": str(uuid.uuid4()),
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [cnt],
                    },
                    "properties": {
                        "objectType": "annotation",
                        "classification": clazz
                    }
                }
                features.append(feature)
        return features, affine_features

    @property
    def slides(self):
        if self.slide_list:
            return self.slide_list
        else:
            return [file for file in os.listdir(self.slide_dir) if self.ihc_ext in file]

    def run(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.contour, slide) for slide in self.slides]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    traceback.print_exc()


parser = BaseOptions().parse()
parser.add_argument('--contour_dir', type=str, default=None, help='contour dir')
parser.add_argument('--transform_dir', type=str, default=None, help='transform dir')
parser.add_argument('--transform', type=bool, default=False)
parser.add_argument('--transform_ori', type=str, default='IHC2HE')
parser.add_argument('--ihc_ext', type=str, default='CK')
if __name__ == '__main__':
    args = parser.parse_args()
    Contour(args).run()
