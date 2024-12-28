import json
import math
import os
import pathlib
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


class BaseContour:
    def __init__(self, options):
        self.slide_dir = options.slide_dir if options.slide_dir else os.path.join(options.data_root, 'slides')
        self.patch_dir = options.patch_dir if options.patch_dir else os.path.join(options.data_root, f'patch/{options.patch_size}/image')
        self.contour_dir = options.contour_dir if options.contour_dir else os.path.join(options.data_root, f'patch/{options.patch_size}/contour')
        self.ihc_slide_dir = os.path.join(options.data_root, 'IHC')
        self.points_dir = os.path.join(options.data_root, f'points')
        self.patch_size = options.patch_size
        self.ihc_ext = options.ihc_ext
        self.slide_list = options.slide_list

        self.skip_done = options.skip_done

        param_log(self)
        for directory in [self.patch_dir, self.contour_dir]:
            os.makedirs(directory, exist_ok=True)

    def run(self, slide):
        raise NotImplementedError

    def cv_contour(self, patch: str | pathlib.Path):
        # 根据色彩范围，使用 opencv 框出目标
        lower_bound = np.array([5, 5, 10])
        upper_bound = np.array([220, 220, 185])

        image_path = os.path.join(self.patch_dir, patch)
        image = cv2.imread(image_path)
        image = cv2.copyMakeBorder(image, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # 分离暗区和亮区
        mask = cv2.inRange(image, lower_bound, upper_bound)
        dark_region = cv2.bitwise_not(mask)

        # 寻找暗区中的轮廓
        cnt_info = cv2.findContours(dark_region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return cnt_info

    def qupath_feature(self, cnt_info, patch):
        # 获取 qupath 格式数据
        match = re.search(r'_(\d+)_(\d+)\.jpg', patch)
        x_str, y_str = match.groups()
        x, y = int(x_str), int(y_str)
        contours, hierarchy = cnt_info
        features = []

        for cnt, h in zip(contours, hierarchy[0]):
            area = cv2.contourArea(cnt)
            patch_area = (self.patch_size - 1) ** 2
            parent_area = cv2.contourArea(contours[h[3]]) if h[3] != -1 else float('inf')

            # 存在父contour 且 父contour不为整张图的  且 父contour面积远大于子contour 且 子contour面积很小
            if patch_area // 6000 < area < patch_area and not (h[3] != -1 and parent_area < patch_area and area < parent_area // 200):

                # 轮廓平滑 CV2 的平滑太锐利了
                start = 0
                while start < len(cnt):
                    start_point = cnt[start][0]
                    for i, end_point in enumerate(cnt[start + 5: start + 25]):
                        end_point = end_point[0]
                        if math.sqrt((start_point[0] - end_point[0]) ** 2 + (start_point[1] - end_point[1]) ** 2) < 25:
                            cnt = np.vstack((cnt[:start], cnt[start + i + 5:]))
                            break
                    start += 1

                cnt += np.array([x - 3, y - 3])
                cnt = cnt.reshape((-1, 2))
                cnt = cnt.tolist()
                cnt.append(cnt[0])

                if len(cnt) > 3:
                    feature = {
                        "type": "Feature",
                        "id": str(uuid.uuid4()),
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [cnt],
                        }
                    }
                    features.append(feature)
        return features

    @property
    def slides(self):
        if self.slide_list:
            slides = self.slide_list
        else:
            slides = [f for f in os.listdir(self.slide_dir) if os.path.isfile(os.path.join(self.slide_dir, f))]
        if True:
            points = [os.path.splitext(p)[0] for p in os.listdir(self.points_dir)]
            ihc_wsi = [os.path.splitext(p)[0] for p in os.listdir(self.ihc_slide_dir)]
            slides = [slide for slide in slides if f'{os.path.splitext(slide)[0]}-{self.ihc_ext}' in ihc_wsi]  # 过滤无配对数据
            slides = [slide for slide in slides if os.path.splitext(slide)[0] in points]  # 过滤无标点数据
        return slides

    def parallel_run(self):
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(self.run, slide) for slide in self.slides]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    traceback.print_exc()


class HeatmapContour(BaseContour):
    def __init__(self, options):
        super().__init__(options)
        self.heatmap_coord_dir = options.heatmap_coord_dir if options.heatmap_coord_dir else os.path.join(options.data_root, f'patch/{options.patch_size}/heatmap_coord')

    def heatmap_coords(self, slide):
        slide_id, _ = os.path.splitext(slide)
        full_path = os.path.join(self.heatmap_coord_dir, f'{slide_id}.h5')
        with h5py.File(full_path, 'r') as hdf5_file:
            coords = hdf5_file['coords'][:]
        return coords

    def run(self, slide):
        slide_name = os.path.splitext(slide)[0]
        logger.info(f'start to process {slide_name}')
        features = []
        patches = os.listdir(self.patch_dir)
        for patch in patches:
            if not patch.startswith(f'{slide_name}') or not os.path.exists(os.path.join(self.patch_dir, patch)):
                continue
            cnt_info = self.cv_contour(patch)
            features = self.qupath_feature(cnt_info, patch)
        features = self.heatmap_judge(features, slide)

        geojson = {"type": "FeatureCollection", "features": features}
        with open(os.path.join(self.contour_dir, f'{slide_name}.geojson'), 'w') as f:
            json.dump(geojson, f, indent=2)
            logger.info(f'generated {slide_name}.geojson contour json!!!')

    def heatmap_judge(self, features, slide):
        # 根据热力值判断有癌无癌
        h_coords = self.heatmap_coords(slide)

        for feature in features:
            coords = feature.get('geometry').get('coordinates')
            for h_coord in h_coords:
                w, h = h_coord[0], h_coord[1]
                count = 0
                for coord in coords[0]:
                    if w < coord[0] < w + 256 and h < coord[1] < h + 256:
                        count += 1
                if count > 0:
                    feature.update({"properties": {
                        "objectType": "annotation",
                        "classification": {"name": "cancer", "color": [255, 0, 0]}
                    }})
                    break
                else:
                    feature.update({"properties": {
                        "objectType": "annotation",
                        "classification": {"name": "non-cancer", "color": [0, 255, 0]}
                    }})
        return features


class PatchContour(BaseContour):
    def run(self, slide):
        slide_name = os.path.splitext(slide)[0]
        logger.info(f'start to process {slide_name}')
        features = []
        patches = os.listdir(self.patch_dir)

        for patch in patches:
            if not patch.startswith(f'{slide_name}') or not os.path.exists(os.path.join(self.patch_dir, patch)):
                continue
            cnt_info = self.cv_contour(patch)
            feature = self.qupath_feature(cnt_info, patch)
            features.extend(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        with open(os.path.join(self.contour_dir, f'{slide_name}.geojson'), 'w') as f:
            json.dump(geojson, f, indent=2)
            logger.info(f'generated {slide_name}.geojson contour json!!!')


parser = BaseOptions().parse()
parser.add_argument('--contour_dir', type=str, default=None, help='contour dir')
parser.add_argument('--transform_dir', type=str, default=None, help='transform dir')
parser.add_argument('--transform', type=bool, default=False)
parser.add_argument('--transform_ori', type=str, default='IHC2HE')
parser.add_argument('--ihc_ext', type=str, default='CK')
if __name__ == '__main__':
    args = parser.parse_args()
    PatchContour(args).parallel_run()
