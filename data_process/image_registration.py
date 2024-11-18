import json
import os
import shutil
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor

import h5py
import openslide
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from skimage import transform
from sklearn.model_selection import train_test_split
from torch.fx.experimental.unification.multipledispatch.dispatcher import source

from options.base_options import BaseOptions
from utils.core_utils import train

sys.path.insert(0, r'/data2/yhhu/LLB/Code/aslide')
from aslide import Aslide


def geojson2txt(filepath, typ='.geojson'):
    # 通常一批文件放在一个路径下
    file_list = os.listdir(filepath)
    for file in file_list:
        if not file.endswith(typ):
            continue
        json_path = os.path.join(filepath, file)
        txt_path = json_path.replace(typ, '.txt')
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        with open(txt_path, 'w') as txt:
            coordinates = json_data.get('features', [])[0].get('geometry', {}).get('coordinates', [])
            for coord in coordinates:
                x_value = coord[0]
                y_value = coord[1]
                txt.write("x:{}, y:{}\n".format(x_value, y_value))


def json2txt(filepath):
    file_list = os.listdir(filepath)
    for file in file_list:
        if not file.endswith('.json'):
            continue
        json_path = os.path.join(filepath, file)
        txt_path = json_path.replace('.json', '.txt')
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        with open(txt_path, 'w') as txt:
            for point in json_data:
                region = point.get("region", {})
                x, y = region.get("x", 0), region.get("y", 0)
                txt.write("x:{}, y:{}\n".format(x, y))


def get_points_from_txt(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            # 分割每一行的值
            values = line.strip().split(',')
            # 提取 x 和 y 值
            x_value = float(values[0].split(':')[1].strip())
            y_value = float(values[1].split(':')[1].strip())
            # 将值添加到列表中
            points.append([x_value, y_value])
    return np.array(points)


def get_square_corner_points(top_left_point, side_length):
    # 获取以 top_left_point 为左上角顶点，side_length 为边长的正方形四个角点
    x, y = top_left_point
    corner_pts = np.array([
        [x, y],  # 左上角
        [x + side_length, y],  # 右上角
        [x + side_length, y + side_length],  # 右下角
        [x, y + side_length]  # 左下角
    ], dtype=np.float32)
    return corner_pts


def warp_image_to_square(image, source_corners, output_size):
    """
    Warp an image to a square using perspective transformation.

    Parameters:
    - image: The original image (numpy array [h, w, channels]).
    - source_corners: The corner points of the region to warp (numpy array [4, 2]).
    - output_size: The size of the output square (int).

    Returns:
    - warped_image: The warped image as a square.
    """
    # Define the destination points for the perspective transformation
    img = np.array(image.convert('RGB'))
    dst_pts = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype=np.float32)

    # Estimate the perspective transformation matrix
    transform_model = transform.ProjectiveTransform()
    transform_model.estimate(np.array(source_corners, dtype=np.float32), dst_pts)

    # Apply the inverse perspective transformation to warp the image
    warped_image = transform.warp(img, transform_model.inverse, output_shape=(output_size, output_size))

    # Convert the warped image to uint8 type
    warped_image = (warped_image * 255).astype(np.uint8)

    return Image.fromarray(warped_image)


def affine_transform(points, a, b, c, d, e, f):
    """
    Apply an affine transformation to a set of 2D points.

    Parameters:
    - points: The input points (numpy array of shape [n, 2]).
    - a, b, c, d, e, f: The affine transformation parameters.

    Returns:
    - transformed_points: The transformed points (numpy array of shape [n, 2]).
    """
    # Perform the affine transformation
    # The points are expected to be in the form of [x, y]
    x_new = a * points[:, 0] + b * points[:, 1] + c
    y_new = d * points[:, 0] + e * points[:, 1] + f

    return np.column_stack((x_new, y_new)).flatten()


class Registration:
    def __init__(self, opt):
        self.data_root = opt.data_root
        self.slide_dir = os.path.join(opt.data_root, 'slides')
        self.patch_dir = os.path.join(opt.data_root, f'patch/{opt.patch_size}/pair')
        self.coord_dir = os.path.join(opt.data_root, f'patch/{opt.patch_size}/coord')
        self.points_dir = os.path.join(opt.data_root, f'points')
        self.transform_dir = os.path.join(opt.data_root, f'transform')
        self.he_dir = os.path.join(opt.data_root, f'patch/{opt.patch_size}/pair/he')
        self.ihc_dir = os.path.join(opt.data_root, f'patch/{opt.patch_size}/pair/ihc')
        self.regi_dir = os.path.join(opt.data_root, f'regi/{opt.patch_size}')

        self.patch_size = opt.patch_size
        self.patch_level = opt.patch_level
        self.transform_ori = opt.transform_ori
        self.ihc_ext = opt.ihc_ext
        self.alpha = opt.alpha
        self.slide_list = opt.slide_list

        for directory in [self.slide_dir, self.coord_dir, self.patch_dir, self.coord_dir, self.points_dir,
                          self.transform_dir, self.he_dir, self.ihc_dir, self.regi_dir]:
            os.makedirs(directory, exist_ok=True)

        json2txt(self.points_dir)

    def get_points_from_txt(self, slide):
        points = []
        points_path = os.path.join(self.points_dir, f'{slide}.txt')
        if not os.path.isfile(points_path):
            return
        with open(points_path, 'r') as file:
            for line in file:
                # 分割每一行的值
                values = line.strip().split(',')
                # 提取 x 和 y 值
                x_value = float(values[0].split(':')[1].strip())
                y_value = float(values[1].split(':')[1].strip())
                # 将值添加到列表中
                points.append([x_value, y_value])
        return np.array(points)

    def crop_image_and_adjust_corners(self, wsi, corners):
        """
        Crop an image and adjust the corner points accordingly.

        Parameters:
        - image: The original image (numpy array [h, w, channels]).
        - corners: The corner points of the region to crop (numpy array [4, 2]).
        - crop_dimension: The size of the crop (int).

        Returns:
        - cropped_image: The cropped image.
        - adjusted_corners: The corner points adjusted to the cropped image.
        """
        # Calculate the center of the region defined by the corner points
        width, height = wsi.level_dimensions[0]
        center_x = int(np.mean(corners[:, 0]))
        center_y = int(np.mean(corners[:, 1]))

        # Calculate the boundaries of the crop region
        half_crop = self.patch_size
        top = max(0, center_y - half_crop)
        bottom = min(height, center_y + half_crop)
        left = max(0, center_x - half_crop)
        right = min(width, center_x + half_crop)

        wsi_image = wsi.read_region((left, top), self.patch_level, (2 * self.patch_size, 2 * self.patch_size))
        if isinstance(wsi_image, np.ndarray):
            wsi_image = Image.fromarray(wsi_image)
        cropped_image = wsi_image.crop((0, 0, right - left, bottom - top))
        # Adjust the corner points to the cropped image
        adjusted_corners = corners.copy()
        adjusted_corners[:, 0] -= left
        adjusted_corners[:, 1] -= top

        return cropped_image, adjusted_corners

    def get_reg_param(self, filename):
        try:
            he_points = self.get_points_from_txt(filename)
            result = {}
            ihc_points = get_points_from_txt(os.path.join(self.points_dir, f'{filename}-{self.ihc_ext}.txt'))
            if self.transform_ori == "HE2IHC":
                points1, points2 = he_points, ihc_points.flatten()
            else:
                points1, points2 = ihc_points, he_points.flatten()
            popt, _ = curve_fit(affine_transform, points1, points2, p0=[0, 0, 0, 0, 0, 0])
            result[f'{filename}-{self.ihc_ext}.svs'] = list(popt)
            json_name = f'{filename}.json' if self.transform_ori == "HE2IHC" else f'{filename}-{self.ihc_ext}.json'
            result_json = os.path.join(self.transform_dir, json_name)
            with open(result_json, 'w') as f:
                json.dump(result, f)
        except:
            traceback.print_exc()
            return

    def merge_show_img(self, img1, img2, save_path):
        patch_size = self.patch_size

        w, h = patch_size * 2, patch_size * 2
        merged_image = Image.new("RGBA", (w, h))
        merged_image.paste(img1, (0, 0))
        merged_image.paste(img2, (patch_size, 0))
        img2_copy = img2.copy()
        half_size = patch_size // 2
        # Define the box coordinates for the quadrants
        # 左上，右上，左下，右下
        boxes = [
            (0, 0, half_size, half_size),
            (half_size, half_size, 2 * half_size, 2 * half_size)
        ]

        for box in boxes:
            sub_img1 = img1.crop(box)
            sub_img2 = img2_copy.crop(box)
            img2.paste(sub_img1, box)
            img1.paste(sub_img2, box)

        merged_image.paste(img1, (0, patch_size))
        merged_image.paste(img2, (patch_size, patch_size))

        for i in range(0, 3):
            y = i * h // 2
            ImageDraw.Draw(merged_image).line([(0, y), (w, y)], fill='black', width=5)
            x = i * w // 2
            ImageDraw.Draw(merged_image).line([(x, 0), (x, h)], fill='black', width=5)

        for x in range(0, w, 10):
            for y in [h // 4 * 3]:
                ImageDraw.Draw(merged_image).point((x, y), fill='black')
        for x in [w // 4, w // 4 * 3]:
            for y in range(h // 2, h, 10):
                ImageDraw.Draw(merged_image).point((x, y), fill='black')
        merged_image.save(save_path)
        plt.imshow(merged_image)
        plt.axis('on')
        plt.show()

    def registration(self, slide):
        slide_name, slide_ext = os.path.splitext(os.path.basename(slide))
        try:
            self.get_reg_param(slide_name)
            file_name = f'{slide_name}.json' if self.transform_ori == 'HE2IHC' else f'{slide_name}-{self.ihc_ext}.json'
            affine_path = os.path.join(self.transform_dir, file_name)
            with open(affine_path, 'r') as f:
                reg_params = json.load(f)
        except:
            traceback.print_exc()
            return

        file = h5py.File(os.path.join(self.coord_dir, f'{slide_name}.h5'), mode='r')
        src_points = list(file['coords'][:])
        he_path = os.path.join(self.slide_dir, slide)
        he_wsi = Aslide(he_path) if '.kfb' in slide else openslide.OpenSlide(he_path)


        ihc_path = os.path.join(self.slide_dir, f'{slide_name}-{self.ihc_ext}{slide_ext}')
        ihc_wsi = Aslide(ihc_path) if '.kfb' in slide else openslide.OpenSlide(ihc_path)
        a, b, c, d, e, f = reg_params[f'{slide_name}-{self.ihc_ext}.svs']

        for i, coord in enumerate(src_points):
            coord = (int(coord[0]), int(coord[1]))
            if self.transform_ori == 'HE2IHC':
                src_wsi, dst_wsi = he_wsi, ihc_wsi
            else:
                src_wsi, dst_wsi = ihc_wsi, he_wsi

            src_img = src_wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size))
            if isinstance(src_img, np.ndarray):
                src_img = Image.fromarray(src_img)
            dst_points = affine_transform(get_square_corner_points(coord, self.patch_size), a, b, c, d, e, f)
            dst_points = np.reshape(dst_points, (len(dst_points) // 2, 2))
            try:
                cropped_img, corner_points = self.crop_image_and_adjust_corners(dst_wsi, dst_points)
            except:
                continue
            ihc_img = warp_image_to_square(cropped_img, corner_points, self.patch_size)

            # he_img_path = os.path.join(self.he_dir, f'{slide_name}_{coord[0]}_{coord[1]}.png')
            # ihc_img_path = os.path.join(self.ihc_dir, f'{slide_name}_{coord[0]}_{coord[1]}.png')
            # he_img.save(he_img_path)
            # ihc_img.save(ihc_img_path)
            save_path = os.path.join(self.regi_dir, f'{slide_name}-{self.ihc_ext}_{coord[0]}_{coord[1]}.png')
            if (i + 1) % (len(src_points) // 20) == 0:
                self.merge_show_img(src_img, ihc_img, save_path)

    @property
    def slides(self):
        if self.slide_list:
            return self.slide_list
        else:
            slides = [f for f in os.listdir(self.slide_dir) if os.path.isfile(os.path.join(self.slide_dir, f))]
            slides = [slide for slide in slides if not any(s in slide for s in self.ihc_ext)]
            points = os.listdir(self.points_dir)
            return [slide for slide in slides if any(os.path.splitext(slide)[0] in pt for pt in points)]

    def run(self):

        with ThreadPoolExecutor(max_workers=8) as executor:
            [executor.submit(self.registration, slide) for slide in self.slides]

        # self.split_data()

    def split_data(self):
        train_a_dir = os.path.join(self.patch_dir, 'trainA')
        train_b_dir = os.path.join(self.patch_dir, 'trainB')
        test_a_dir = os.path.join(self.patch_dir, 'testA')
        test_b_dir = os.path.join(self.patch_dir, 'testB')
        for directory in [train_a_dir, train_b_dir, test_a_dir, test_b_dir]:
            os.makedirs(directory, exist_ok=True)

        images = os.listdir(self.he_dir)
        train_set, test_set = train_test_split(images, test_size=0.2, random_state=42)
        for img in train_set:
            shutil.copy(os.path.join(self.he_dir, img), os.path.join(train_a_dir, img))
            shutil.copy(os.path.join(self.ihc_dir, img), os.path.join(train_b_dir, img))

        for img in test_set:
            shutil.copy(os.path.join(self.he_dir, img), os.path.join(test_a_dir, img))
            shutil.copy(os.path.join(self.ihc_dir, img), os.path.join(test_b_dir, img))


parser = BaseOptions().parse()
parser.add_argument('--alpha', type=int, default=100, help='')
parser.add_argument('--transform_ori', type=str, default='IHC2HE')
parser.add_argument('--ihc_ext', type=str, default='CK')
if __name__ == '__main__':
    args = parser.parse_args()
    Registration(args).run()
