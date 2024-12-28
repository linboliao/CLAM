import argparse
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image
from loguru import logger
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default=r'/data2/lbliao/Data/MSI/')
parser.add_argument('--ihc_ext', type=str, default='M6')
parser.add_argument('--patch_size', type=int, default=1024)
parser.add_argument('--A_dir', type=str)
parser.add_argument('--B_dir', type=str)
parser.add_argument('--ssim_threshold', type=float, default=0.2)

args = parser.parse_args()

he_dir = args.A_dir if args.A_dir else rf'{args.data_root}/pair/{args.patch_size}/{args.ihc_ext}/tmp/he'
ihc_dir = args.B_dir if args.B_dir else rf'{args.data_root}/pair/{args.patch_size}/{args.ihc_ext}/tmp/ihc'
dst_he_dir = he_dir.replace('tmp', 'dst')
dst_ihc_dir = ihc_dir.replace('tmp', 'dst')
os.makedirs(dst_he_dir, exist_ok=True)
os.makedirs(dst_ihc_dir, exist_ok=True)
pair_dir = rf'{args.data_root}/pair/{args.patch_size}/'
he_dir = rf'{args.data_root}/pair/{args.patch_size}/{args.ihc_ext}/tmp/he'
ihc_dir = rf'{args.data_root}/pair/{args.patch_size}/{args.ihc_ext}/tmp/ihc'
ihc_ext = args.ihc_ext


# 相似度过滤
def ssim_filter(filename, test=False):
    he_path = os.path.join(he_dir, filename)
    ihc_path = os.path.join(ihc_dir, filename)

    img1 = Image.open(he_path)
    img2 = Image.open(ihc_path)
    img1_gray = img1.convert('L')  # 'L'模式表示灰度图
    img2_gray = img2.convert('L')
    img1_gray_np = np.array(img1_gray)
    img2_gray_np = np.array(img2_gray)
    ssim_value = ssim(img1_gray_np, img2_gray_np)
    if test and ssim_value > args.ssim_threshold:
        dst_he_path = os.path.join(dst_he_dir, filename)
        dst_ihc_path = os.path.join(dst_ihc_dir, filename)
        shutil.copy(he_path, dst_he_path)
        shutil.copy(ihc_path, dst_ihc_path)
    elif not test and ssim_value < args.ssim_threshold:
        os.remove(he_path)
        os.remove(ihc_path)
        logger.info(f'remove {filename}, ssim {ssim_value}')

    return ssim_value, filename


def parallel_run():
    results = []

    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_file = {executor.submit(ssim_filter, filename): filename for filename in os.listdir(he_dir)}

        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                ssim_value, _ = future.result()
                results.append(ssim_value)
                logger.info(f"file: {filename}, ssim: {ssim_value}")
            except Exception as exc:
                logger.error(f"file: {filename} generated an exception: {exc}")


# 过滤没有腺体的图片
def ann_filter():
    ann = r'/data2/lbliao/Data/前列腺癌数据/CKPan/pair/1024-1/dst/label/'
    hes = os.listdir(dst_he_dir)
    anns = os.listdir(ann)
    for he in hes:
        base, ext = os.path.splitext(he)
        if f'{base}.txt' not in anns:
            os.remove(os.path.join(dst_he_dir, he))
            os.remove(os.path.join(ihc_dir, he))
            logger.info(f'{he}没有腺体')


# 图像大小过滤
def size_filter(size: int):
    ann_dir = r'/data2/lbliao/Data/前列腺癌数据/CKPan/pair/1024-1/tmp/label/'

    # 遍历目录
    for filename in os.listdir(he_dir):
        base, _ = os.path.splitext(filename)
        he_path = os.path.join(he_dir, filename)
        ihc_path = os.path.join(ihc_dir, filename)
        ann_path = os.path.join(ann_dir, f"{base}.txt")

        if os.path.isfile(he_path) and os.path.isfile(ihc_path):
            filesize = os.path.getsize(ihc_path)

            if filesize < size * 1024:
                os.remove(he_path)
                os.remove(ihc_path)
                if os.path.isfile(ann_path):
                    os.remove(ann_path)
                logger.info(f"Deleted {filename} due to size less than {size}KB")


# 指定你的文件夹路径
def split_data():
    he_dir = r'/data2/lbliao/Data/前列腺癌数据/CKPan/pair/1024-1/dst/he/'
    ihc_dir = r'/data2/lbliao/Data/前列腺癌数据/CKPan/pair/1024-1/dst/ihc/'
    ann_path = r'/data2/lbliao/Data/前列腺癌数据/CKPan/pair/1024-1/dst/label/'

    img_dir = r'/data2/lbliao/Data/前列腺癌数据/CKPan/pair/1024-1/yolo'
    label_dir = r'/data2/lbliao/Data/前列腺癌数据/CKPan/pair/1024-1/labels/'

    he_imgs = os.listdir(he_dir)
    length = len(he_imgs)
    train = int(length * 0.8)
    for i in range(train):
        base, ext = os.path.splitext(he_imgs[i])
        img = os.path.join(img_dir, 'train/images')
        label = os.path.join(img_dir, 'train/labels')
        os.makedirs(img, exist_ok=True)
        os.makedirs(label, exist_ok=True)
        shutil.copy(os.path.join(he_dir, he_imgs[i]), os.path.join(img, he_imgs[i]))
        shutil.copy(os.path.join(ann_path, f'{base}.txt'), os.path.join(label, f'{base}.txt'))

    for i in range(train, length):
        base, ext = os.path.splitext(he_imgs[i])
        img = os.path.join(img_dir, 'val/images')
        label = os.path.join(img_dir, 'val/labels')
        os.makedirs(img, exist_ok=True)
        os.makedirs(label, exist_ok=True)
        shutil.copy(os.path.join(he_dir, he_imgs[i]), os.path.join(img, he_imgs[i]))
        shutil.copy(os.path.join(ann_path, f'{base}.txt'), os.path.join(label, f'{base}.txt'))


def gland_filter():
    train_a_dir = he_dir
    train_b_dir = ihc_dir
    target_a = he_dir.replace('1024', '1024-f')
    target_b = ihc_dir.replace('1024', '1024-f')
    os.makedirs(target_a, exist_ok=True)
    os.makedirs(target_b, exist_ok=True)
    patch_size = 1024
    ihc_images = os.listdir(train_b_dir)
    for img in ihc_images:
        lower_bound = np.array([5, 5, 10])
        upper_bound = np.array([150, 220, 180])
        img_path = os.path.join(train_b_dir, img)
        image = cv2.imread(img_path)
        image = cv2.copyMakeBorder(image, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # 分离暗区和亮区
        mask = cv2.inRange(image, lower_bound, upper_bound)
        dark_region = cv2.bitwise_not(mask)

        # 寻找暗区中的轮廓
        contours, hierarchy = cv2.findContours(dark_region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        total_area = 0
        patch_area = (patch_size - 1) ** 2
        for cnt, h in zip(contours, hierarchy[0]):
            area = cv2.contourArea(cnt)
            parent_area = cv2.contourArea(contours[h[3]]) if h[3] != -1 else float('inf')

            # 存在父contour 且 父contour不为整张图的  且 父contour面积远大于子contour 且 子contour面积很小
            if patch_area // 600 < area < patch_area and (h[3] != -1 and patch_area <= parent_area):
                total_area += area
        if total_area > patch_area * 0.05 or random.random() > 0.5:
            shutil.copy(os.path.join(train_a_dir, img), os.path.join(target_a, img))
            shutil.copy(os.path.join(train_b_dir, img), os.path.join(target_b, img))


def miss_filter():
    train_a_dir = he_dir
    train_b_dir = ihc_dir
    he_images = os.listdir(train_a_dir)
    ihc_images = os.listdir(train_b_dir)
    for img in he_images:
        he_img = os.path.join(train_a_dir, img)
        ihc_img = os.path.join(train_b_dir, img)
        if not os.path.isfile(ihc_img):
            os.remove(he_img)
    for img in ihc_images:
        he_img = os.path.join(train_a_dir, img)
        ihc_img = os.path.join(train_b_dir, img)
        if not os.path.isfile(he_img):
            os.remove(ihc_img)


# parallel_run()
# gland_filter()
miss_filter()


def split_data():
    train_a_dir = os.path.join(pair_dir, f'{ihc_ext}/trainA')
    train_b_dir = os.path.join(pair_dir, f'{ihc_ext}/trainB')
    test_a_dir = os.path.join(pair_dir, f'{ihc_ext}/testA')
    test_b_dir = os.path.join(pair_dir, f'{ihc_ext}/testB')
    for directory in [train_a_dir, train_b_dir, test_a_dir, test_b_dir]:
        os.makedirs(directory, exist_ok=True)

    images = os.listdir(he_dir)
    train_set, test_set = train_test_split(images, test_size=0.2, random_state=42)
    for img in train_set:
        if os.path.isfile(os.path.join(he_dir, img)) and os.path.isfile(os.path.join(ihc_dir, img)):
            shutil.copy(os.path.join(he_dir, img), os.path.join(train_a_dir, img))
            shutil.copy(os.path.join(ihc_dir, img), os.path.join(train_b_dir, img))

    for img in test_set:
        if os.path.isfile(os.path.join(he_dir, img)) and os.path.isfile(os.path.join(ihc_dir, img)):
            shutil.copy(os.path.join(he_dir, img), os.path.join(test_a_dir, img))
            shutil.copy(os.path.join(ihc_dir, img), os.path.join(test_b_dir, img))


split_data()
