import os
import traceback
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from loguru import logger


class ImageCropper:
    def __init__(self, opt):
        self.data_root = opt.data_root
        self.he_dir = opt.he_dir if opt.he_dir else os.path.join(opt.data_root, f'pair/{opt.patch_size}/{opt.ihc_ext}/he/')
        self.ihc_dir = opt.ihc_dir if opt.ihc_dir else os.path.join(opt.data_root, f'pair/{opt.patch_size}/{opt.ihc_ext}/dhr/reg_ihc/')
        self.out_dir = opt.output_dir if opt.output_dir else os.path.join(opt.data_root, f'pair/{opt.output_size}/{opt.ihc_ext}/')
        self.patch_size = opt.patch_size
        self.output_size = opt.output_size
        self.crop_size = opt.crop_size

        if self.output_size > self.patch_size:
            raise Exception(f'Output size {self.output_size} > patch size {self.patch_size}， Suggest: Mask 256, patch 1024')

        self.tmp_he_dir = os.path.join(self.out_dir, f'tmp/he')
        self.tmp_ihc_dir = os.path.join(self.out_dir, f'tmp/ihc')
        os.makedirs(self.tmp_he_dir, exist_ok=True)
        os.makedirs(self.tmp_ihc_dir, exist_ok=True)

    def crop_image(self, img):
        i = 0
        base, _ = os.path.splitext(img)
        he_img = Image.open(os.path.join(self.he_dir, img))
        ihc_img = Image.open(os.path.join(self.ihc_dir, f'{base}.jpg'))
        times = self.patch_size // self.crop_size - 1
        for j in range(self.crop_size // 4, self.crop_size * times, int(self.crop_size // 1.5)):
            for k in range(self.crop_size // 4, self.crop_size * times, int(self.crop_size // 1.5)):
                sub_he = he_img.crop((j, k, j + self.crop_size, k + self.crop_size))
                sub_ihc = ihc_img.crop((j, k, j + self.crop_size, k + self.crop_size))
                sub_he = sub_he.resize((self.output_size, self.output_size))
                sub_ihc = sub_ihc.resize((self.output_size, self.output_size))
                sub_he.save(os.path.join(self.tmp_he_dir, f'{base}_{i}.png'))
                sub_ihc.save(os.path.join(self.tmp_ihc_dir, f'{base}_{i}.png'))

                i += 1
        logger.info(f'裁切完成，共生成图片{i}条')

    def parallel_run(self):
        he_images = os.listdir(self.he_dir)
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(self.crop_image, img) for img in he_images if img.endswith('.png')]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    traceback.print_exc()


parser = ArgumentParser()
parser.add_argument('--data_root', type=str)
parser.add_argument('--he_dir', type=str)
parser.add_argument('--ihc_dir', type=str)
parser.add_argument('--ihc_ext', type=str)
parser.add_argument('--patch_size', type=int, default=4096)
parser.add_argument('--crop_size', type=int, default=1024)
parser.add_argument('--output_size', type=int, default=1024)
parser.add_argument('--output_dir', type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    processor = ImageCropper(args)
    processor.parallel_run()
