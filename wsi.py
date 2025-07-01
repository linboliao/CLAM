import os
import random
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import openslide
from PIL import Image
from openslide import lowlevel

sys.path.insert(0, r'/data2/lbliao/Code/aslide/')
from aslide import Aslide

sys.path.insert(1, r'/data2/lbliao/Code/opensdpc/')
from opensdpc.opensdpc import OpenSdpc


def is_background(img, threshold=20):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    diff = np.ptp(img, axis=2)  # ptp直接计算max-min
    return (diff > threshold).mean() < 0.4


class WSIOperator(openslide.OpenSlide):
    def __init__(self, filename):
        if isinstance(filename, str):
            filename = Path(filename)
        suffix = filename.suffix.lower()
        self.suffix = suffix
        if self.suffix == '.kfb':
            slide = Aslide(str(filename))
            self.mpp = slide.mpp
        elif self.suffix == '.sdpc':
            slide = OpenSdpc(str(filename))
            self.mpp = 20
        else:
            slide = openslide.OpenSlide(str(filename))
            self.mpp = int(slide.properties.get('aperio.AppMag', '20'))
        self.wsi = slide

    def read_region(self, location, level, size, check_background=False):
        """统一区域读取接口"""
        img = self.wsi.read_region(location, level, size)
        if check_background and is_background(img):
            return None
        if isinstance(img, np.ndarray):
            return Image.fromarray(img)
        return img

    @property
    def level_dimensions(self):
        """获取原始尺寸"""
        return self.wsi.level_dimensions