import argparse


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--data_root', type=str, default=r'/data2/yhhu/LLB/Data/前列腺癌数据/test/')
        self.parser.add_argument('--patch_size', type=int, default=256)
        self.parser.add_argument('--patch_level', type=int, default=0)

        # 提取过程中使用默认文件夹结构
        self.parser.add_argument('--slide_dir', type=str)
        self.parser.add_argument('--coord_dir', type=str)
        self.parser.add_argument('--mask_dir', type=str)
        self.parser.add_argument('--stitch_dir', type=str)
        self.parser.add_argument('--count_dir', type=str)

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        return self.parser
