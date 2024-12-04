import argparse


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--data_root', type=str)
        self.parser.add_argument('--patch_size', type=int)
        self.parser.add_argument('--patch_level', type=int, default=0)

        # 提取过程中使用默认文件夹结构
        self.parser.add_argument('--slide_dir', type=str)
        self.parser.add_argument('--coord_dir', type=str)
        self.parser.add_argument('--mask_dir', type=str)
        self.parser.add_argument('--stitch_dir', type=str)
        self.parser.add_argument('--count_dir', type=str)
        self.parser.add_argument('--patch_dir', type=str)
        # self.parser.add_argument('--slide_list', type=list,
        #                          default=['1642001.1-CK.kfb', '1638897.11-CK.kfb', '1638897.16-CK.kfb', '1734281.11.kfb', '1547583.13.kfb', '1638897.12-CK.kfb', '1641996.7-CK.kfb', '1641996.7.kfb', '1641996.6.kfb', '1641996.11.kfb', '1547583.17-CK.kfb', '1547583.14-CK.kfb', '1641996.2.kfb', '1604701.16.kfb', '1641996.5-CK.kfb', '1641996.8.kfb', '1547583.14.kfb', '1638897.16.kfb', '1638897.13.kfb', '1638897.9.kfb', '1638897.15.kfb', '1636600.10-CK.kfb', '1641996.4-CK.kfb', '1641996.2-CK.kfb', '1638897.19-CK.kfb', '1636600.10.kfb', '1547583.20.kfb', '1547583.10-CK.kfb', '1638897.13-CK.kfb', '1641996.5.kfb', '1642001.1.kfb', '1641996.11-CK.kfb', '1547583.12.kfb', '1604701.16-CK.kfb', '1604701.12.kfb', '1641996.8-CK.kfb', '1734281.11-CK.kfb', '1638897.12.kfb', '1641996.10-CK.kfb', '1604701.12-CK.kfb', '1547583.10.kfb', '1641996.10.kfb', '1638897.9-CK.kfb', '1638897.19.kfb', '1547583.17.kfb', '1547583.18-CK.kfb', '1641996.4.kfb', '1641996.6-CK.kfb', '1547583.18.kfb', '1638897.11.kfb', '1547583.13-CK.kfb', '1547583.20-CK.kfb', '1547583.12-CK.kfb', '1638897.15-CK.kfb'])
        self.parser.add_argument('--slide_list', type=list,)
                                 # default=['1734281.11.kfb', '1547583.13.kfb', '1641996.7.kfb', '1641996.6.kfb', '1641996.11.kfb', '1641996.2.kfb', '1604701.16.kfb', '1641996.8.kfb', '1547583.14.kfb', '1638897.16.kfb', '1638897.13.kfb', '1638897.9.kfb', '1638897.15.kfb', '1636600.10.kfb', '1547583.20.kfb', '1641996.5.kfb', '1642001.1.kfb', '1547583.12.kfb', '1604701.12.kfb', '1638897.12.kfb', '1547583.10.kfb', '1641996.10.kfb', '1638897.19.kfb', '1547583.17.kfb', '1641996.4.kfb', '1547583.18.kfb', '1638897.11.kfb'])
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        return self.parser
