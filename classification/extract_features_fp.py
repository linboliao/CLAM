import os
import re
import sys
import time

import h5py
import numpy as np
import openslide
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_modules.dataset_h5 import DatasetAllBags, WholeSlideBagFp
from models import get_encoder
from models.resnet_custom_dep import resnet18_baseline, resnet50_baseline
from options.train_options import TrainOptions
from utils.constants import MODEL2CONSTANTS
from utils.file_utils import save_hdf5
from utils.transform_utils import get_eval_transforms
from utils.utils import param_log

sys.path.append('/data2/lbliao/Code/aslide/')
from aslide import Aslide

sys.path.insert(1, r'/data2/lbliao/Code/opensdpc/')
from opensdpc.opensdpc import OpenSdpc


class ExtractFeaturesFP:
    def __init__(self, opt):
        self.slide_dir = opt.slide_dir if opt.slide_dir else os.path.join(opt.data_root, 'slides')
        self.coord_dir = opt.coord_dir if opt.coord_dir else os.path.join(opt.data_root, f'patch/{opt.patch_size}/coord')
        self.feat_dir = opt.feat_dir if opt.feat_dir else os.path.join(opt.data_root, f'features/{opt.patch_size}/{opt.feat_model}_patient/')
        self.count_dir = opt.count_dir if opt.count_dir else os.path.join(opt.data_root, f'patch/{opt.patch_size}/')
        self.count_path = os.path.join(self.count_dir, f'count.csv')

        self.model_name = opt.feat_model
        self.batch_size = opt.batch_size
        self.device = torch.device(f'cuda:{opt.gpus}') if torch.cuda.is_available() else torch.device('cpu')
        self.skip_done = opt.skip_done

        param_log(self)
        os.makedirs(self.feat_dir, exist_ok=True)
        os.makedirs(os.path.join(self.feat_dir, 'pt_files'), exist_ok=True)
        os.makedirs(os.path.join(self.feat_dir, 'h5_files'), exist_ok=True)

    def compute_with_dataloader(self, output_path, loader, model, verbose=0):
        """
        Process a data loader and compute features using the provided model.

        Args:
            output_path (str): Directory to save computed features (.h5 file).
            loader (DataLoader): PyTorch DataLoader.
            model (nn.Module): PyTorch model.
            verbose (int): Level of feedback.
        """
        if verbose > 0:
            logger.info(f'Processing {len(loader)} batches')

        mode = 'w'
        for count, data in enumerate(tqdm(loader)):
            if data is None:
                continue
            with torch.inference_mode():
                batch = data['img'].to(self.device, non_blocking=True)
                features = model(batch)
                features = features.cpu().numpy()

                asset_dict = {'features': features, 'coords': np.array(data['coord']).astype(np.int32)}
                save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
                mode = 'a'

        return output_path

    def load_simclr_pretrained_model(self, model, simclr_save_path):
        # add mlp projection head
        dim_mlp = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)
        # load simclr pretrained model parameters
        simclr_saved = torch.load(simclr_save_path)
        state_dict = {}
        for key, value in simclr_saved['state_dict'].items():
            new_key = key.replace("backbone.", "")
            state_dict[new_key] = value
        model.load_state_dict(state_dict)
        print('load simclr pretrained model successfully.')
        model.fc = nn.Identity()

        return model

    def get_model(self):
        if self.model_name == 'resnet18_256':
            model = resnet18_baseline(pretrained=True)
        elif self.model_name == 'resnet50_1024':
            model = resnet50_baseline(pretrained=True)
        elif self.model_name == 'resnet18_512':
            model = torchvision.models.resnet18(pretrained=True)
            model.fc = nn.Identity()
        elif self.model_name == 'resnet50_2048':
            model = torchvision.models.resnet50(pretrained=True)
            model.fc = nn.Identity()
        elif self.model_name == 'simclr_resnet18_512':
            model = torchvision.models.resnet18(pretrained=False, num_classes=128)
            self.load_simclr_pretrained_model(model, args.simclr_save_path)
        elif self.model_name == 'simclr_resnet50_1024':
            model = resnet50_baseline(pretrained=False)
            model.fc = nn.Linear(1024, 128)
            self.load_simclr_pretrained_model(model, args.simclr_save_path)
        elif self.model_name == 'simclr_resnet50_2048':
            model = torchvision.models.resnet50(pretrained=False, num_classes=128)
            self.load_simclr_pretrained_model(model, args.simclr_save_path)
            model.fc = nn.Identity()
        return model.to(self.device)

    def extract(self):
        bags_dataset = DatasetAllBags(self.count_path)
        constants = MODEL2CONSTANTS['resnet50_trunc']
        # img_transforms = get_eval_transforms(mean=constants['mean'], std=constants['std'], target_img_size=256)
        model, img_transforms = get_encoder(self.model_name)
        model = self.get_model()
        model.eval()
        model = model.to(self.device)
        total = len(bags_dataset)

        loader_kwargs = {'num_workers': 8, 'pin_memory': True} if self.device.type == "cuda" else {}
        dest_files = os.listdir(os.path.join(self.feat_dir, 'pt_files'))
        df = pd.read_csv(os.path.join('/NAS2/Data4/llb/Data/CRC/labels', 'label.csv'))
        count = 0
        for idx in tqdm(range(total)):
            if count >= 50:
                break
            slide_id, slide_ext = os.path.splitext(bags_dataset[idx])
            bag_name1 = slide_id + '.h5'
            name_without_ext = os.path.splitext(slide_id)[0]
            prefix = re.split(r"[-_]", name_without_ext, maxsplit=1)[0]
            bag_name = prefix + '.h5'
            output_path = os.path.join(self.feat_dir, 'h5_files', bag_name)
            # if os.path.exists(output_path):
            #     logger.info(f'Skipping {bag_name} because it already exists.')
            #     continue
            h5_file_path = os.path.join(self.coord_dir, bag_name1)
            slide_file_path = os.path.join(self.slide_dir, slide_id + slide_ext)
            logger.info(f'progress: {idx}/{total},{slide_id}')

            if self.skip_done and slide_id + '.pt' in dest_files:
                logger.info(f'skipped {slide_id}')
                continue
            v = df.loc[df['slide_id'] == bags_dataset[idx], 'label'].values.tolist()
            # processed = ['431607', 'ZZ54', '443348', '407385', '409804', '1186631', 'ZZ19', '1180981', '358773', '335130', '456716', '374189', '584016', '432938', '1218143', '438863', '606582', '418670', '364586', 'ZZ12', '1195978', '440297', '408935', '483114', '444619', '466755', '407388', '461119', '441639', '363848', '375600', 'ZZ35', 'ZZ21', 'ZZ28', '610362', '421650', 'ZZ25', '358374', '362853', '417824', '356464', '384848', '389005', '1222919', '479676', 'ZZ38', 'ZZ45', '1154348', '538048', '412196', '381657', '1186304', '428511', 'ZZ39', '359800', 'ZZ50', '480324', '374911', 'ZZ55', '375394', '1250157 B5', '1171040', '341177', '554723', '370993', '1250157 B6', '780564', '1232205', '563028', '407163', '372785', '1177401', '1152597', '548546', 'ZZ48', '392334', '403316', '352908', '561895', '750937', '369361', '331161', '745400', 'ZZ14', '1209004', '1259902 A5', '436483', 'ZZ27', '436485', '1247753 B5', 'ZZ34', '420666', '402641', '1232590', '377758', 'ZZ57', '367104', '1246808 A11 举例癌区分辨', '408545', '1247753 B6粘液', '372308', '368366', '1246808 A9', 'ZZ4', '436880', '364992', '1148306', '1190123', 'ZZ6', '439261', '412463', '605724', '552091', '1214513', '1209576']
            # name_without_ext = os.path.splitext(slide_id)[0]
            # prefix = re.split(r"[-_]", name_without_ext, maxsplit=1)[0]
            # if prefix in processed:
            #     logger.info(f'skipped {slide_id}, slide in processed file')
            #     continue
            if len(v) >0 and v[0] != 1:
                logger.info(f'skipped {slide_id}, no label 1')
                continue

            time_start = time.time()
            if slide_ext == '.kfb':
                wsi = Aslide(slide_file_path)
            elif slide_ext == '.sdpc':
                wsi = OpenSdpc(slide_file_path)
            else:
                wsi = openslide.OpenSlide(slide_file_path)
            dataset = WholeSlideBagFp(file_path=h5_file_path, wsi=wsi, img_transforms=img_transforms)

            def collate_fn(batch):
                original_batch_size = len(batch)  # 记录原始批次大小
                batch = [item for item in batch if item is not None]  # 过滤无效数据
                if len(batch) < self.batch_size:
                    return None
                # # 重新采样逻辑
                # while len(batch) < original_batch_size:
                #     # 随机生成新索引（需根据实际数据集长度调整）
                #     new_idx = np.random.randint(int(len(dataset) * 0.4), int(len(dataset) * 0.6))  # 假设dataset是全局可访问的
                #     new_item = dataset[new_idx]  # 重新采样
                #
                #     if new_item is not None:  # 仅添加有效样本
                #         batch.append(new_item)

                # 合并有效数据
                imgs = torch.stack([item['img'] for item in batch])
                coords = [item['coord'] for item in batch]
                return {'img': imgs, 'coord': coords}

            loader = DataLoader(dataset=dataset, batch_size=self.batch_size, collate_fn=collate_fn, **loader_kwargs)
            output_file_path = self.compute_with_dataloader(output_path, loader=loader, model=model, verbose=1)

            time_elapsed = time.time() - time_start
            logger.info('computing features for {} took {} s'.format(output_file_path, time_elapsed))

            with h5py.File(output_file_path, "r") as file:
                features = file['features'][:]
                logger.info(f'features size: {features.shape}, coordinates size: {file["coords"].shape}')

            features = torch.from_numpy(features)
            bag_base, _ = os.path.splitext(bag_name)
            torch.save(features, os.path.join(self.feat_dir, 'pt_files', bag_base + '.pt'))
            count += 1


parser = TrainOptions().parse()

if __name__ == '__main__':
    args = parser.parse_args()
    ExtractFeaturesFP(args).extract()
