import argparse
import os
import time

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_modules.dataset_h5 import Dataset_All_Bags_Patient, Whole_Slide_Bag_FP_NoCoords
from models import get_encoder
from utils.file_utils import save_hdf5
from wsi import WSIOperator

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def compute_w_loader(output_path, loader, model, slide_id, verbose=0):
    """
    args:
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        verbose: level of feedback
    """
    if verbose > 0:
        print(f'processing a total of {len(loader)} batches'.format(len(loader)))

    mode = 'w'
    for count, data in enumerate(tqdm(loader)):
        with torch.inference_mode():
            batch = data['img']
            # TODO 去除掉为空的 img 和 coords
            coords = data['coord'].numpy().astype(np.int32)
            batch = batch.to(device, non_blocking=True)

            features = model(batch)
            features = features.cpu().numpy().astype(np.float32)

            asset_dict = {'features': features, 'coords': coords, 'slide_id': slide_id}
            save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
            mode = 'a'

    return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default='/NAS2/Data4/llb/Data/CRC/patch/256/coord')
parser.add_argument('--data_slide_dir', type=str, default='/NAS2/Data4/llb/Data/CRC/slides')
parser.add_argument('--csv_path', type=str, default='/NAS2/Data4/llb/Data/CRC/labels/label.csv')
parser.add_argument('--feat_dir', type=str, default='/NAS2/Data1/lbliao/Data/CRC/features_patient')
parser.add_argument('--model_name', type=str, default='uni_v1', choices=['resnet50_trunc', 'uni_v1', 'conch_v1'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
args = parser.parse_args()

if __name__ == '__main__':
    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    # TODO 指定医院
    df = df[df['data_source'] == '中日友好医院结直肠癌数据']
    patient_df = df.drop_duplicates(subset=['patient_id'])
    patient_df.to_csv('tmp_patient.csv', index=False)

    bags_dataset = Dataset_All_Bags_Patient('tmp_patient.csv')

    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

    model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)

    _ = model.eval()
    model = model.to(device)
    total = len(bags_dataset)

    loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

    # 按patient_id分组，将slide_id转换为列表
    patient_dict = df.groupby('patient_id')['slide_id'].apply(list).to_dict()

    for bag_candidate_idx in tqdm(range(total)):
        patient_id = str(bags_dataset[bag_candidate_idx])
        bag_name = patient_id + '.h5'
        if not patient_dict[patient_id]:
            print(f'{patient_id} 下没有 slide')
            continue
        if not args.no_auto_skip and patient_id + '.pt' in dest_files:
            print('skipped {}'.format(patient_id))
            continue
        for slide_name in patient_dict[patient_id]:
            slide_id, ext = os.path.splitext(slide_name)
            sub_bag_name = slide_id + '.h5'
            h5_file_path = os.path.join(args.data_h5_dir, sub_bag_name)
            slide_file_path = os.path.join(args.data_slide_dir, slide_name)
            print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
            print(slide_id)

            output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
            time_start = time.time()
            wsi = WSIOperator(slide_file_path)
            dataset = Whole_Slide_Bag_FP_NoCoords(file_path=h5_file_path,
                                                  wsi=wsi,
                                                  img_transforms=img_transforms)

            loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
            output_file_path = compute_w_loader(output_path, loader=loader, model=model, slide_id=slide_id, verbose=1)

            time_elapsed = time.time() - time_start
            print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
            with h5py.File(output_file_path, "r") as file:
                features = file['features'][:]
                print('features size: ', features.shape)
                print('coordinates size: ', file['coords'].shape)

            features = torch.from_numpy(features)
            bag_base, _ = os.path.splitext(bag_name)
            torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base + '.pt'))
