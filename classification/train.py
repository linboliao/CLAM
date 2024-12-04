import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn

from dataset_modules.dataset_generic import GenericMilDataset
from models.model import ABMIL, TransMIL, FCLayer, BClassifier, MILNet
from models.model_clam import CLAM_SB
from options.train_options import TrainOptions
from utils.core_utils import train
from utils.file_utils import save_pkl


class Train:
    def __init__(self, opt):
        self.opt = opt
        self.feat_dir = opt.feat_dir if opt.feat_dir else os.path.join(opt.data_root, f'features/{opt.patch_size}/{opt.feat_model}/h5_files')
        self.label_path = opt.label_path if opt.label_path else os.path.join(opt.data_root, f'labels', 'label.csv')
        self.output_dir = opt.output_dir if opt.output_dir else os.path.join(opt.data_root, f'result/{opt.patch_size}/{opt.cls_model}/{opt.feat_model}')

        self.lr = opt.lr
        self.batch_size = opt.batch_size
        self.max_epochs = opt.max_epochs
        self.fold_num = opt.k
        self.seed = opt.seed
        self.model = self.get_model(opt.cls_model, 2, opt.feat_size)

        self.dataset = GenericMilDataset(csv_path=self.label_path, data_dir=os.path.join(self.feat_dir), shuffle=False,
                                         seed=self.seed, print_info=True, patient_strat=False, ignore=[])

        self.set_random_seed()
        os.makedirs(self.output_dir, exist_ok=True)

    def set_random_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    def get_model(self, model_name, num_classes, feat_size):
        if model_name == 'ABMIL':
            return ABMIL(num_classes, feat_size)
        elif model_name == 'TransMIL':
            return TransMIL(num_classes, feat_size)
        elif model_name == 'DSMIL':
            i_classifier = FCLayer(feat_size, num_classes)
            b_classifier = BClassifier(feat_size, num_classes)
            return MILNet(i_classifier, b_classifier)
        elif model_name == 'CLAM':
            return CLAM_SB(num_classes, feat_size)
        else:
            raise NotImplementedError(f'No model named {model_name}')

    def get_loss_fn(self):
        if self.weighted_loss:
            weights = torch.FloatTensor(['2'])  # 定义你的损失权重
            return nn.CrossEntropyLoss(weight=weights)
        else:
            return nn.CrossEntropyLoss()

    def run(self):
        all_test_auc, all_val_auc, all_test_acc, all_val_acc = [], [], [], []

        for i in range(self.fold_num):
            train_dataset, val_dataset, test_dataset = self.dataset.return_splits(from_id=False, csv_path=f'{self.label_path}/splits_{i}.csv')
            results, test_auc, val_auc, test_acc, val_acc = train((train_dataset, val_dataset, test_dataset), i, self.opt)
            all_test_auc.append(test_auc)
            all_val_auc.append(val_auc)
            all_test_acc.append(test_acc)
            all_val_acc.append(val_acc)
            save_pkl(os.path.join(self.output_dir, f'split_{i}_results.pkl'), results)

        final_df = pd.DataFrame({'folds': [_ for _ in range(self.fold_num)], 'test_auc': all_test_auc,
                                 'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc': all_val_acc})
        final_df.to_csv(os.path.join(self.output_dir, 'summary.csv'))


parser = TrainOptions().parse()

if __name__ == '__main__':
    args = parser.parse_args()
    Train(args).run()
