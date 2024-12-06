import importlib
import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn, softmax
from tqdm import tqdm

from dataset_modules.dataset_generic import Generic_MIL_Dataset
from options.train_options import TrainOptions
from utils.core_utils import train
from utils.file_utils import save_pkl


class Train:
    def __init__(self, opt):
        self.opt = opt
        self.feat_dir = opt.feat_dir if opt.feat_dir else os.path.join(opt.data_root, f'features/{opt.patch_size}/{opt.feat_model}/')
        self.label_path = opt.label_path if opt.label_path else os.path.join(opt.data_root, f'labels', 'label.csv')
        self.output_dir = opt.output_dir if opt.output_dir else os.path.join(opt.data_root, f'result/{opt.patch_size}/{opt.cls_model}/{opt.feat_model}')

        self.lr = opt.lr
        self.batch_size = opt.batch_size
        self.max_epochs = opt.max_epochs
        self.fold_num = opt.k
        self.seed = opt.seed
        self.cls_model = opt.cls_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self.get_model(2, opt.feat_size)

        self.dataset = Generic_MIL_Dataset(csv_path=self.label_path,
                                           data_dir=self.feat_dir,
                                           shuffle=False,
                                           print_info=True,
                                           label_dict={0: 0, 1: 1},
                                           patient_strat=False,
                                           ignore=[])

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

    def get_model(self, num_classes, feat_size):
        module = "models.model"
        module = importlib.import_module(module)
        clazz = getattr(module, self.cls_model)
        return clazz(n_classes=num_classes, feat_size=feat_size).to(self.device)

    def get_loss_fn(self):
        module = "models.loss"
        module = importlib.import_module(module)
        return getattr(module, self.cls_model)
        # return clazz(n_classes=num_classes, feat_size=feat_size).to(self.device)
        # if self.weighted_loss:
        #     weights = torch.FloatTensor(['2'])  # 定义你的损失权重
        #     return nn.CrossEntropyLoss(weight=weights)
        # else:
        #     return nn.CrossEntropyLoss()

    def train_one_epoch(self, train_loader, loss_fn, device, optimizer, batch_size):
        running_loss = 0.0
        self.model.train()

        slide_ids, preds, labels, probs = [], [], [], []

        batch_out, batch_pred, batch_label = [], [], []

        for i, batch in enumerate(tqdm(train_loader)):
            pass

        for i, (slide_id, bag, label) in enumerate(train_loader):
            slide_id = slide_id[0]
            bag = bag.squeeze(0)
            label = label.squeeze(0)

            bag = bag.to(device)
            label = label.to(device)
            output = self.model(bag)
            # print(output.shape)
            # input()
            _, pred = torch.max(output, dim=1)  # 返回每一行中最大值的元素及其位置索引

            slide_ids.append(slide_id)
            preds.append(pred.detach().item())
            labels.append(label.detach().item())
            probs.append(softmax(output).detach().cpu().numpy().squeeze(0))

            batch_out.append(output)
            batch_pred.append(pred.detach().item())
            batch_label.append(label)

            # batch批量反传
            if (i + 1) % batch_size == 0 or i == len(train_loader) - 1:
                batch_out = torch.cat(batch_out)
                batch_label = torch.tensor(batch_label, device=device)

                loss = loss_fn(batch_out, batch_label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_label = [lab.detach().item() for lab in batch_label]
                print('loss: {}, label: {}, pred: {} '.format(loss.item(), batch_label, batch_pred))

                running_loss += loss.item() * len(batch_label)
                batch_out, batch_pred, batch_label = [], [], []

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss, slide_ids, labels, preds, probs

    def train(self):
        for epoch in self.max_epochs:
            pass


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
