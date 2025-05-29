import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random
import numpy as np
import torchvision.models as models
from tqdm import tqdm
import argparse
from datetime import datetime
import json
from torch.utils.tensorboard import SummaryWriter
import logging
import torchxrayvision as xrv
import sys
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_random_seed(seed):
    random.seed(seed)  # python random module
    np.random.seed(seed)  # numpy module
    torch.manual_seed(seed)  # 为cpu设置
    torch.cuda.manual_seed(seed)  # 为当前gpu设置
    torch.cuda.manual_seed_all(seed)  # 为所有gpu设置
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    torch.backends.cudnn.benchmark = False  # 设置cuDNN：cudnn中对卷积操作进行了优化，牺牲了精度来换取计算效率。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


## 对dataloader设置随机种子：如果dataloader采用了多线程(num_workers > 1), 那么由于读取数据的顺序不同，最终运行结果也会有差异。
## 在PyTorch的DataLoader函数中为不同的work设置初始化函数，确保您的dataloader在每次调用时都以相同的顺序加载样本（随机种子固定时）
# def worker_init_fn(worker_id):
#     np.random.seed(1024 + worker_id)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transformer = transforms.Compose([
    # transforms.Resize(512),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
val_transformer = transforms.Compose([
    # transforms.Resize(512),
    transforms.ToTensor(),
    normalize
])


class PatchDataset(Dataset):
    def __init__(self, img_dir, label_csv, transform=None):
        """
        Args:
            label_csv (csv file) include "patch_id", "label"
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv = pd.read_csv(label_csv)
        self.img_paths = [os.path.join(img_dir, f"{patch_id}") for patch_id in self.csv['patch_id'].tolist()]
        self.labels = self.csv['label'].tolist()
        self.transform = transform
        print(f"Number of samples in dataset: {len(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if not os.path.exists(self.img_paths[idx]):
            raise FileNotFoundError(f"Image file not found: {self.img_paths[idx]}")

        image = Image.open(self.img_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = int(self.labels[idx])
        sample = {'img': image,
                  'img_path': self.img_paths[idx],
                  'label': label}
        return sample


def creat_dirs_for_result(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%m%d%H%M%S")
    train_name = str(timestamp) + '_' + str(args.model_name)
    log_dir = os.path.join(args.output_dir, train_name, 'logs')
    model_dir = os.path.join(args.output_dir, train_name, 'models')
    label_dir = os.path.join(args.output_dir, train_name, 'label_split')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir, exist_ok=True)
    return log_dir, model_dir, label_dir


def save_args_json(data, log_dir):
    file = os.path.join(log_dir, 'args.json')
    if not os.path.exists(file):
        with open(file, 'w') as f:
            json.dump(data, f, indent=4)


def get_logger(log_dir, name=None):
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(message)s')

    # logging.basicConfig(filename=os.path.join(log_dir, 'train.log'),
    #                     format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
    #                     level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    file_handler = logging.FileHandler(filename=os.path.join(log_dir, 'train.log'), mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handle = logging.StreamHandler(sys.stderr)
    console_handle.setLevel(logging.INFO)
    console_handle.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handle)

    return logger


class DenseNetModel(nn.Module):

    def __init__(self):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        super(DenseNetModel, self).__init__()

        self.dense_net = xrv.models.DenseNet(num_classes=2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        logits = self.dense_net(x)
        return logits


def generate_resnet(model_name, **kwargs):
    # assert model_depth in [10, 18, 34, 50, 101, 152, 200]
    if model_name == 'ResNet10':
        model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    elif model_name == 'ResNet18':
        model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    elif model_name == 'ResNet34':
        model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    elif model_name == 'ResNet50':
        model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    elif model_name == 'ResNet101':
        model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    elif model_name == 'ResNet152':
        model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    elif model_name == 'ResNet200':
        model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    else:
        raise NotImplementedError('no model:{}'.format(model_name))
    return model


def train_one_epoch(model, train_loader, loss_fn, optimizer, epoch, device):
    running_loss = 0.0
    running_correct = 0.0
    model.train()

    epoch_img_paths, epoch_preds, epoch_labels, epoch_probs = [], [], [], []
    for batch_samples in train_loader:
        # print(f"Batch samples: {batch_samples}")
        images = batch_samples['img'].to(device)
        img_paths = batch_samples['img_path']
        labels = batch_samples['label'].to(device)

        outputs = model(images)

        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)  # 返回每一行中最大值的元素及其位置索引

        # print('loss: {}, label: {}, pred: {} '
        #         .format(loss.item(), labels.cpu(), preds.cpu().detach()))

        running_loss += loss.item() * images.size(0)  # 如果要*images.size(0)后面需要除以len(dataset)
        # 如果不*就直接除以len(dataloader)
        running_correct += (torch.sum(preds == labels.data)).item()

        epoch_img_paths += img_paths
        epoch_preds += preds.tolist()
        epoch_labels += labels.tolist()
        epoch_probs += torch.softmax(outputs, dim=1).tolist()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_correct / len(train_loader.dataset)

    print('[epoch %d] train_Loss: %.3f train_Acc: %.3f' % (epoch + 1, epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc, epoch_img_paths, epoch_preds, epoch_labels, epoch_probs


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    with torch.no_grad():
        losses = 0.0
        correct = 0.0
        epoch_img_paths, epoch_preds, epoch_labels, epoch_probs = [], [], [], []
        for batch_samples in dataloader:
            images = batch_samples['img'].to(device)
            img_paths = batch_samples['img_path']
            labels = batch_samples['label'].to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            losses += loss.item() * images.size(0)  # 如果要*images.size(0)后面需要除以len(dataset)
            _, preds = torch.max(outputs, 1)
            # print('loss: {}, label: {}, pred: {}'.format(loss.item(), labels, preds))
            correct += (torch.sum(preds == labels.data)).item()

            # data_pred += [outputs.cpu()]
            # data_true += [labels.cpu()]
            epoch_img_paths += img_paths
            epoch_preds += preds.tolist()
            epoch_labels += labels.tolist()
            epoch_probs += torch.softmax(outputs, dim=1).tolist()

        loss = losses / len(dataloader.dataset)
        acc = correct / len(dataloader.dataset)

    return loss, acc, epoch_img_paths, epoch_preds, epoch_labels, epoch_probs


def train_and_eval(args, train_csv, val_csv, test_csv, log_dir, model_dir, fold_index):
    # 1定义dataset
    trainset = PatchDataset(img_dir=args.img_dir, label_csv=train_csv, transform=train_transformer)
    valset = PatchDataset(img_dir=args.img_dir, label_csv=val_csv, transform=val_transformer)
    testset = PatchDataset(img_dir=args.img_dir, label_csv=test_csv, transform=val_transformer)
    # 2定义dataload
    train_loader = DataLoader(trainset, batch_size=args.batch_size, drop_last=True, shuffle=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size, drop_last=True, shuffle=False)
    test_loader = DataLoader(testset, batch_size=args.batch_size, drop_last=True, shuffle=False)

    # 3定义model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model_name == 'DenseNet_medical':
        model = DenseNetModel().to(device)
    elif 'ResNet' in args.model_name:
        model = generate_resnet(args.model_name, num_classes=args.num_class).to(device)
    else:
        raise NotImplementedError('no model:{}'.format(args.model_name))

    model = torch.nn.DataParallel(model)  # 多GPU训练torch.nn.DataParallel(model, device_ids=[0,1])

    # 4定义损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 5定义优化算法
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    writer = SummaryWriter(log_dir)
    logger = get_logger(log_dir)

    # 6模型训练
    best_acc = 0.0
    kfold_test_acc = 0.0
    kfold_test_probs = []
    kfold_test_preds = []

    for epoch in tqdm(range(args.epochs)):
        train_loss, train_acc, train_img_paths, train_preds, train_labels, train_probs = train_one_epoch(model, train_loader, loss_fn, optimizer, epoch, device)
        scheduler.step()
        #### val the model per epoch
        val_loss, val_acc, val_img_paths, val_preds, val_labels, val_probs = evaluate(model, val_loader, loss_fn, device)
        print('[epoch %d] val_loss: %.3f val_acc: %.3f' % (epoch + 1, val_loss, val_acc))
        #### test the model per epoch
        test_loss, test_acc, test_img_paths, test_preds, test_labels, test_probs = evaluate(model, test_loader, loss_fn, device)
        print('[epoch %d] test_loss: %.3f test_acc: %.3f' % (epoch + 1, test_loss, test_acc))

        # 保存每个折的模型
        torch.save(model.state_dict(), os.path.join(model_dir, f'model_fold{fold_index}_epoch{epoch + 1}.pth'))

        # 保存多分类best_acc
        if val_acc > best_acc:
            best_acc = val_acc
            kfold_test_acc = test_acc
            kfold_test_preds = test_preds
            kfold_test_probs = test_probs

            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model_e{:d}_{:.4f}.pth'.format(epoch + 1, best_acc)))
            torch.save(model.state_dict(), os.path.join(model_dir, 'model_best.pth'))

            csv = {'slide_id': train_img_paths + val_img_paths + test_img_paths, 'label': train_labels + val_labels + test_labels,
                   'pred': train_preds + val_preds + test_preds, 'prob': train_probs + val_probs + test_probs}
            csv = pd.DataFrame(csv)
            csv.to_csv(os.path.join(log_dir, 'best_epoch_output.csv'), index=False)

            with open(os.path.join(log_dir, 'best_epoch.txt'), mode='w') as file:
                file.write('{:d}\n'.format(epoch + 1))
                file.write('{:.4f}\n'.format(train_acc))
                file.write('{:.4f}\n'.format(val_acc))
                file.write('{:.4f}\n'.format(test_acc))

                # 提取正类的概率
                test_probs = np.array(test_probs)
                # test_probs = test_probs[:, 1]  # 提取第二列（正类概率）

                file.write('macro_auc_ovr {:.4f}\n'.format(roc_auc_score(test_labels, test_probs, average="macro", multi_class="ovr")))
                file.write('macro_auc_ovo {:.4f}\n'.format(roc_auc_score(test_labels, test_probs, average="macro", multi_class="ovo")))
                file.write('micro_auc_ovr {:.4f}\n'.format(roc_auc_score(test_labels, test_probs, average="micro", multi_class="ovr")))
                # file.write('micro_auc_ovo {:.4f}\n'.format(roc_auc_score(labels, probs, average="micro", multi_class="ovo")))#这种算不了
                file.write(str(classification_report(test_labels, test_preds)))
                file.write('train\n' + str(confusion_matrix(train_labels, train_preds)) + '\n')
                file.write('val\n' + str(confusion_matrix(val_labels, val_preds)) + '\n')
                file.write('test\n' + str(confusion_matrix(test_labels, test_preds)) + '\n')
                # #每隔50保存一次
        # if (epoch+1) % 50 == 0:
        #     torch.save(model.state_dict(),
        #         os.path.join(model_dir, 'model_e{:d}_{:.4f}.pth'.format(epoch+1, val_acc)))

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch + 1)
        writer.add_scalar('train_Loss', train_loss, epoch + 1)
        writer.add_scalar('val_loss', val_loss, epoch + 1)
        writer.add_scalar('test_loss', val_loss, epoch + 1)
        writer.add_scalar('train_Acc', train_acc, epoch + 1)
        writer.add_scalar('val_acc', val_acc, epoch + 1)
        writer.add_scalar('test_acc', test_acc, epoch + 1)
        logger.info('[epoch %d] train_Loss: %.3f train_Acc: %.3f val_loss: %.3f val_acc: %.3f test_loss: %.3f  test_acc: %.3f' %
                    (epoch + 1, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))
    # save the latest_model
    torch.save(model.state_dict(), os.path.join(model_dir, 'model_latest.pth'))

    return kfold_test_acc, test_img_paths, test_labels, kfold_test_preds, kfold_test_probs


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1024, help='Random seed to use.')
    parser.add_argument('--gpus', type=str, default='0,1', help="GPU indices ""comma separated, e.g. '0,1' ")
    # parser.add_argument('--dataset', type=str, default='/NAS2/Data4/llb/Data/cls/高速下载- NCT-CRC-HE_含数据集介绍/NCT-CRC-HE_含数据集介绍/CRC-VAL-HE-7K/CRC-VAL-HE-7K', help='')
    parser.add_argument('--dataset', type=str, default='/NAS2/Data4/llb/Data/cls/高速下载- NCT-CRC-HE_含数据集介绍/NCT-CRC-HE_含数据集介绍/NCT-CRC-HE-100K/NCT-CRC-HE-100K', help='')
    # parser.add_argument('--img_dir', type=str, default='/NAS2/Data4/llb/Data/cls/高速下载- NCT-CRC-HE_含数据集介绍/NCT-CRC-HE_含数据集介绍/CRC-VAL-HE-7K/CRC-VAL-HE-7K/images', help='')
    parser.add_argument('--img_dir', type=str, default='/NAS2/Data4/llb/Data/cls/高速下载- NCT-CRC-HE_含数据集介绍/NCT-CRC-HE_含数据集介绍/NCT-CRC-HE-100K/NCT-CRC-HE-100K/images', help='')
    parser.add_argument('--num_class', type=int, default=9, help='')
    parser.add_argument('--fold_num', type=int, default=5)
    parser.add_argument('--output_dir', default='./train_results-CRC', help='Path to experiment output, config, checkpoints, etc.')
    # 'ResNet' 'DenseNet_medical'
    parser.add_argument('--model_name', type=str, default='ResNet50', help='What model architecture to use.')
    parser.add_argument('--batch_size', type=int, default=192, help='Dataloaders batch size.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = args_parser()  # 声明中给定了各个参数的默认值
    set_random_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    dataset_basename = os.path.basename(args.dataset)
    args.output_dir = os.path.join(args.output_dir, dataset_basename, args.model_name, str(args.fold_num) + 'fold')

    # 计算结果保存
    test_acc_list, img_paths, labels, preds, probs = [], [], [], [], []
    # k折数据划分与计算
    skf = StratifiedKFold(n_splits=args.fold_num, random_state=args.seed, shuffle=True)
    df = pd.read_csv(os.path.join(args.dataset, 'labels.csv'))
    # df = df[df['label'] != 4]
    for i, (train_index, test_index) in enumerate(skf.split(df['patch_id'], df['label'])):
        # 保存当前实验参数和结果的操作
        args.logger_name = 'logger' + str(i + 1)
        log_dir, model_dir, label_dir = creat_dirs_for_result(args)
        save_args_json(args.__dict__, log_dir)

        train_csv = os.path.join(label_dir, 'train.csv')
        val_csv = os.path.join(label_dir, 'val.csv')
        test_csv = os.path.join(label_dir, 'test.csv')

        train_df = df.iloc[train_index]
        # 训练集中分出1折做验证集，用于挑选模型
        train_df, val_df = train_test_split(train_df, test_size=1 / (args.fold_num - 1), stratify=train_df['label'])
        # 将训练集、验证集、测试集暂时保存成csv文件
        train_df.to_csv(train_csv, index=0)
        val_df.to_csv(val_csv, index=0)
        df.iloc[test_index].to_csv(test_csv, index=0)

        kfold_test_acc, test_img_paths, test_labels, kfold_test_preds, kfold_test_probs = \
            train_and_eval(args, train_csv, val_csv, test_csv, log_dir, model_dir, fold_index=i)

        test_acc_list.append(kfold_test_acc)
        img_paths += test_img_paths
        labels += test_labels
        preds += kfold_test_preds
        probs += kfold_test_probs

    preds = np.array(preds)
    probs = np.array(probs)[:, 1]
    # 多折交叉验证计算结束，结果处理
    with open(os.path.join(args.output_dir, 'kfold_result.txt'), mode='w') as file:
        formatted_acc_list = ["{:.4f}".format(num) for num in test_acc_list]
        file.write(str(formatted_acc_list) + '\n')
        file.write('average {:.4f}\n'.format(sum(test_acc_list) / len(test_acc_list)))
        file.write('macro_auc_ovr {:.4f}\n'.format(roc_auc_score(labels, probs, average="macro", multi_class="ovr")))
        file.write('macro_auc_ovo {:.4f}\n'.format(roc_auc_score(labels, probs, average="macro", multi_class="ovo")))
        file.write('micro_auc_ovr {:.4f}\n'.format(roc_auc_score(labels, probs, average="micro", multi_class="ovr")))
        # file.write('micro_auc_ovo {:.4f}\n'.format(roc_auc_score(labels, probs, average="micro", multi_class="ovo")))#这种算不了
        file.write(str(classification_report(labels, preds, digits=4)))
        file.write(str(confusion_matrix(labels, preds)) + '\n')

    csv = {'img_path': img_paths, 'label': labels, 'pred': preds, 'probs': probs}
    csv = pd.DataFrame(csv)
    csv.to_csv(os.path.join(args.output_dir, 'prob_output.csv'), index=False)
    np.save(os.path.join(args.output_dir, 'probs.npy'), np.array(probs))

