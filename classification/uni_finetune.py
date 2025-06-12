import timm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import has_UNI

HAS_UNI, UNI_CKPT_PATH = has_UNI()
assert HAS_UNI, 'UNI is not available'
model = timm.create_model("vit_large_patch16_224", init_values=1e-5, num_classes=0, dynamic_img_size=True)
model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)

# 2. 修改模型头部 (适配您的分类任务)
num_classes = 10  # 根据您的实际类别数修改
model.fc = nn.Sequential(
    nn.Dropout(0.5),  # 添加Dropout防止过拟合[6,9](@ref)
    nn.Linear(model.fc.in_features, num_classes)
)

# 3. 数据预处理 (医学影像专用)
# 使用与UNI预训练相同的标准化参数
med_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪[6](@ref)
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),  # 小角度旋转增强[9](@ref)
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 颜色扰动
    transforms.ToTensor(),
    med_norm
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    med_norm
])


# 4. 数据集类 (优化内存管理)
class MedicalPatchDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.class_weights = self._calculate_class_weights()

    def _calculate_class_weights(self):
        """计算类别权重解决不平衡问题"""
        class_counts = self.df['label'].value_counts().sort_index()
        total = class_counts.sum()
        return torch.tensor([total / count for count in class_counts], dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df.iloc[idx]['patch_id'])
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label, self.class_weights[label]


# 5. 创建数据加载器
def create_data_loaders(data_dir, csv_path, batch_size=32, val_split=0.1):
    """创建训练/验证数据加载器"""
    full_dataset = MedicalPatchDataset(data_dir, csv_path)

    # 分层分割保持类别分布
    from sklearn.model_selection import train_test_split
    indices = list(range(len(full_dataset)))
    labels = [full_dataset.df.iloc[i]['label'] for i in indices]

    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, stratify=labels, random_state=42
    )

    train_set = torch.utils.data.Subset(full_dataset, train_idx)
    val_set = torch.utils.data.Subset(full_dataset, val_idx)

    # 应用不同变换
    train_set.dataset.transform = train_transform
    val_set.dataset.transform = val_transform

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    return train_loader, val_loader


# 6. 微调训练函数 (带早停机制)
def fine_tune_model(model, train_loader, val_loader, num_epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 使用带类别权重的损失函数
    criterion = nn.CrossEntropyLoss(weight=train_loader.dataset.dataset.class_weights.to(device))

    # 分层学习率设置 (特征提取层更低)
    optimizer = optim.AdamW([
        {'params': model.parameters()[:-2], 'lr': 1e-4},  # 特征提取层
        {'params': model.parameters()[-2:], 'lr': 1e-3}  # 新分类头
    ], weight_decay=1e-4)  # 权重衰减[6,9](@ref)

    # 动态学习率调整
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    best_acc = 0.0
    best_model_wts = None
    early_stop_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print('-' * 10)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels, _ in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        # 验证阶段
        val_loss, val_acc = evaluate_model(model, val_loader, device, criterion)

        # 学习率调整
        scheduler.step(val_acc)

        # 早停机制
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= 5:
                print(f"⚠️ 早停触发: 验证精度连续5轮未提升")
                break

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model


# 7. 模型评估函数
def evaluate_model(model, data_loader, device, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels, _ in tqdm(data_loader, desc="Validating"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_corrects.double() / len(data_loader.dataset)

    return epoch_loss, epoch_acc


# 8. 主执行流程
if __name__ == "__main__":
    # 配置参数
    DATA_DIR = "/path/to/your/patches"  # 替换为您的数据路径
    CSV_PATH = "/path/to/labels.csv"  # 替换为您的标签文件
    BATCH_SIZE = 64  # 根据GPU内存调整
    NUM_EPOCHS = 30

    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(
        DATA_DIR, CSV_PATH, batch_size=BATCH_SIZE
    )

    # 微调模型
    print("🚀 开始微调UNI模型...")
    model = fine_tune_model(model, train_loader, val_loader, NUM_EPOCHS)

    # 保存微调后的特征提取器
    torch.save({
        'model_state_dict': model.state_dict(),
        'classifier': model.fc.state_dict()  # 单独保存分类头
    }, "fine_tuned_uni_feature_extractor.pth")

    print("✅ 特征提取器已保存为: fine_tuned_uni_feature_extractor.pth")