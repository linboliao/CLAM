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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from models import has_UNI

HAS_UNI, UNI_CKPT_PATH = has_UNI()
assert HAS_UNI, 'UNI is not available'
model = timm.create_model("vit_large_patch16_224", init_values=1e-5, num_classes=0, dynamic_img_size=True)
model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)

# 2. 修改模型头部
num_classes = 9  # NCT-CRC-HE-100K有9个类别
# 正确方式：替换分类头
if hasattr(model, 'head'):
    if isinstance(model.head, nn.Identity):
        dummy_input = torch.randn(1, 3, 224, 224)
        features = model.forward_features(dummy_input)
        input_dim = features.shape[-1]
        model.head = nn.Linear(input_dim, num_classes)
    else:
        model.head = nn.Linear(model.head.in_features, num_classes)
elif hasattr(model, 'classifier'):
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
else:
    # 备用方案：直接添加新的分类头
    model.head = nn.Linear(model.embed_dim, num_classes)
    model.classifier = model.head  # 添加别名方便访问

# 3. 医学影像专用预处理
med_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # 更激进的随机裁剪
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    med_norm
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    med_norm
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    med_norm
])


# 4. 数据集类 (内存优化版)
class MedicalPatchDataset(Dataset):
    def __init__(self, root_dir, df, label_encoder=None, transform=None):
        """
        root_dir: 图像根目录
        df: 包含'patch_id'和'label'的DataFrame
        label_encoder: 预训练的标签编码器 (用于测试集)
        transform: 数据增强变换
        """
        self.root_dir = root_dir
        self.df = df
        self.transform = transform

        # 处理字符串标签
        if label_encoder:
            self.le = label_encoder
        else:
            self.le = LabelEncoder()
            self.df['encoded_label'] = self.le.fit_transform(self.df['label'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df.iloc[idx]['patch_id'])
        image = Image.open(img_path).convert('RGB')

        # 处理字符串标签
        if 'encoded_label' in self.df.columns:
            label = self.df.iloc[idx]['encoded_label']
        else:
            label = self.le.transform([self.df.iloc[idx]['label']])[0]

        if self.transform:
            image = self.transform(image)

        return image, label


# 5. 改进的数据加载器创建
def create_data_loaders(data_dir, csv_path, batch_size=64, val_split=0.1, test_split=0.1):
    """创建训练/验证/测试数据加载器"""
    df = pd.read_csv(csv_path)

    # 确保路径一致
    df['patch_id'] = df['patch_id'].apply(lambda x: x.split('/')[-1] if '/' in x else x)

    # 分层分割数据集
    train_df, test_df = train_test_split(
        df, test_size=test_split, stratify=df['label'], random_state=42
    )
    train_df, val_df = train_test_split(
        train_df, test_size=val_split, stratify=train_df['label'], random_state=42
    )

    # 初始化标签编码器
    le = LabelEncoder()
    le.fit(df['label'])

    # 创建数据集
    train_dataset = MedicalPatchDataset(data_dir, train_df, le, train_transform)
    val_dataset = MedicalPatchDataset(data_dir, val_df, le, val_transform)
    test_dataset = MedicalPatchDataset(data_dir, test_df, le, test_transform)

    # 计算类别权重（基于训练集）
    class_counts = np.bincount(train_dataset.df['label'])
    total = class_counts.sum()
    class_weights = torch.tensor([total / count for count in class_counts], dtype=torch.float32)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader, class_weights, le


# 6. 增强的微调训练函数
def fine_tune_model(model, train_loader, val_loader, class_weights, num_epochs=40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")
    model = model.to(device)

    # 使用带类别权重的损失函数
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # 分层学习率设置
    if hasattr(model, 'head') and hasattr(model.head, 'weight'):
        head_params = [param for name, param in model.named_parameters()
                       if 'head' in name or 'fc' in name or 'classifier' in name]
        backbone_params = [param for name, param in model.named_parameters()
                           if 'head' not in name and 'fc' not in name and 'classifier' not in name]
    else:
        head_params = model.head.parameters()
        backbone_params = [param for param in model.parameters() if param not in head_params]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-4},
        {'params': head_params, 'lr': 1e-3}
    ], weight_decay=1e-4)

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

        for inputs, labels in tqdm(train_loader, desc="训练中"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

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
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_wts,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, "best_model_checkpoint.pth")
            print("💾 保存最佳模型权重")
        else:
            early_stop_counter += 1
            if early_stop_counter >= 5:
                print(f"⚠️ 早停触发: 验证精度连续5轮未提升")
                break

        print(f'训练损失: {epoch_loss:.4f} | 训练精度: {epoch_acc:.4f}')
        print(f'验证损失: {val_loss:.4f} | 验证精度: {val_acc:.4f}')
        print(f"学习率: {optimizer.param_groups[0]['lr']:.2e}")

    # 加载最佳模型权重
    if best_model_wts:
        model.load_state_dict(best_model_wts)
    return model


# 7. 改进的模型评估函数
def evaluate_model(model, data_loader, device, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(data_loader, desc="评估中"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_corrects.double() / len(data_loader.dataset)

    return epoch_loss, epoch_acc


# 8. 添加模型测试函数
def test_model(model, test_loader, device, label_encoder):
    model.eval()
    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(test_loader, desc="测试中"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 将编码标签转换回原始标签名称
    pred_labels = label_encoder.inverse_transform(all_preds)
    true_labels = label_encoder.inverse_transform(all_labels)

    # 计算详细指标
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels)
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    print(f"\n测试精度: {accuracy:.4f}")
    print("分类报告:")
    print(report)
    print("混淆矩阵:")
    print(conf_matrix)

    # 保存结果
    results_df = pd.DataFrame({
        'True_Label': true_labels,
        'Predicted_Label': pred_labels,
        'Correct': [t == p for t, p in zip(true_labels, pred_labels)]
    })
    results_df.to_csv("test_results.csv", index=False)

    return accuracy


# 9. 主执行流程
if __name__ == "__main__":
    # 配置参数
    DATA_DIR = "/NAS2/Data4/llb/Data/cls/高速下载- NCT-CRC-HE_含数据集介绍/NCT-CRC-HE_含数据集介绍/NCT-CRC-HE-100K/NCT-CRC-HE-100K/images"
    CSV_PATH = "/NAS2/Data4/llb/Data/cls/高速下载- NCT-CRC-HE_含数据集介绍/NCT-CRC-HE_含数据集介绍/NCT-CRC-HE-100K/NCT-CRC-HE-100K/labels.csv"
    BATCH_SIZE = 64  # 根据GPU内存调整
    NUM_EPOCHS = 40

    # 创建数据加载器
    print("📂 正在加载数据...")
    train_loader, val_loader, test_loader, class_weights, label_encoder = create_data_loaders(
        DATA_DIR, CSV_PATH, batch_size=BATCH_SIZE, val_split=0.1, test_split=0.1
    )

    # 显示数据统计
    print(f"\n📊 数据统计:")
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"验证样本数: {len(val_loader.dataset)}")
    print(f"测试样本数: {len(test_loader.dataset)}")
    print(f"类别权重: {class_weights.tolist()}")
    print(f"类别编码: {label_encoder.classes_}\n")

    # 微调模型
    print("🚀 开始微调UNI模型...")
    model = fine_tune_model(model, train_loader, val_loader, class_weights, NUM_EPOCHS)

    # 评估测试集
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_acc = test_model(model, test_loader, device, label_encoder)

    # 保存整个模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder': label_encoder,
        'test_acc': test_acc
    }, "fine_tuned_uni_model.pth")

    print(f"\n✅ 训练完成! 最终测试精度: {test_acc:.4f}")
    print("🚀 模型已保存为: fine_tuned_uni_model.pth")
