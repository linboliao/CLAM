# import os
# import shutil
# import pandas as pd
#
# patch_dir = r'/NAS2/Data4/llb/Data/cls/高速下载- NCT-CRC-HE_含数据集介绍/NCT-CRC-HE_含数据集介绍/NCT-CRC-HE-100K/NCT-CRC-HE-100K'  # 需替换为实际路径
# label_dict = {'TUM': 0, 'STR': 1, 'NORM': 2, 'MUS': 3, 'MUC': 4, 'LYM': 5, 'DEB': 6, 'BACK': 7, 'ADI': 8}
#
# # 初始化DataFrame存储结构[3,4](@ref)
# data = {'patch_id': [], 'label': []}
# df = pd.DataFrame(data)
#
# # 创建目标图像目录
# new_img_dir = os.path.join(patch_dir, 'images')
# os.makedirs(new_img_dir, exist_ok=True)
#
# # 遍历子目录处理文件[2](@ref)
# for sub_folder in os.listdir(patch_dir):
#     src_dir = os.path.join(patch_dir, sub_folder)
#
#     # 跳过非目录文件和新创建的images目录
#     if not os.path.isdir(src_dir) or sub_folder == 'images':
#         continue
#
#     # 获取当前子目录对应的标签[6](@ref)
#     label = label_dict.get(sub_folder, -1)  # -1表示未知类别
#
#     # 处理每个图像文件
#     for img_file in os.listdir(src_dir):
#         src_path = os.path.join(src_dir, img_file)
#         dest_path = os.path.join(new_img_dir, img_file)
#
#         # 复制文件到统一目录[2](@ref)
#         shutil.copyfile(src_path, dest_path)
#
#         # 向DataFrame追加记录[3,4](@ref)
#         new_row = pd.DataFrame({'patch_id': dest_path,'label': label}, index=[0])
#         df = pd.concat([df, new_row], ignore_index=True)  # 合并并重置索引
#
# # 保存为CSV文件[3,4](@ref)
# csv_path = os.path.join(patch_dir, 'labels.csv')
# df.to_csv(csv_path, index=False)  # 不保存索引列
# print(f"CSV文件已保存至：{csv_path}")



import os
import re

def get_prefix_list(folder_path):
    prefix_set = set()  # 用集合自动去重
    for root, dirs, files in os.walk(folder_path):  # 遍历文件夹[3,6,8](@ref)
        for file in files:
            if file.endswith(".h5"):
                # 分割文件名和扩展名[5](@ref)
                filename = os.path.splitext(file)[0]
                # 正则匹配第一个 "-" 或 "_" 前的部分[10](@ref)
                prefix = re.split(r"[-_]", filename, maxsplit=1)[0]
                prefix_set.add(prefix)
    return list(prefix_set)

# 使用示例
folder_path = "/NAS2/Data4/llb/Data/CRC/features/256/resnet50_1024/h5_files"  # 替换为您的文件夹路径
result_list = get_prefix_list(folder_path)
print(result_list)