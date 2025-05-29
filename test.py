import os

dir = f'/NAS2/Data1/lbliao/Data/MSI/0327/slides'
sl_li = ['2430307', '2432037', '2433237', '2434780', '2437749', '2437926', '2442659', '2446627', '2449557', '2451800', '2451906', '2467184', '2469024', '2470526', '2480514', '2500404', '2501490', '2501504', '2504614', '2507625', '2509278']

files = os.listdir(dir)
result = []
no = []
for s in sl_li:
    if f'{s}.kfb' in files:
        result.append(f'{s}.kfb')
    elif f'{s}.svs' in files:
        result.append(f'{s}.svs')
    else:
        no.append(s)
print(result)
print(len(result))
print(no)
# import os
# import shutil
#
#
# def copy_geojson_files(src_dir, dst_dir):
#     # 创建目标目录（若不存在）
#     if not os.path.exists(dst_dir):
#         os.makedirs(dst_dir)
#         print(f"创建目标目录：{dst_dir}")
#
#     # 遍历源目录及其子目录
#     for root, dirs, files in os.walk(src_dir):
#         for file in files:
#             # 筛选.geojson文件
#             if file.endswith(".geojson"):
#                 src_path = os.path.join(root, file)
#                 dst_path = os.path.join(dst_dir, file)
#
#                 # 复制文件并保留元数据
#                 if not os.path.exists(dst_path):
#                     shutil.copy(src_path, dst_path)
#                     print(f"已复制：{src_path} -> {dst_path}")
#
#
# # 路径配置
# srcdir = '/NAS2/Data1/lbliao/Data/MSI/虚拟染色课题（MSI）'
# dstdir = '/NAS2/Data1/lbliao/Data/MSI/0327/points'
#
# # 执行复制操作
# copy_geojson_files(srcdir, dstdir)