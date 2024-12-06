# import os
#
# import pandas as pd
#
# # dir = r'/data2/lbliao/Data/前列腺癌数据/CKPan/points/'
# # json = os.listdir(dir)
# # json = [f.replace('.json', '.kfb') for f in json if f.endswith('.json')]
# # json = [f for f in json if '-CK' not in f]
# # print(json)
#
# dir = r'/data2/lbliao/Data/前列腺癌数据/数据验证/slides/'
# kfb = os.listdir(dir)
# df = pd.DataFrame(columns=['slide_id', 'label'])
# df[ 'slide_id'] = kfb
# df['label'] = [1] * len(kfb)
# df.to_csv('test.csv')
# import os
#
# he_dir = r'/data2/lbliao/Data/MSI/pair/1024/MSH6/trainA/'
# ihc_dir = r'/data2/lbliao/Data/MSI/pair/1024/MSH6/trainB/'
#
#
# # 遍历目录
# for filename in os.listdir(he_dir):
#     he_path = os.path.join(he_dir, filename)
#     ihc_path = os.path.join(ihc_dir, filename)
#
#     if os.path.isfile(he_path) and os.path.isfile(ihc_path):
#         filesize = os.path.getsize(ihc_path)
#
#         # 如果文件大小小于100KB，则删除
#         if filesize < 130 * 1024:  # 100KB = 100 * 1024 bytes
#             os.remove(he_path)
#             os.remove(ihc_path)
#             print(f"Deleted {filename} due to size less than 100KB")
import os.path

path = rf'/data2/lbliao/Data/前列腺癌数据/CKPan/slides/'
ihc_path = rf'/data2/lbliao/Data/前列腺癌数据/CKPan/IHC/'
ls = ['1734281.11.kfb', '1547583.13.kfb', '1641996.7.kfb', '1641996.6.kfb', '1641996.11.kfb', '1641996.2.kfb', '1604701.16.kfb', '1641996.8.kfb', '1547583.14.kfb', '1638897.16.kfb', '1638897.13.kfb', '1638897.9.kfb', '1638897.15.kfb', '1636600.10.kfb', '1547583.20.kfb', '1641996.5.kfb', '1642001.1.kfb', '1547583.12.kfb', '1604701.12.kfb', '1638897.12.kfb', '1547583.10.kfb', '1641996.10.kfb', '1638897.19.kfb', '1547583.17.kfb', '1641996.4.kfb', '1547583.18.kfb', '1638897.11.kfb']
for wsi in ls:
    ihc = wsi.replace('.kfb', '-CK.kfb')
    if os.path.isfile(os.path.join(path, ihc)):
        os.rename(os.path.join(path,ihc), os.path.join(ihc_path, ihc))