import os
import shutil

import pandas as pd

base_dir = rf'/NAS2/Data4/llb/Data/'
dest_dir = rf'/NAS2/Data4/llb/Data/CRC/slides'
files = ['中日友好医院结直肠癌数据', '协和医院结直肠癌数据', '浙江省肿瘤医院结直肠癌数据']
label = []
slide_id = []
source = []
for file in files:
    a_dir = os.path.join(base_dir, file)
    a_dirs = os.listdir(a_dir)
    for cls_dir in a_dirs:
        if cls_dir == 'LS':
            clazz = 1
        elif cls_dir == 'nonLS':
            clazz = 0
        else:
            continue
        b_dir = os.path.join(a_dir, cls_dir)
        b_dirs = os.listdir(b_dir)
        for patient_dir in b_dirs:
            c_dir = os.path.join(b_dir, patient_dir)
            for slide in os.listdir(c_dir):
                label.append(clazz)
                slide_id.append(slide)
                source.append(file)
                src_path = os.path.join(c_dir, slide)
                dst_path = os.path.join(dest_dir, slide)
                shutil.move(src_path, dst_path)
data = {'slide_id': slide_id, 'label': label, 'data_source': source}
df = pd.DataFrame(data)
label_path = r'/NAS2/Data4/llb/Data/CRC/labels/label.csv'
df.to_csv(label_path, index=False)
