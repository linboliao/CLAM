import os

from huggingface_hub import login, hf_hub_download

# 使用您的用户访问令牌登录，可在 https://huggingface.co/settings/tokens 找到
login(token='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')

# 指定下载目录
local_dir = "uni/"
os.makedirs(local_dir, exist_ok=True)  # 如果目录不存在则创建

# 下载模型参数
hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)