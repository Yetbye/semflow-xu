import torch
from safetensors.torch import load_file

# 替换成你的实际路径
path = "work_dirs/semflow/checkpoint-85000/model.safetensors" 

try:
    sd = load_file(path)
except:
    sd = torch.load(path, map_location="cpu")

keys = list(sd.keys())
print(keys)
print("包含 RRDB 吗?", any("RRDB" in k for k in keys))
print("包含 embedding_table 吗?", any("embedding_table" in k for k in keys))