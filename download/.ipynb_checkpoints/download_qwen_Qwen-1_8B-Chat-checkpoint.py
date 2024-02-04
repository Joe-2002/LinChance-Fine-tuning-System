# 下载glm3
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('qwen/Qwen-1_8B-Chat', cache_dir='/root/autodl-tmp/models')