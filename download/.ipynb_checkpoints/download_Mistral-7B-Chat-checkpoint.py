# 下载glm3
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('AI-ModelScope/Mistral-7B-v0.1', cache_dir='/root/autodl-tmp/models')