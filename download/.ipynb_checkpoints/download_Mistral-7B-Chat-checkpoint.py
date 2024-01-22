# 下载glm3
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('TabbyML/Mistral-7B', cache_dir='/root/autodl-tmp/models', revision='master')