
# 下载glm3
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('sdfdsfe/bert-base-uncased', cache_dir='/root/autodl-tmp/models')