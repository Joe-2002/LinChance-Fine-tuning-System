
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os

model_dir = snapshot_download('shakechen/Llama-2-7b',cache_dir='/root/autodl-tmp/models')