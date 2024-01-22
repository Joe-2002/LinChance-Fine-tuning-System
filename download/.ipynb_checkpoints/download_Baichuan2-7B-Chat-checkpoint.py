import torch
import os
from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('baichuan-inc/Baichuan2-7B-Chat', cache_dir='/root/autodl-tmp/models', revision='master')