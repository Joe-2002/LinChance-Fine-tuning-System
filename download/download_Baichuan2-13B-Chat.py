import torch
import os
from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('baichuan-inc/Baichuan2-13B-Chat', cache_dir='/root/autodl-tmp/models/baichuan2_13b', revision='master')