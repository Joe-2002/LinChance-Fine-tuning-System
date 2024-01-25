# merge.py

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from datetime import datetime
import sys

model_path = sys.argv[1]
model_save_path = sys.argv[2]

def merge_lora_to_base_model():
    # 定义或获取当前时间
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")

    model_name_or_path = model_path
    adapter_name_or_path = model_save_path
    save_path = f'/root/LinChance-Fine-tuning-System/merge_model/baichuan2_finetuned_{current_time}'

    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False if config.model_type == 'llama' else True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map={'': 'cpu'}
    )
    model = PeftModel.from_pretrained(model, adapter_name_or_path, device_map={'': 'cpu'})
    model = model.merge_and_unload()

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    
    print(f"Merge completed! Successfully saved in {save_path}")

if __name__ == '__main__':
    merge_lora_to_base_model()
