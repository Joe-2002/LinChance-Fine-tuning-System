python /root/LinChance-Fine-tuning-System/LLaMA-Factory/src/export_model.py \
    --model_name_or_path /root/autodl-tmp/models/shakechen/Llama-2-7b \
    --adapter_name_or_path /root/LinChance-Fine-tuning-System/output_models/output_llama2 \
    --template default \
    --finetuning_type lora \
    --export_dir /root/LinChance-Fine-tuning-System/merge/merge_model \
    --export_size 2 \
    --export_legacy_format False