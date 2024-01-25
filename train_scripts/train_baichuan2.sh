CUDA_VISIBLE_DEVICES=0 python /root/LinChance-Fine-tuning-System/LLaMA-Factory/src/train_bash.py \
    --stage sft \
    --model_name_or_path /root/autodl-tmp/models/baichuan-inc/Baichuan2-7B-Chat \
    --do_train \
    --dataset_dir /root/LinChance-Fine-tuning-System/LLaMA-Factory/data \
    --dataset lima \
    --template baichuan \
    --finetuning_type lora \
    --lora_target W_pack \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --eval_steps 500 \
    --lr_scheduler_type cosine \
    --logging_steps 50 \
    --save_steps 200 \
    --overwrite_output_dir \
    --output_dir /root/LinChance-Fine-tuning-System/output_models/output_baichuan2 \
    --learning_rate 5e-05 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --lora_rank 8 \
    --lora_alpha 20 \
    --fp16
    