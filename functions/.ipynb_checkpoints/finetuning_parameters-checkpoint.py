# 用于存储不同语言的文本
language_texts = {
    "English": {
        "sidebar_title": "Fine-tuning Parameters",
        "info_btn_text": "ℹ️ Info",
        "dataset_directory": "Dataset Directory:",
        "dataset_info": "Directory where your dataset is located.",
        "dataset_name": "Dataset Name:",
        "dataset_name_info": "Name of your dataset.",
        "model_name_or_path": "Model Name or Path:",
        "model_name_or_path_info": "Name or path of the pre-trained model.",
        "output_directory": "Output Directory:",
        "output_dir_info": "Directory where the fine-tuned model will be saved.",
        "batch_size_per_device": "Batch Size per Device:",
        
        "batch_size_info": """
        Number of samples processed in one iteration on each GPU. \n
        📈 Larger batch size can lead to faster training but may require more GPU memory. \n
        📉 Smaller batch size can reduce GPU memory usage but may slow down training.
        """,
        
        "accumulation_steps": "Gradient Accumulation Steps:",
        "accumulation_steps_info": """
        Number of steps before backpropagation and optimization. \n
        📈 Increasing accumulation steps can accumulate gradients over more steps and may help with memory constraints. \n
        📉 Decreasing accumulation steps can reduce memory usage but may slow down training.
        """,
        
        "learning_rate": "Learning Rate:",
        "learning_rate_info": """
        Rate at which the model's parameters are updated. \n
        📈 Increasing learning rate can lead to faster convergence but may cause instability. \n
        📉 Decreasing learning rate can make the training more stable but may slow down convergence.
        """,
        
        "num_train_epochs": "Number of Training Epochs:",
        "epochs_info": """
        Number of times the entire dataset is passed through the model. \n
        📈 Increasing epochs can improve model performance but may increase training time. \n
        📉 Decreasing epochs may result in underfitting.
        """,
        
        "lora_rank": "LoRA Rank:",
        "lora_rank_info": """
        Rank parameter for LoRA method. \n
        📈 Increasing rank can capture more complex patterns but may increase computation. \n
        📉 Decreasing rank may simplify the model but may lose information.
        """,
        
        "lora_alpha": "LoRA Alpha:",
        "lora_alpha_info": """
        Alpha parameter for LoRA method. \n
        📈 Increasing alpha can give more weight to local context but may overfit. \n
        📉 Decreasing alpha may make the model more globally focused but may underfit.
        """,
        
        "eval_steps": "Evaluation Steps:",
        "eval_steps_info": """
        Number of steps between model evaluation. \n
        📈 Increasing evaluation steps can reduce evaluation frequency but may save time. \n
        📉 Decreasing evaluation steps may provide more frequent feedback but may increase time.
        """,
        
        "logging_steps": "Logging Steps:",
        "logging_steps_info": """
        Number of steps between logs. \n
        📈 Increasing logging steps can reduce log frequency but may save space. \n
        📉 Decreasing logging steps may provide more detailed logs but may use more space.
        """,
        
        "save_steps": "Save Steps:",
        "save_steps_info": """
        Number of steps between model saving. \n
        📈 Increasing save steps can reduce model saving frequency but may save space. \n
        📉 Decreasing save steps may provide more frequent model checkpoints but may use more space.
        """,
        
        "enable_fp16": "Enable FP16",
        "fp16_info": """
        Enable or disable FP16 training. \n
        📈 Enabling FP16 can reduce memory usage but may affect numerical stability. \n
        📉 Disabling FP16 may require more GPU memory but may improve numerical stability.
        """,
    },
    "中文": {
        "sidebar_title": "微调参数",
        "info_btn_text": "ℹ️ 信息",
        "dataset_directory": "数据集目录：",
        "dataset_info": "数据集所在的目录。",
        "dataset_name": "数据集名称：",
        "dataset_name_info": "您的数据集的名称。",
        "model_name_or_path": "模型名称或路径：",
        "model_name_or_path_info": "预训练模型的名称或路径。",
        "output_directory": "输出目录：",
        "output_dir_info": "保存微调模型的目录。",
        "batch_size_per_device": "每个设备的批量大小：",
        "batch_size_info": """
        在每个GPU上一次迭代中处理的样本数量。 \n
        📈 较大的批量大小可能导致更快的训练，但可能需要更多的GPU内存。 \n
        📉 较小的批量大小可以减少GPU内存使用，但可能减慢训练速度。
        """,
        "accumulation_steps": "梯度累积次数：",
        "accumulation_steps_info": """
        反向传播和优化之前的步数。 \n
        📈 增加累积步数可以在更多步骤上累积梯度，并可能有助于内存约束。 \n
        📉 减少累积步数可以减少内存使用，但可能减慢训练速度。
        """,
        "learning_rate": "学习率：",
        "learning_rate_info": """
        模型参数更新的速率。 \n
        📈 增加学习率可能导致更快的收敛，但可能导致不稳定性。 \n
        📉 减小学习率可以使训练更稳定，但可能减慢收敛速度。
        """,
        "num_train_epochs": "训练轮数：",
        "epochs_info": """
        整个数据集通过模型的次数。 \n
        📈 增加轮数可以提高模型性能，但可能增加训练时间。 \n
        📉 减少轮数可能导致欠拟合。
        """,
        "lora_rank": "LoRA等级：",
        "lora_rank_info": """
        LoRA方法的等级参数。 \n
        📈 增加等级可以捕捉更复杂的模式，但可能增加计算。 \n
        📉 减小等级可能简化模型，但可能丢失信息。
        """,
        "lora_alpha": "LoRA Alpha：",
        "lora_alpha_info": """
        LoRA方法的Alpha参数。 \n
        📈 增加Alpha可以更多地权衡局部上下文，但可能过拟合。 \n
        📉 减小Alpha可能使模型更加全局，但可能欠拟合。
        """,
        "eval_steps": "评估步数：",
        "eval_steps_info": """
        模型评估之间的步数。 \n
        📈 增加评估步数可以减少评估频率，但可能节省时间。 \n
        📉 减少评估步数可能提供更频繁的反馈，但可能增加时间。
        """,
        "logging_steps": "日志记录步数：",
        "logging_steps_info": """
        记录之间的步数。 \n
        📈 增加日志记录步数可以减少日志频率，但可能节省空间。 \n
        📉 减少日志记录步数可能提供更详细的日志，但可能使用更多的空间。
        """,
        "save_steps": "保存步数：",
        "save_steps_info": """
        模型保存之间的步数。 \n
        📈 增加保存步数可以减少模型保存频率，但可能节省空间。 \n
        📉 减少保存步数可能提供更频繁的模型检查点，但可能使用更多的空间。
        """,
        "enable_fp16": "启用FP16",
        "fp16_info": """
        启用或禁用FP16训练。 \n
        📈 启用FP16可以减少内存使用，但可能影响数值稳定性。 \n
        📉 禁用FP16可能需要更多的GPU内存，但可能提高数值稳定性。
        """,
    },
}