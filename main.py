import io
import json
import os
import time
import numpy as np
import streamlit as st
import pandas as pd
from io import StringIO
import subprocess
import torch
import torchvision.transforms as transforms
from PIL import Image
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import TaskType, get_peft_model, LoraConfig
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from datetime import datetime
import base64
import zipfile
import os
import shutil

# import openai
import streamlit as st
from streamlit_chat import message
from transformers import AutoModel, AutoTokenizer
from datetime import datetime


model=None
selected_dataset =None
options = None
script_path=None
model_name = None
train_scripts = None
model_path = None
global on 
global model_output_dir
lora_target = None 
model_save_path = None # 模型保存路径
merge_scripts = None # 合并脚本路径

#############
st.sidebar.markdown('<span style="color: #ff6347; font-size: 1.5em; font-weight: bold;">Model</span>', unsafe_allow_html=True)

st.title('Model Fine-tuning System')

# 模型选择
model_list =  ("ChatGLM3-6B-Chat", "Mistral-7B-Chat", "Llama2-7B-Chat", "Baichuan2-7B-Chat")

options = st.sidebar.selectbox(
    "Select the model you want to fine-tune",
    model_list,
    # index=None,
)

if options == "ChatGLM3-6B-Chat":
    st.header(':red[ChatGLM3-6B-Chat]')
    st.image("/root/LinChance-Fine-tuning-System/images/chatglm.png", width=600,)
    model = "/root/LinChance-Fine-tuning-System/download/download_ChatGLM3-6B-Chat.py"
    model_name = "ChatGLM3-6B-Chat"
    train_scripts = "/root/LinChance-Fine-tuning-System/train_scripts/train_chatglm3.sh"
    model_path = '/root/autodl-tmp/models/ZhipuAI/chatglm3-6b'
    model_output_dir = "/root/LinChance-Fine-tuning-System/output_models/output_chatglm3"
    lora_target = "query_key_value"
    # train_scripts = "/root/LLaMA-Factory/train.sh"
if options == "Mistral-7B-Chat":
    st.header(':red[Mistral-7B-Chat]')
    st.image("/root/LinChance-Fine-tuning-System/images/mistral.png", width=600,)
    model = "/root/LinChance-Fine-tuning-System/download/download_Mistral-7B-Chat.py"
    model_name = "Mistral-7B-Chat"
    train_scripts = "/root/LLaMA-Factory/Mistral_ans"
    model_path = '/root/autodl-tmp/models/TabbyML/Mistral-7B'
    model_output_dir = "/root/LinChance-Fine-tuning-System/output_models/output_mistral"
if options == "Llama2-7B-Chat":
    st.header(':red[Llama2-7B-Chat]')
    st.image("/root/LinChance-Fine-tuning-System/images/llama2.png", width=600,)
    model = "/root/LinChance-Fine-tuning-System/download/download_Llama2-7B-Chat.py"
    model_name = "Llama2-7B-Chat"
    train_scripts = "/root/LLaMA-Factory/Llama2_ans.sh"
    model_path = '/root/autodl-tmp/models/Llama2-7B-Chat'
    model_output_dir = "/root/LinChance-Fine-tuning-System/output_models/output_llama2"
if options == "Baichuan2-7B-Chat":
    st.header(':red[Baichuan2-7B-Chat]')
    st.image("/root/LinChance-Fine-tuning-System/images/baichuan.png", width=600,)
    model = "/root/LinChance-Fine-tuning-System/download/download_Baichuan2-7B-Chat.py"
    model_name = "Baichuan2-7B-Chat"
    train_scripts = "/root/LinChance-Fine-tuning-System/train_scripts/train_baichuan2.sh"
    model_path = '/root/autodl-tmp/models/baichuan-inc/Baichuan2-7B-Chat'
    model_output_dir = "/root/LinChance-Fine-tuning-System/output_models/output_baichuan2"
    lora_target = "W_pack"
    merge_scripts = "/root/LinChance-Fine-tuning-System/merge/merge_scripts/baichuan_export.sh"
########


# 检查本地是否已经存在指定的模型文件夹
models_to_check = ["chatglm3-6b", "Mistral-7B", "Llama-2-7b", "Baichuan2-7B-Chat"]
model_exists = any(os.path.exists(os.path.join("models", model_folder)) for model_folder in models_to_check)

# 如果模型不存在，则显示下载按钮
if not model_exists:
    st.title('Model download')
    # 添加内容
    st.write("Stick here to start Model download")

    # 添加操作按钮
    if st.button("Download Model", type="primary"):
        # 执行下载命令
        st.text(model)
        script_path = model

        # 创建一个StringIO对象，用于捕获下载输出
        download_output = StringIO()

        # 设置进度条
        progress_bar = st.progress(0)

        # 使用subprocess.Popen启动下载脚本
        process = subprocess.Popen(f"python {script_path}", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)

        # 获取或创建session_state.download_info_placeholder
        if "download_info_placeholder" not in st.session_state:
            st.session_state.download_info_placeholder = st.empty()

        # 显示下载输出的前三行
        for i in range(3):
            line = process.stderr.readline()
            st.session_state.download_info_placeholder.text(line.strip())

        with tqdm(total=None, file=download_output, leave=False, disable=True, dynamic_ncols=True) as t:
            while True:
                # 实时读取stderr输出
                line = process.stderr.readline()
                if not line:
                    break

                # 查找进度信息
                if "Progress:" in line:
                    progress_str = line.split("Progress:")[1].strip()
                    progress = float(progress_str.strip("%")) / 100.0
                    progress_bar.progress(progress)

                # 更新tqdm输出
                t.write(line)

                # 显示下载输出
                st.session_state.download_info_placeholder.text(line.strip())

                # 仅显示前三行
                if t.n > 2:
                    break

        # 等待下载完成
        process.wait()

        # 清空下载信息
        st.session_state.download_info_placeholder.empty()

        # 显示整个下载输出
        st.code(download_output.getvalue())

        # 显示成功消息
        st.success("Download Complete")

        # 设置显示窗口位置和大小
        st.markdown(
            """
            <style>
                .css-1l02zg8 {
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    width: 800px;  /* 设置窗口宽度 */
                    height: 600px; /* 设置窗口高度 */
                }
            </style>
            """,
            unsafe_allow_html=True
        )

        
        
        
# 侧边栏标题，突出显示、加粗并加上颜色
st.sidebar.markdown('<span style="color: #ff6347; font-size: 1.5em; font-weight: bold;">Dataset</span>', unsafe_allow_html=True)

# 上传函数
def upload_and_display_data():
    data_dir = "/root/LLaMA-Factory/data"
    uploaded_file = st.sidebar.file_uploader("Select your dataset and upload it")

    # Automatically name the uploaded dataset as "lima.json"
    file_name = "lima.json"

    if uploaded_file is not None:
        save_file_path = os.path.join(data_dir, file_name)

        # Save the uploaded file as "lima.json"
        with open(save_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        st.sidebar.success(f"Success: {save_file_path}")

def select_and_display_dataset():
    data_dir = "/root/LinChance-Fine-tuning-System/LLaMA-Factory/data"
    file_list = [file for file in os.listdir(data_dir) if file.endswith(".json")]

    if not file_list:
        st.warning("No JSON files found in the data directory.")
    else:
        selected_dataset = st.sidebar.selectbox(
            "Choose your dataset",
            file_list,
            index=0  # Set the default index to 0
        )

        if selected_dataset:
            st.title("Dataset")
            st.caption(f'Selected Dataset: :green[{selected_dataset}]')
            selected_file_path = os.path.join(data_dir, selected_dataset)

            with open(selected_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            first_five_data = data[:5]

            expander = st.expander("View first five data entries", expanded=False)
            with expander:
                for entry in first_five_data:
                    st.code(entry)

upload_and_display_data()

select_and_display_dataset()
                    
# 侧边栏标题
st.sidebar.markdown('<span style="color: #ff6347; font-size: 1.5em; font-weight: bold;">Fine-tuning Parameters</span>', unsafe_allow_html=True)

# 侧边栏


def show_info(info_text):
    st.info(info_text)

with st.sidebar.expander("Fine-tuning Parameters", expanded=False):
    # 数据集选择
    dataset_dir = st.text_input("Dataset Directory:", "/root/LinChance-Fine-tuning-System/LLaMA-Factory/data")
    if st.button("ℹ️ Info", key="dataset_info"):
        show_info("Directory where your dataset is located.")

    dataset_name = st.text_input("Dataset Name:", "lima")
    if st.button("ℹ️ Info", key="dataset_name_info"):
        show_info("Name of your dataset.")

    # 模型相关参数
    model_name_or_path = st.text_input("Model Name or Path:", model_path)
    if st.button("ℹ️ Info", key="model_name_or_path_info"):
        show_info("Name or path of the pre-trained model.")

    output_dir = st.text_input("Output Directory:", model_output_dir)
    if st.button("ℹ️ Info", key="output_dir_info"):
        show_info("Directory where the fine-tuned model will be saved.")

    # 训练参数
    per_device_train_batch_size = st.slider("Batch Size per Device:", 1, 8, 2)
    if st.button("ℹ️ Info", key="batch_size_info"):
        show_info("""
        Number of samples processed in one iteration on each GPU. \n
        📈 Larger batch size can lead to faster training but may require more GPU memory. \n
        📉 Smaller batch size can reduce GPU memory usage but may slow down training.
        """)

    gradient_accumulation_steps = st.slider("Gradient Accumulation Steps:", 1, 10, 2)
    if st.button("ℹ️ Info", key="accumulation_steps_info"):
        show_info("""
        Number of steps before backpropagation and optimization. \n
        📈 Increasing accumulation steps can accumulate gradients over more steps and may help with memory constraints. \n
        📉 Decreasing accumulation steps can reduce memory usage but may slow down training.
        """)

    learning_rate = st.number_input("Learning Rate:", value=5e-5)
    if st.button("ℹ️ Info", key="learning_rate_info"):
        show_info("""
        Rate at which the model's parameters are updated. \n
        📈 Increasing learning rate can lead to faster convergence but may cause instability. \n
        📉 Decreasing learning rate can make the training more stable but may slow down convergence.
        """)

    num_train_epochs = st.slider("Number of Training Epochs:", 1, 10, 3)
    if st.button("ℹ️ Info", key="epochs_info"):
        show_info("""
        Number of times the entire dataset is passed through the model. \n
        📈 Increasing epochs can improve model performance but may increase training time. \n
        📉 Decreasing epochs may result in underfitting.
        """)

    # lora 参数
    lora_rank = st.slider("LoRA Rank:", 1, 20, 10)
    if st.button("ℹ️ Info", key="lora_rank_info"):
        show_info("""
        Rank parameter for LoRA method. \n
        📈 Increasing rank can capture more complex patterns but may increase computation. \n
        📉 Decreasing rank may simplify the model but may lose information.
        """)

    lora_alpha = st.slider("LoRA Alpha:", 1, 50, 20)
    if st.button("ℹ️ Info", key="lora_alpha_info"):
        show_info("""Alpha parameter for LoRA method. \n
        📈 Increasing alpha can give more weight to local context but may overfit. \n
        📉 Decreasing alpha may make the model more globally focused but may underfit.
        """)

    # 其他参数
    eval_steps = st.number_input("Evaluation Steps:", value=500)
    if st.button("ℹ️ Info", key="eval_steps_info"):
        show_info("""
        Number of steps between model evaluation. \n
        📈 Increasing evaluation steps can reduce evaluation frequency but may save time. \n
        📉 Decreasing evaluation steps may provide more frequent feedback but may increase time.
        """)

    logging_steps = st.number_input("Logging Steps:", value=50)
    if st.button("ℹ️ Info", key="logging_steps_info"):
        show_info("""
        Number of steps between logs. \n
        📈 Increasing logging steps can reduce log frequency but may save space. \n
        📉 Decreasing logging steps may provide more detailed logs but may use more space.
        """)

    save_steps = st.number_input("Save Steps:", value=500)
    if st.button("ℹ️ Info", key="save_steps_info"):
        show_info("""
        Number of steps between model saving. \n
        📈 Increasing save steps can reduce model saving frequency but may save space. \n
        📉 Decreasing save steps may provide more frequent model checkpoints but may use more space.
        """)

    # 是否启用 FP16
    fp16 = st.checkbox("Enable FP16", value=True)
    if st.button("ℹ️ Info", key="fp16_info"):
        show_info("""
        Enable or disable FP16 training. \n
        📈 Enabling FP16 can reduce memory usage but may affect numerical stability. \n
        📉 Disabling FP16 may require more GPU memory but may improve numerical stability.
        """)

st.title("Fine-tuning-scripts")
st.caption(f'Selected Train_scripts: :green[{model_name}]')
expander = st.expander("View first five data entries", expanded=False)

# 训练按钮
if st.sidebar.button("save"):
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # 构建保存路径，包含模型名称、训练后标识符和当前时间戳
    # model_save_path = f'{model_output_dir}/{current_time}'
    model_save_path = model_output_dir
    # 构建命令
    command = f"""
CUDA_VISIBLE_DEVICES=0 python /root/LinChance-Fine-tuning-System/LLaMA-Factory/src/train_bash.py \\
    --stage sft \\
    --model_name_or_path {model_name_or_path} \\
    --do_train \\
    --dataset_dir {dataset_dir} \\
    --dataset {dataset_name} \\
    --template chatglm3 \\
    --finetuning_type lora \\
    --lora_target {lora_target} \\
    --overwrite_cache \\
    --per_device_train_batch_size {per_device_train_batch_size} \\
    --gradient_accumulation_steps {gradient_accumulation_steps} \\
    --eval_steps {eval_steps} \\
    --lr_scheduler_type cosine \\
    --logging_steps {logging_steps} \\
    --save_steps {save_steps} \\
    --overwrite_output_dir \\
    --output_dir {model_save_path} \\
    --learning_rate {learning_rate} \\
    --num_train_epochs {num_train_epochs} \\
    --plot_loss \\
    --lora_rank {lora_rank} \\
    --lora_alpha {lora_alpha} \\
    --fp16
"""
    with open("/root/LinChance-Fine-tuning-System/train_scripts/train_chatglm3.sh", "w") as file:
        file.write(command)

    # 在保存后更新 expander 的内容
    with open(train_scripts, 'r', encoding='utf-8') as script_file:
        script_content = script_file.read()
        # 更新 expander 的内容
        expander.code(script_content, language="sh")
    # 显示成功消息
    st.success("参数保存成功！")

# Button to start fine-tuning
if st.button("Start Fine-tuning", type="primary"):
    # Set up progress bar
    progress_bar = st.progress(0)
    
    # Area to display command line output
    console_output = st.empty()
    
    # Print script path
    st.text(f"Running script: {train_scripts}")
    
    # Create StringIO object to capture tqdm output
    tqdm_output = StringIO()

    # Use subprocess.Popen to start the shell script
    process = subprocess.Popen(f"sh {train_scripts}", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)

    # Wrap stderr stream with tqdm
    with tqdm(total=None, file=tqdm_output, leave=False, disable=True, dynamic_ncols=True) as t:
        while True:
            # Read real-time stderr output
            line = process.stderr.readline()
            if not line:
                break

            # Search for progress information
            if "Progress:" in line:
                progress_str = line.split("Progress:")[1].strip()
                progress = float(progress_str.strip("%")) / 100.0
                progress_bar.progress(progress)

            # Update tqdm output
            t.write(line)

            # Display command line output
            console_output.text(line.strip())

    # Wait for the command to complete
    process.wait()

    # Display the entire command line output
    console_output.code(tqdm_output.getvalue())
    torch.cuda.empty_cache()
    # Display success message
    st.success("Model training process completed successfully!")
    

# 合并模型权重
st.title('Fine-tuned Merge')
# 添加内容
st.write("Merge lora to fine-tune model weights")

# 运行合并函数
if st.button("Merge", type="primary"):
    # Set up progress bar
    progress_bar = st.progress(0)
    
    # Area to display command line output
    console_output = st.empty()

    # Print script path
    st.text(f"Running script: {merge_scripts}")
    
    # Create StringIO object to capture tqdm output
    tqdm_output = StringIO()

    # Use subprocess.Popen to start the shell script
    process = subprocess.Popen(f"sh {train_scripts}", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)

    # Wrap stderr stream with tqdm
    with tqdm(total=None, file=tqdm_output, leave=False, disable=True, dynamic_ncols=True) as t:
        while True:
            # Read real-time stderr output
            line = process.stderr.readline()
            if not line:
                break

            # Search for progress information
            if "Progress:" in line:
                progress_str = line.split("Progress:")[1].strip()
                progress = float(progress_str.strip("%")) / 100.0
                progress_bar.progress(progress)

            # Update tqdm output
            t.write(line)

            # Display command line output
            console_output.text(line.strip())

    # Wait for the command to complete
    process.wait()

    # Display the entire command line output
    console_output.code(tqdm_output.getvalue())
    torch.cuda.empty_cache()
    # Display success message
    st.success("Model merge process completed successfully!")   


    
# 定义获取压缩文件内容的函数
def get_zip_file_content(file_path):
    with open(file_path, "rb") as file:
        file_content = file.read()
    return file_content

# 添加一个按钮，当点击时触发下载
st.title('Fine-tuned Model Download')
if st.button("Packing", type="primary"):
    # 在这里添加下载模型的逻辑
    # model_output_dir = "/path/to/your/model/output/directory"

    # 创建一个临时目录用于存储压缩文件
    temp_dir = "/root/LinChance-Fine-tuning-System/temp"
    os.makedirs(temp_dir, exist_ok=True)

    # 设置压缩文件路径
    zip_file_path = os.path.join(temp_dir, "fine_tuned_model.zip")

    # 使用 shutil.make_archive 创建压缩文件
    shutil.make_archive(zip_file_path[:-4], 'zip', model_output_dir)

    # 获取压缩文件内容
    zip_file_content = get_zip_file_content(zip_file_path)

    # 提供下载按钮
    st.download_button(
        label="Download",
        type="primary",
        data=zip_file_content,
        file_name="fine_tuned_model.zip",
        mime="application/zip"
    )



st.title("Trained Model Chat")
def chat_with_mistral():
    
    return 0
def chat_with_llama2():
    return 0
def chat_with_baichuan2():
    return 0
on = st.toggle("chat with trained model")
if on:
    if model_name == "ChatGLM3-6B-Chat":
        @st.cache_resource
        def get_model_chatglm3():
            MODEL_PATH = os.environ.get('MODEL_PATH', model_path)
            TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
            model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()
            return tokenizer, model

        def chat_with_chatglm3():
            tokenizer, model = get_model_chatglm3()

            if "history" not in st.session_state:
                st.session_state.history = []
            if "past_key_values" not in st.session_state:
                st.session_state.past_key_values = None

            max_length = st.sidebar.slider("max_length", 0, 32768, 8192, step=1)
            top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
            temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.6, step=0.01)

            buttonClean = st.sidebar.button("清理会话历史", key="clean")
            if buttonClean:
                st.session_state.history = []
                st.session_state.past_key_values = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                st.rerun()

            for i, message in enumerate(st.session_state.history):
                if message["role"] == "user":
                    with st.chat_message(name="user", avatar="user"):
                        st.markdown(message["content"])
                else:
                    with st.chat_message(name="assistant", avatar="assistant"):
                        st.markdown(message["content"])

            with st.chat_message(name="user", avatar="user"):
                input_placeholder = st.empty()
            with st.chat_message(name="assistant", avatar="assistant"):
                message_placeholder = st.empty()

            prompt_text = st.text_input("请输入您的问题")  # Use text_input instead of chat_input for caching issues
            if prompt_text:
                input_placeholder.markdown(prompt_text)
                history = st.session_state.history
                past_key_values = st.session_state.past_key_values
                for response, history, past_key_values in model.stream_chat(
                        tokenizer,
                        prompt_text,
                        history,
                        past_key_values=past_key_values,
                        max_length=max_length,
                        top_p=top_p,
                        temperature=temperature,
                        return_past_key_values=True,
                ):
                    message_placeholder.markdown(response)
                st.session_state.history = history
                st.session_state.past_key_values = past_key_values
                
        chat_with_chatglm3()

    if model_name == "Mistral-7B-Chat":
        chat_with_mistral()
    if model_name == "Llama2-7B-Chat":
        chat_with_llama2()
    if model_name == "Baichuan2-7B-Chat":
        chat_with_baichuan2()
