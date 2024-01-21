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
 
import openai
import streamlit as st
from streamlit_chat import message

model=None
selected_dataset =None
options = None
script_path=None
model_name = None
train_scripts = None

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
    st.image("./img/2024-01-15_185750.png", width=600,)
    model = "./download/download_ChatGLM3-6B-Chat.py"
    model_name = "ChatGLM3-6B-Chat"
    train_scripts = "/root/LLaMA-Factory/train_chatglm3.sh"
    # train_scripts = "/root/LLaMA-Factory/train.sh"
if options == "Mistral-7B-Chat":
    st.header(':red[Mistral-7B-Chat]')
    st.image("./img/v2-275337791c4e6cb2bed40b0639f4d847_r.png", width=600,)
    model = "./download/download_Mistral-7B-Chat.py"
    model_name = "Mistral-7B-Chat"
    train_scripts = "/root/LLaMA-Factory/Mistral_ans"
if options == "Llama2-7B-Chat":
    st.header(':red[Llama2-7B-Chat]')
    st.image("./img/llama2.png", width=600,)
    model = "./download/download_Llama2-7B-Chat.py"
    model_name = "Llama2-7B-Chat"
    train_scripts = "/root/LLaMA-Factory/Llama2_ans.sh"
if options == "Baichuan2-7B-Chat":
    st.header(':red[Baichuan2-7B-Chat]')
    st.image("./img/2024-01-15_213753.png", width=600,)
    model = "./download/download_Baichuan2-7B-Chat.py"
    model_name = "Baichuan2-7B-Chat"
    train_scripts = "/root/LLaMA-Factory/Baichuan_ans.sh"
########


    
    

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
    data_dir = "/root/LLaMA-Factory/data"
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

# Call the functions
on = st.sidebar.button("upload dataset")
if on:
    upload_and_display_data()

select_and_display_dataset()
                




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
                st.text(line.strip())

        # 等待下载完成
        process.wait()

        # 显示整个下载输出
        st.code(download_output.getvalue())

        # 显示成功消息
        st.success("Download Complete")

st.title("Fine-tuning-scripts")
st.caption(f'Selected Train_scripts: :green[{model_name}]')
expander = st.expander("View first five data entries", expanded=False)

with expander:
    if train_scripts:
        # st.info(f"Displaying content of selected script: {train_scripts}")
        with open(train_scripts, 'r', encoding='utf-8') as script_file:
            script_content = script_file.read()
            # first_five_data = script_content[:5]
            st.code(script_content, language="sh")
    else:
        st.warning("No script selected.")

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
    

# learning_rate = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.1)
# num_epochs = st.sidebar.slider("Number of Epochs", 1, 50, 10)
# UI 标题
# 侧边栏标题，突出显示、加粗并加上颜色
st.sidebar.markdown('<span style="color: #ff6347; font-size: 1.5em; font-weight: bold;">Fine-tuning Parameters</span>', unsafe_allow_html=True)

# 使用 st.sidebar.beta_expander 创建一个侧边栏部件
with st.sidebar.expander("Fine-tuning Parameters", expanded=False):
    # 数据集选择
    dataset_dir = st.text_input("Dataset Directory:", "/root/LLaMA-Factory/data")
    dataset_name = st.text_input("Dataset Name:", "lima")

    # 模型相关参数
    model_name_or_path = st.text_input("Model Name or Path:", "/root/autodl-tmp/ZhipuAI/chatglm3-6b")
    output_dir = st.text_input("Output Directory:", "/root/LLaMA-Factory/output_chatglm3")

    # 训练参数
    per_device_train_batch_size = st.slider("Batch Size per Device:", 1, 8, 2)
    gradient_accumulation_steps = st.slider("Gradient Accumulation Steps:", 1, 10, 2)
    learning_rate = st.number_input("Learning Rate:", value=5e-5)
    num_train_epochs = st.slider("Number of Training Epochs:", 1, 10, 3)

    # lora 参数
    lora_rank = st.slider("LoRA Rank:", 1, 20, 10)
    lora_alpha = st.slider("LoRA Alpha:", 1, 50, 20)

    # 其他参数
    eval_steps = st.number_input("Evaluation Steps:", value=500)
    logging_steps = st.number_input("Logging Steps:", value=50)
    save_steps = st.number_input("Save Steps:", value=500)

    # 是否启用 FP16
    fp16 = st.checkbox("Enable FP16", value=True)

# 训练按钮
if st.sidebar.button("save"):
    # 构建命令
    command = (
        f"CUDA_VISIBLE_DEVICES=0 python /root/LLaMA-Factory/src/train_bash.py \n"
        f"--stage sft \n"
        f"--model_name_or_path {model_name_or_path} \n"
        f"--do_train \n"
        f"--dataset_dir {dataset_dir} \n"
        f"--dataset {dataset_name} \n"
        f"--template chatglm3 \n"
        f"--finetuning_type lora \n"
        f"--lora_target query_key_value \n"
        f"--overwrite_cache \n"
        f"--per_device_train_batch_size {per_device_train_batch_size} \n"
        f"--gradient_accumulation_steps {gradient_accumulation_steps} \n"
        f"--eval_steps {eval_steps} \n"
        f"--lr_scheduler_type cosine \n"
        f"--logging_steps {logging_steps} \n"
        f"--save_steps {save_steps} \n"
        f"--overwrite_output_dir \n"
        f"--output_dir {output_dir} \n"
        f"--learning_rate {learning_rate} \n"
        f"--num_train_epochs {num_train_epochs} \n"
        f"--plot_loss \n"
        f"--lora_rank {lora_rank} \n"
        f"--lora_alpha {lora_alpha} \n"
        f"--fp16"
    )
    with open("/root/LLaMA-Factory/train.sh", "w") as file:
        file.write(command)
    # # 打印命令并执行
    st.code(f"Running command: {command}",language="sh")
def run_training_script():
    # Replace 'train.py' with the actual path to your Streamlit app script
    script_path = './train.py'

    # Open a new terminal window and run the Streamlit app
    os.system(f"python {script_path}")

    st.success("Model training process started successfully!")



