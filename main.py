import io
import json
import os
import time
import numpy as np
import streamlit as st
import pandas as pd
from io import StringIO
import subprocess
import torchvision.transforms as transforms
from PIL import Image
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer,AutoConfig,AutoModel
from peft import TaskType, get_peft_model, LoraConfig,PeftModel
from tqdm import tqdm
import torch
import base64
import zipfile
import os
import shutil
from threading import Thread
from streamlit_chat import message
from datetime import datetime
import optuna
import psutil


from functions.start_function import start
from functions.data_utils import upload_and_display_data, select_and_display_dataset
from functions.finetuning_parameters import language_texts

st.set_page_config(
    page_title="Home",
    page_icon="👋",
)
# 定义全局变量

model=None
selected_dataset =None
options = None
script_path=None
model_name = None
train_scripts = None
model_path = None
on = None 
model_output_dir = None
lora_target = None 
model_save_path = None # 模型保存路径
merge_scripts = None # 合并脚本路径
template = None
models_to_check = None
chat_on = None
train_on = None
download_on = None



#############
st.sidebar.markdown('<span style="color: #ff6347; font-size: 1.5em; font-weight: bold;">Model</span>', unsafe_allow_html=True)

st.title('Model Fine-tuning System')

# 模型选择
model_list =  ("ChatGLM3-6B-Chat", "Mistral-7B-v0.1", "Llama2-7B-Chat", "Baichuan2-7B-Chat","Baichuan2-13B-Chat")

options = st.sidebar.selectbox(
    "Select the model you want to fine-tune",
    model_list,
    # index=None,
)

if options == "ChatGLM3-6B-Chat":
    st.header(':red[ChatGLM3-6B-Chat]')
    st.image("./images/chatglm.png", width=600,)
    model_download = "./download/download_ChatGLM3-6B-Chat.py"
    model_name = "ChatGLM3-6B-Chat"
    train_scripts = "train_scripts/train_chatglm3.sh"
    model_path = '/root/autodl-tmp/models/ZhipuAI/chatglm3-6b'
    model_output_dir = "./output_models/output_chatglm3"
    merge_scripts = "merge/merge_scripts/chatglm3_export.sh"
    lora_target = "query_key_value"
    template = "chatglm3"
    models_to_check = "ZhipuAI"
    # train_scripts = "/root/LLaMA-Factory/train.sh"
if options == "Mistral-7B-v0.1":
    st.header(':red[Mistral-7B-v0.1]')
    st.image("./images/mistral.png", width=600,)
    model_download = "./download/download_Mistral-7B-Chat.py"
    model_name = "Mistral-7B-v0.1"
    train_scripts = "./train_scripts/train_mistral.sh"
    model_path = '/root/autodl-tmp/models/AI-ModelScope/Mistral-7B-v0.1'
    model_output_dir = "output_models/output_mistral"
    merge_scripts = "merge/merge_scripts/mistral_export.sh"
    lora_target = "q_proj,v_proj"
    template = "mistral"
    models_to_check =  "AI-ModelScope"
if options == "Llama2-7B-Chat":
    st.header(':red[Llama2-7B-Chat]')
    st.image("./images/llama2.png", width=600,)
    model_download = "./download/download_Llama2-7B-Chat.py"
    model_name = "Llama2-7B-Chat"
    train_scripts = "./train_scripts/train_llama2.sh"
    model_path = '/root/autodl-tmp/models/shakechen/Llama-2-7b'
    model_output_dir = "./output_models/output_llama2"
    merge_scripts = "merge/merge_scripts/llama2_export.sh"
    lora_target = "q_proj,v_proj"
    template = "llama2"
    models_to_check = "shakechen"
if options == "Baichuan2-7B-Chat":
    st.header(':red[Baichuan2-7B-Chat]')
    st.image("./images/baichuan.png", width=600,)
    model_download = "./download/download_Baichuan2-7B-Chat.py"
    model_name = "Baichuan2-7B-Chat"
    train_scripts = "./train_scripts/train_baichuan2.sh"
    model_path = '/root/autodl-tmp/models/baichuan-inc/Baichuan2-7B-Chat'
    model_output_dir = "./output_models/output_baichuan2"
    lora_target = "W_pack"
    merge_scripts = "merge/merge_scripts/baichuan_export.sh"
    template = "baichuan2"
    models_to_check = "baichuan-inc"
if options == "Baichuan2-13B-Chat":
    st.header(':red[Baichuan2-13B-Chat]')
    st.image("./images/baichuan.png", width=600,)
    model_download = "./download/download_Baichuan2-13B-Chat.py"
    model_name = "Baichuan2-13B-Chat"
    train_scripts = "./train_scripts/train_baichuan2_13b.sh"
    model_path = '/root/autodl-tmp/models/baichuan-inc/Baichuan2-13B-Chat'
    model_output_dir = "./output_models/output_baichuan2_13b"
    lora_target = "W_pack"
    merge_scripts = "merge/merge_scripts/baichuan_export.sh"
    template = "baichuan2"
    models_to_check = "baichuan2_13b"


### Model Download
model_folder = "../autodl-tmp/models"
model_exists = os.path.exists(os.path.join(model_folder, models_to_check))
if model_exists:
    st.title('Model Already Downloaded')
    st.write(f"The selected model (<span style='color: green;'>{model_name}</span>) is already downloaded. You can proceed to the next step.",unsafe_allow_html=True)
else:
    st.title('Model Download')
    st.write(f"The selected model (<span style='color: red;'>{model_name}</span>) is not downloaded. Please download it.", unsafe_allow_html=True)
    down_on = st.button("Download Model", type="primary")
    if down_on:
        script_path = model_download 
        download_output = StringIO()
        progress_bar = st.progress(0)
        process = subprocess.Popen(f"python {script_path}", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)
        if "download_info_placeholder" not in st.session_state:
            st.session_state.download_info_placeholder = st.empty()
        with tqdm(total=None, file=download_output, leave=False, disable=True, dynamic_ncols=True) as t:
            while True:
                line = process.stderr.readline()
                if not line:
                    break
                if "Progress:" in line:
                    progress_str = line.split("Progress:")[1].strip()
                    progress = float(progress_str.strip("%")) / 100.0
                    progress_bar.progress(progress)
                t.write(line)
                st.session_state.download_info_placeholder.text(line.strip())
                if t.n > 2:
                    break
        st.session_state.download_info_placeholder.empty()
        st.code(download_output.getvalue())
        st.success("Download Complete")
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


### Upload dataset    

st.sidebar.markdown('<span style="color: #ff6347; font-size: 1.5em; font-weight: bold;">Dataset</span>', unsafe_allow_html=True)
data_dir = "./LLaMA-Factory/data"

# Upload and display data
upload_and_display_data(data_dir)
# Select and display dataset
select_and_display_dataset(data_dir)


### Fine-tuning parameters

# Global variable representing the currently selected language
current_language = "English"

# Function to retrieve text for the current language based on a key
def get_text(key):
    return language_texts[current_language].get(key, f"Missing text for key: {key}")

# Language options for the selection dropdown
language_options = ["English", "中文"]
selected_language = st.sidebar.selectbox("Select Language:", language_options)

# If the language has changed, update the global language variable
if selected_language != current_language:
    current_language = selected_language

# The following section pertains to Fine-tuning parameters in the sidebar
st.sidebar.markdown(f'<span style="color: #ff6347; font-size: 1.5em; font-weight: bold;">{get_text("sidebar_title")}</span>', unsafe_allow_html=True)

def show_info(info_text):
    st.info(info_text)

with st.sidebar.expander(get_text("sidebar_title"), expanded=False):
    dataset_dir = st.text_input(get_text("dataset_directory"), "./LLaMA-Factory/data")
    if st.button(get_text("info_btn_text"), key="dataset_info"):
        show_info(get_text("dataset_info"))

    dataset_name = st.text_input(get_text("dataset_name"), "lima")
    if st.button(get_text("info_btn_text"), key="dataset_name_info"):
        show_info(get_text("dataset_name_info"))

    model_name_or_path = st.text_input(get_text("model_name_or_path"), model_path)
    if st.button(get_text("info_btn_text"), key="model_name_or_path_info"):
        show_info(get_text("model_name_or_path_info"))

    output_dir = st.text_input(get_text("output_directory"), model_output_dir)
    if st.button(get_text("info_btn_text"), key="output_directory_info"):
        show_info(get_text("output_directory_info"))

    per_device_train_batch_size = st.slider(get_text("batch_size_per_device"), 1, 8, 2)
    if st.button(get_text("info_btn_text"), key="batch_size_info"):
        show_info(get_text("batch_size_info"))

    gradient_accumulation_steps = st.slider(get_text("accumulation_steps"), 1, 10, 2)
    if st.button(get_text("info_btn_text"), key="accumulation_steps_info"):
        show_info(get_text("accumulation_steps_info"))

    learning_rate_input = st.text_input(get_text("learning_rate"), value="5e-5")
    learning_rate = float(learning_rate_input) if learning_rate_input else 5e-5
    if st.button(get_text("info_btn_text"), key="learning_rate_info"):
        show_info(get_text("learning_rate_info"))

    num_train_epochs = st.slider(get_text("num_train_epochs"), 1, 10, 3)
    if st.button(get_text("info_btn_text"), key="epochs_info"):
        show_info(get_text("epochs_info"))

    lora_rank = st.slider(get_text("lora_rank"), 1, 20, 10)
    if st.button(get_text("info_btn_text"), key="lora_rank_info"):
        show_info(get_text("lora_rank_info"))

    lora_alpha = st.slider(get_text("lora_alpha"), 1, 50, 20)
    if st.button(get_text("info_btn_text"), key="lora_alpha_info"):
        show_info(get_text("lora_alpha_info"))

    eval_steps = st.number_input(get_text("eval_steps"), value=500)
    if st.button(get_text("info_btn_text"), key="eval_steps_info"):
        show_info(get_text("eval_steps_info"))

    logging_steps = st.number_input(get_text("logging_steps"), value=50)
    if st.button(get_text("info_btn_text"), key="logging_steps_info"):
        show_info(get_text("logging_steps_info"))

    save_steps = st.number_input(get_text("save_steps"), value=500)
    if st.button(get_text("info_btn_text"), key="save_steps_info"):
        show_info(get_text("save_steps_info"))

    fp16 = st.checkbox(get_text("enable_fp16"), value=True)
    if st.button(get_text("info_btn_text"), key="fp16_info"):
        show_info(get_text("fp16_info"))


###  Fine-tuning-scripts

st.title("Fine-tuning-scripts")
st.caption(f'Selected Train_scripts: :green[{model_name}]')
####  超参数优化
def read_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def objective(trial, data):
    total_samples = len(data)
    avg_text_length = sum(len(entry['input']) for entry in data) / total_samples

    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_int('batch_size', 1, 3)
    num_train_epochs = trial.suggest_int('num_train_epochs', 3, 5)

    print(f"Suggested Learning Rate: {learning_rate}")
    print(f"Suggested Batch Size: {batch_size}")
    print(f"Suggested Number of Training Epochs: {num_train_epochs}")

    # In a real scenario, you would run your model training and return an objective value (e.g., validation loss)
    # For simplicity, we return a dummy objective value here
    objective_value = (learning_rate * avg_text_length / batch_size) - num_train_epochs

    return objective_value
    
expander = st.expander("View first five data entries", expanded=False)

if st.sidebar.button("Save and Optimize"):
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")

    # 设置模型保存路径和数据集路径

    file_path = f"{dataset_dir}/{dataset_name}.json"
    dataset = read_dataset(file_path)

    # 创建一个Optuna Study对象，并使用优化目标函数优化超参数
    study = optuna.create_study(direction='minimize')
    objective_function = lambda trial: objective(trial, dataset)
    study.optimize(objective_function, n_trials=10)

    # 显示最佳超参数
    st.write("Best Hyperparameters:")
    st.write(f"Learning Rate: {study.best_params['learning_rate']}")
    st.write(f"Batch Size: {study.best_params['batch_size']}")
    st.write(f"Number of Training Epochs: {study.best_params['num_train_epochs']}")

    # Generate and display the training command with the optimized parameters
    t_command = f"""
CUDA_VISIBLE_DEVICES=0 python ./LLaMA-Factory/src/train_bash.py \\
    --stage sft \\
    --model_name_or_path {model_name_or_path} \\
    --do_train \\
    --dataset_dir {dataset_dir} \\
    --dataset {dataset_name} \\
    --template {template} \\
    --finetuning_type lora \\
    --lora_target {lora_target} \\
    --overwrite_cache \\
    --per_device_train_batch_size {study.best_params['batch_size']} \\
    --gradient_accumulation_steps {gradient_accumulation_steps} \\
    --eval_steps {eval_steps} \\
    --lr_scheduler_type cosine \\
    --logging_steps {logging_steps} \\
    --save_steps {save_steps} \\
    --overwrite_output_dir \\
    --output_dir {output_dir} \\
    --learning_rate {study.best_params['learning_rate']} \\
    --num_train_epochs {study.best_params['num_train_epochs']} \\
    --plot_loss \\
    --lora_rank {lora_rank} \\
    --lora_alpha {lora_alpha} \\
    --fp16
"""
    st.write("Generated Training Command:")
    st.code(t_command)

    # Write the training command to the specified script file
    with open(train_scripts, "w") as file:
        file.write(t_command)

    # Read the content of the script file
    with open(train_scripts, 'r', encoding='utf-8') as script_file:
        script_content = script_file.read()

        # Display the script content using an expander with syntax highlighting for shell script (language="sh")
        expander.code(script_content, language="sh")

    # Display a success message after successfully saving the parameters
    st.success("Parameters saved successfully!")



# Create global variables to store the process and output pipe
process = None
out_r, out_w = None, None

@st.cache_resource
def get_pipe():
    print('Creating pipe')
    global out_r, out_w
    out_r, out_w = os.pipe()
    return out_r, out_w

@st.cache_resource
def get_Popen(out_w,command):
    print('Creating process')
    global process
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=out_w, 
        stderr=out_w, 
        universal_newlines=False
    )
    return process

def start_fine_tuning(command):
    global process, out_r, out_w

    if process is None:
        out_r, out_w = get_pipe()
        process = get_Popen(out_w, command)

        st.markdown("## Output")
        output_container = st.empty()

        stop_button = st.button("Turn to the background")

        while True:
            if stop_button:
                process.terminate()
                st.warning("Process terminated.")
                process = None  # Reset process
                break

            raw_data = os.read(out_r, 1000)
            try:
                logs = raw_data.decode("utf-8", errors="ignore")
                output_container.text(logs)  # Use text instead of write
            except UnicodeDecodeError as e:
                print(f'UnicodeDecodeError: {e}')
            time.sleep(0.5)

    # Close the write end of the pipe
    os.close(out_w)
    os.close(out_r)
    
# Button to start fine-tuning
train_on = st.button("Start Fine-tuning", type="primary")
if train_on:
    train_command = f"sh {train_scripts}" 
    start_fine_tuning(train_command)
image_path = f"output_models/output_{template}/training_loss.png"

if os.path.exists(image_path):
    with st.expander("Training Loss Plot"):
        st.image(image_path, caption='Training Loss Plot', use_column_width=True)

if model_name == "Mistral-7B-v0.1":
    import plotly.express as px
    # 读取JSON Lines文件
    log_file_path = f"output_models/output_{template}/trainer_log.jsonl"

    log_data = []
    with open(log_file_path, 'r') as file:
        for line in file:
            log_data.append(json.loads(line))
    df = pd.DataFrame(log_data)
    if not df.empty:

        with st.expander("Training Log Plot"):
            # 绘制损失曲线和学习率曲线
            fig = px.line(df, x='current_steps', y=['loss', 'learning_rate'], labels={'value': 'Metric', 'variable': 'Metric Type'}, 
                          title='Training Loss and Learning Rate Over Steps')
            fig.update_layout(
                xaxis_title="Steps",
                yaxis_title="Value",
                legend_title="Metrics",
            )
            st.plotly_chart(fig)

#     import matplotlib.pyplot as plt

#     log_file_path = f"output_models/output_{template}/training_loss.png"

#     log_data = []
#     with open(log_file_path, 'r') as file:
#         for line in file:
#             log_data.append(json.loads(line))
#     df = pd.DataFrame(log_data)
#     if not df.empty:
#         with st.expander("Training Log Plot"):
#             fig, ax = plt.subplots()
#             ax.plot(df['current_steps'], df['loss'], label='Training Loss')
#             ax.set_xlabel('Steps')
#             ax.set_ylabel('Loss')
#             ax.legend()
#             st.pyplot(fig)



### Fine-tuned Merge  
st.title('Fine-tuned Merge')

st.write("Merge lora to fine-tune model weights")
merge_on = st.button("Merge", type="primary")
if merge_on:
    merge_process = None
    merge_out_r, merge_out_w = None, None

    @st.cache_resource
    def merge_get_pipe():
        print('Creating merge pipe')
        global merge_out_r, merge_out_w
        merge_out_r, merge_out_w = os.pipe()
        return merge_out_r, merge_out_w

    @st.cache_resource
    def merge_get_Popen(merge_out_w,merge_command):
        print('Creating merge process')
        global merge_process
        merge_process = subprocess.Popen(
            merge_command, 
            shell=True, 
            stdout=merge_out_w, 
            stderr=merge_out_w, 
            universal_newlines=False
        )
        return merge_process

    def start_fine_tuning(command):
        global merge_process, merge_out_r, merge_out_w

        if merge_process is None:
            merge_out_r, merge_out_w = get_pipe()
            merge_process = get_Popen(merge_out_w, merge_command)

            st.markdown("## Output")
            output_container = st.empty()

            stop_button = st.button("Turn to the background")

            while True:
                if stop_button:
                    merge_process.terminate()
                    st.warning("Process terminated.")
                    merge_process = None  # Reset process
                    break

                raw_data = os.read(out_r, 1000)
                try:
                    logs = raw_data.decode("utf-8", errors="ignore")
                    output_container.text(logs)  # Use text instead of write
                except UnicodeDecodeError as e:
                    print(f'UnicodeDecodeError: {e}')
                time.sleep(0.5)

        # Close the write end of the pipe
        os.close(merge_out_w)
        os.close(merge_out_r)
    merge_command = f"sh {merge_scripts}" 
    start_fine_tuning(merge_command)

    
### Fine-tuned Model Download
def get_zip_file_content(file_path):
    with open(file_path, "rb") as file:
        file_content = file.read()
    return file_content

st.title('Fine-tuned Model Download')
pack_on = st.button("Packing", type="primary")
if pack_on:
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    time = current_time
    # Create a temporary directory for storing the compressed file
    temp_dir = "temp" 
    os.makedirs(temp_dir, exist_ok=True)

    # Set the path for the compressed file
    zip_file_path = os.path.join(temp_dir, f"lora_fine-tuned_{model_name}.zip")

    # Use shutil.make_archive to create the compressed file
    shutil.make_archive(zip_file_path[:-4], 'zip', model_output_dir)

    # Get the content of the compressed file
    zip_file_content = get_zip_file_content(zip_file_path)

    # Provide a download button for the compressed file
    st.download_button(
        label="Download",
        type="primary",
        data=zip_file_content,
        file_name=f"lora_fine-tuned_{model_name}.zip",
        mime="application/zip"
    )


st.title("Trained Model Chat")
chat_on = st.toggle("chat with trained model")
if chat_on:
    @st.cache_resource
    def get_model(model_path):
        MODEL_PATH = os.environ.get('MODEL_PATH', model_path)
        TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
        model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()
        return tokenizer, model

    def chat_with_model(model_path):
        tokenizer, model = get_model(model_path)

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

    chat_with_model(model_path)


