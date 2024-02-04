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
import shutil
from threading import Thread
from streamlit_chat import message
from datetime import datetime
import optuna
import psutil
from functions.finetuning import determine_batch_size,read_dataset,determine_learning_rate,determine_num_train_epochs

from functions.start_function import start
from functions.data_utils import upload_and_display_data, select_and_display_dataset
from functions.finetuning_parameters import language_texts

st.set_page_config(
    page_title="Home",
    page_icon="👋",
)

# global
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

# models
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
    
expander = st.expander("View first five data entries", expanded=False)
if st.sidebar.button("Save"):
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
    --per_device_train_batch_size {per_device_train_batch_size} \\
    --gradient_accumulation_steps {gradient_accumulation_steps} \\
    --eval_steps {eval_steps} \\
    --lr_scheduler_type cosine \\
    --logging_steps {logging_steps} \\
    --save_steps {save_steps} \\
    --overwrite_output_dir \\
    --output_dir {output_dir} \\
    --learning_rate {learning_rate} \\
    --num_train_epochs {num_train_epochs} \\
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


Optimize_expander = st.expander("View first five data entries", expanded=False)
if st.sidebar.button("Optimize"):
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")

    file_path = f"{dataset_dir}/{dataset_name}.json"
    data = read_dataset(file_path)
    num_samples = len(data)
    Optimize_learning_rate = determine_learning_rate(num_samples)
    st.sidebar.text(f"learning rate: {Optimize_learning_rate}")
    Optimize_batch_size = determine_batch_size(model_name)
    st.sidebar.text(f"batch_size for {model_name}: {Optimize_batch_size}")
    Optimize_num_train_epochs = determine_num_train_epochs(num_samples)
    st.sidebar.text(f"num_train_epochs: {Optimize_num_train_epochs}")
    Optimize_on = st.sidebar.button("Optimize and Save")
    if Optimize_on:
        try:
            Optimize_command = f"""
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
                --per_device_train_batch_size {Optimize_batch_size} \\
                --gradient_accumulation_steps {gradient_accumulation_steps} \\
                --eval_steps {eval_steps} \\
                --lr_scheduler_type cosine \\
                --logging_steps {logging_steps} \\
                --save_steps {save_steps} \\
                --overwrite_output_dir \\
                --output_dir {output_dir} \\
                --learning_rate {Optimize_learning_rate} \\
                --num_train_epochs {Optimize_num_train_epochs} \\
                --plot_loss \\
                --lora_rank {lora_rank} \\
                --lora_alpha {lora_alpha} \\
                --fp16
            """
            st.write("Optimize Parameters Generated Training Command:")
            st.code(Optimize_command)
            
            # Write the training command to the specified script file
            with open(train_scripts, "w") as file:
                file.write(Optimize_command)
            
            # Read the content of the script file
            with open(train_scripts, 'r', encoding='utf-8') as script_file:
                script_content = script_file.read()
                
                # Display the script content using an expander with syntax highlighting for shell script (language="sh")
                Optimize_expander.code(script_content, language="sh")
            
            # Display a success message after successfully saving the parameters
            st.success("Optimize Parameters saved successfully!")
            
        except Exception as e:
            # Display an error message if an exception occurs during the process
            st.error(f"Error occurred: {str(e)}")

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

log_on = st.button("View")
if log_on:
    import plotly.express as px
    import plotly.graph_objects as go
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
            fig = go.Figure()

            # 损失曲线
            fig.add_trace(go.Scatter(x=df['current_steps'], y=df['loss'], mode='lines', name='Loss', line=dict(color='blue')))

            # 添加目标损失值横线
            target_loss = 1.0  # 根据实际情况调整目标损失值
            fig.add_shape(go.layout.Shape(type="line", x0=df['current_steps'].min(), x1=df['current_steps'].max(), y0=target_loss, y1=target_loss, line=dict(color="red", width=2, dash="dash")))

            # 学习率曲线
            fig.add_trace(go.Scatter(x=df['current_steps'], y=df['learning_rate'], mode='lines', name='Learning Rate', line=dict(color='green')))

            # 调整布局和标签
            fig.update_layout(
                title='Training Loss and Learning Rate Over Steps',
                xaxis_title="Steps",
                yaxis_title="Value",
                legend_title="Metrics",
                showlegend=True,
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True),
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


def kill_python_processes():
    try:
        # Run the script to kill Python processes
        subprocess.run(["sh functions/kill_python.sh"], check=True, shell=True)
        st.sidebar.success("Python processes terminated successfully.")
    except subprocess.CalledProcessError as e:
        st.sidebar.error(f"Error terminating Python processes: {e}")

def display_confirmation_warning():
    st.sidebar.warning(
        "⚠️ **WARNING:** This action will terminate all Python processes associated with NVIDIA devices and Streamlit.\n"
        "After clicking the button, you'll need to restart the Streamlit server by running `streamlit run main.py` in the terminal."
    )

st.sidebar.title("Terminate All Processes ⚠️")
display_confirmation_warning()
button_clicked = st.sidebar.button("Terminate All Processes", type="primary")
if button_clicked:
    kill_python_processes()

### Fine-tuned Merge  
st.title('Fine-tuned Merge')

st.write("Merge lora to fine-tune model weights")

merge_on = st.button("Merge", type="primary")
if merge_on:
    # Set up progress bar
    progress_bar = st.progress(0)
    
    # Area to display command line output
    console_output = st.empty()

    # Print script path
    st.text(f"Running script: {merge_scripts}")
    
    # Create StringIO object to capture tqdm output
    tqdm_output = StringIO()

    # Use subprocess.Popen to start the shell script
    process = subprocess.Popen(f"sh {merge_scripts}", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)

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

from transformers.generation.utils import GenerationConfig
model_merge_path = "/root/LinChance-Fine-tuning-System/merge/merge_model/merge_baichuan7b"
@st.cache_resource
def init_model():
    model = AutoModelForCausalLM.from_pretrained(
        model_merge_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.generation_config = GenerationConfig.from_pretrained(
        model_merge_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_merge_path,
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer

def clear_chat_history():
    del st.session_state.messages

def init_chat_history():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("您好，我是百川大模型，很高兴为您服务🥰")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = '🧑‍💻' if message["role"] == "user" else '🤖'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages

def chat_with_baichuan():
    model, tokenizer = init_model()
    messages = init_chat_history()

    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar='🧑‍💻'):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})
        print(f"[user] {prompt}", flush=True)
        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()
            for response in model.chat(tokenizer, messages, stream=True):
                placeholder.markdown(response)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        messages.append({"role": "assistant", "content": response})
        print(json.dumps(messages, ensure_ascii=False), flush=True)

        st.button("清空对话", on_click=clear_chat_history)

st.title("Trained Model Chat")
model_path_merge = "model_path_merge"
# model_path_1 = st.selectbox("Select Model 1", ["Baichuan2-7B-Chat", "Other Model 1"])
chat_on = st.toggle("chat with trained model")
if chat_on:
    if model_name == "Baichuan2-7B-Chat":
        chat_with_baichuan()
    else:
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


