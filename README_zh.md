
# LinChance-Fine-tuning-System

<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/Joe-2002/LinChance-Fine-tuning-System/">
    <img src="images/home.PNG" alt="Home">
  </a>
  <p align="center">
    <br />
    <a href="https://github.com/Joe-2002/LinChance-Fine-tuning-System">查看Demo</a>
    ·
    <a href="https://github.com/Joe-2002/LinChance-Fine-tuning-System/issues">报告Bug</a>
    ·
    <a href="https://github.com/Joe-2002/LinChance-Fine-tuning-System/issues">提出新特性</a>
  </p>
</p>

## 目录

- [LinChance-Fine-tuning-System](#linchance-fine-tuning-system)
  - [目录](#目录)
    - [项目介绍](#项目介绍)
    - [环境准备](#环境准备)
    - [**安装步骤**](#安装步骤)
      - [1. 克隆仓库：](#1-克隆仓库)
      - [2. 执行下面的命令：](#2-执行下面的命令)
      - [3. 运行开启命令：](#3-运行开启命令)
      - [4. ngrok 内网穿透](#4-ngrok-内网穿透)
      - [4. WandB 微调输出信息联网绘制图表](#4-wandb-微调输出信息联网绘制图表)
      - [5. 数据集格式](#5-数据集格式)
    - [Web UI 页面操作](#web-ui-页面操作)
      - [点击一键下载模型](#点击一键下载模型)
      - [上传或者选择已有数据集](#上传或者选择已有数据集)
      - [使用默认脚本参数开始微调](#使用默认脚本参数开始微调)
      - [微调参数信息中英文切换](#微调参数信息中英文切换)
      - [超参数自动优化并显示](#超参数自动优化并显示)
      - [自定义并保存微调脚本参数查看并开始微调](#自定义并保存微调脚本参数查看并开始微调)
      - [微调进度重拾](#微调进度重拾)
      - [微调完成之后 loss 图显示](#微调完成之后-loss-图显示)
      - [与训练后模型对话测试](#与训练后模型对话测试)
      - [强制退出重启按钮](#强制退出重启按钮)
    - [已支持模型](#已支持模型)
    - [贡献者](#贡献者)
    - [版权说明](#版权说明)

### 项目介绍  

- 在 `Autodl 3090 24G` 实验环境下，采用 Streamlit 结合 LLaMA-Factory 打造的模型微调 Web UI ——LinChance Fine-tuning System。
- 使用 ngrok 内网穿透实现 Autodl 服务互联网访问。
- 使用 Streamlit 组件和方法实现简约大方的微调系统界面，使用 modelscope 方法实现模型快速下载，支持用户自定义微调参数，选择已有数据集或者上传私有数据集进行私有化便捷使用 Lora 方法微调大模型，利于新手友好操作。
- 使用 Linux 子进程方法实现多 Python 进程运行微调脚本和 Streamlit Web UI。  

### 环境准备

在 [autodl](https://www.autodl.com/) 平台中租一个 3090 等 24G 显存的显卡机器，如下图所示镜像选择 `PyTorch`-->`2.0.0`-->`3.8(ubuntu20.04)`-->`11.8`

![Alt text](images/autodl.png)

接下来打开刚刚租用服务器的 `JupyterLab`，并且打开其中的终端开始环境配置、模型下载和运行 `demo`。

pip 换源和安装依赖包

```shell
# 升级pip
python -m pip install --upgrade pip
# 更换 pypi 源加速库的安装
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope
```  

### **安装步骤**

#### 1. 克隆仓库：  

```sh
git clone https://github.com/Joe-2002/LinChance-Fine-tuning-System.git  
```  

#### 2. 执行下面的命令：

```shell
cd /root/LinChance Fine-tuning System
pip install -r requirements.txt
```  

#### 3. 运行开启命令：

```shell
python -m streamlit run main.py
```

#### 4. ngrok 内网穿透  
  
1. 首先在 `Ngrok` 官网上查看安装命令，我们以 `Linux` 系统为例，有多种方式可以安装，包括压缩包下载、`APT` 安装、`Snap` 安装，这里我们使用 `APT` 安装，执行以下命令：  

    ```shell
    curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list && sudo apt update && sudo apt install ngrok
    ```

2. `Ngrok` 安装完成后，需要到它的官网上注册一个账号，然后在 `Your Authtoken` 菜单中获取 `Authtoken`，这个 `Authtoken` 用于验证用户身份，可以通过以下命令将 `Authtoken` 设置到本地。
  
    ```shell
    ngrok config add-authtoken your-ngrok-authtoken # 这里替换成你的 Authtoken  
    ```  

3. 然后执行以下命令，通过 `Ngrok` 代理本地的  `Streamlit` 服务。
  
    ```shell
    ngrok http 8501 # streamlit 默认端口为 8501
    ```  

4. 访问下图链接即可打开 `Web UI` 页面：  

    ![Alt text](images/ngrok_link.png)

#### 4. WandB 微调输出信息联网绘制图表  

1. 安装 使用以下命令安装 WandB：

    ```bash
    pip install wandb
    ```

2. 安装完成后，你可以按照下面的链接访问 WandB 的快速入门指南： [WandB Quickstart](https://wandb.ai/quickstart)

3. 然后在 WandB 官网注册一个账号，然后获取该账号的私钥。然后在命令行执行：

      ```bash
      wandb login
      ```  

#### 5. 数据集格式  

*默认选择 lima 数据集，上传数据集后将名称改为 lima 即可（由于本仓库基于 LlaMa Factory 开发，修改后可以直接被识别使用）*

```json
[
  {
    "instruction": "",
    "input": "",
    "output": "",
    "history": ""
  },
  {
    "instruction": "",
    "input": "",
    "output": "",
    "history": ""
  },
]
```  

### Web UI 页面操作  

#### 点击一键下载模型  

![Alt text](images/model_download.PNG)  

#### 上传或者选择已有数据集  

![Alt text](images/datasets.PNG)  

#### 使用默认脚本参数开始微调  

![Alt text](images/finetuning.PNG)

#### 微调参数信息中英文切换

![Alt text](images/chinese&english.png)  

#### 超参数自动优化并显示

![Alt text](images/optimize.png)

#### 自定义并保存微调脚本参数查看并开始微调

![Alt text](images/finetuning_save.PNG)

#### 微调进度重拾

![Alt text](images/Recollecting.png)

#### 微调完成之后 loss 图显示

![Alt text](images/log.png)  

#### 与训练后模型对话测试  

![Alt text](images/chat.PNG)

#### 强制退出重启按钮  

> 使用前请先阅读警告信息，慎重使用！！！  

![Alt text](images/kill_all_processes.png)

### 已支持模型  

- [ChatGLM3](https://github.com/THUDM/ChatGLM3.git)
  - [x] ChatGLM3-6B-chat
- [Baichuan2](https://www.baichuan-ai.com/home)
  - [x] Baichuan2-7B-chat
  - [x] Baichuan2-13B-chat
- [Mistral](https://mistral.ai/news/announcing-mistral-7b/)
  - [x] Mistral-7B-v 0.1
- [Llama2](https://ai.meta.com/llama/)
  - [x] LlaMa2-7B-chat  
  
### 贡献者

[**李柯辰**](https://github.com/Joe-2002)

联系邮箱 : <a href="mailto:likechen@linchance.com">likechen@linchance.com</a>  

所属机构 : <a href="http://maxlikelihood.cn/">Likelihood Lab</a>

<div align="center">
  <img src="https://github.com/Joe-2002.png?s=40" alt="李柯辰" width="150" height="150">
  <br>
  <a href="https://github.com/Joe-2002"><b>李柯辰</b></a>
  <p></p>
</div>

### 版权说明