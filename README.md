## 安装与使用指南

### 测试环境

本指南适用于以下测试环境：

- Python 3.10, PyTorch 2.3.1, CUDA 12.1
- Python 3.10, PyTorch 2.3.1, CUDA 11.8


### 1. 克隆代码仓库

```bash
# 克隆项目代码到本地
git clone https://github.com/xyzapphub/dubbing.git
```

### 2. 安装依赖环境

```bash
# 创建名为 'dubbing' 的conda环境，并指定Python版本为3.10
conda create -n dubbing python=3.10 -y

# 激活新创建的环境
conda activate dubbing

# 进入项目目录
cd dubbing/

# 安装ffmpeg工具
# 使用conda安装ffmpeg
conda install ffmpeg==7.0.2 -c conda-forge

# 升级pip到最新版本
python -m pip install --upgrade pip
```

根据您的CUDA版本，使用以下命令安装PyTorch及相关库：

```bash
# 对于CUDA 11.8
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# 对于CUDA 12.1
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```



然后，安装项目的其他依赖项：

```bash
pip install -r requirements.txt
# 安装submodules 下的依赖
pip install -r requirements_module.txt
```

> [!TIP]
>
> 如在安装过程中遇到错误提示“Could not load library libcudnn_ops_infer.so.8”，请按以下步骤修复：
>
> ```bash
> # 设置LD_LIBRARY_PATH以包含正确的cuDNN库路径
> export LD_LIBRARY_PATH=`python3 -c 'import os; import torch; print(os.path.dirname(os.path.dirname(torch.__file__)) +"/nvidia/cudnn/lib")'`:$LD_LIBRARY_PATH
> ```

### 3. 配置环境变量

在运行程序前，您需要配置必要的环境变量。请在项目根目录下的 `.env` 文件中添加以下内容，首先将 `env.example`填入以下环境变量并 改名为 `.env` ：

- `OPENAI_API_KEY`: 您的OpenAI API密钥，格式通常为 `sk-xxx`。
- `MODEL_NAME`: 使用的模型名称，如 `gpt-4` 或 `gpt-3.5-turbo`。
- `OPENAI_API_BASE`: 如使用自部署的OpenAI模型，请填写对应的API基础URL。
- `HF_TOKEN`: Hugging Face的API Token，用于访问和下载模型。
- `HF_ENDPOINT`: 当遇到模型下载问题时，可指定自定义的Hugging Face端点。

> [!NOTE]
>
> 通常，您只需配置 `MODEL_NAME` 和 `HF_TOKEN` 即可。
>
> 默认情况下，`MODEL_NAME` 设为 `Qwen/Qwen1.5-4B-Chat`，因此无需额外配置 `OPENAI_API_KEY`。

> ![TIP]
>
> 可以在 [Hugging Face](https://huggingface.co/settings/tokens) 获取 `HF_TOKEN`。若需使用**说话人分离功能**，务必在[pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)申请访问权限。否则，可以选择不启用该功能。

### 4. 运行程序

在启动程序前，先通过以下命令自动下载所需的模型（包括Qwen，XTTSv2，和faster-whisper-large-v3模型）：

```bash
# Linux 终端运行
bash scripts/download_models.sh

```


下载完成后，使用以下命令启动WebUI用户界面：

```bash
python webui.py
```

启动后，您将看到如下图所示的界面，可以打开 [http://127.0.0.1:6006](http://127.0.0.1:6006) 进行体验：

