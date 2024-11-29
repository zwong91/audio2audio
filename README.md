# rt-audio

Welcome to the rt-audio repository! This project hosts two exciting applications leveraging advanced audio understand and speech generation models to bring your audio experiences to life:

**Voice Chat** :  This application is designed to provide an interactive and natural chatting experience, making it easier to adopt sophisticated AI-driven dialogues in various settings.

For `SenseVoice`, visit [SenseVoice repo](https://github.com/FunAudioLLM/SenseVoice) and [SenseVoice space](https://www.modelscope.cn/studios/iic/SenseVoice).

## Install

**Clone and install**

- Clone the repo and submodules

``` sh

#0  source code

apt update
apt-get install vim  ffmpeg  git-lfs -y

git clone https://github.com/zwong91/rt-audio.git
cd /workspace/rt-audio
git pull

#1 pre_install.sh
# 安装 miniconda, PyTorch/CUDA 的 conda 环境
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash && source ~/miniconda3/bin/activate
conda config --set auto_activate_base false
conda create -n rt python=3.10  -y
conda activate rt

#2  rt-audio
cd /workspace/rt-audio
pip install -r requirements.txt


```

## Basic Usage

**prepare**

[openai](https://platform.openai.com/) api token.

[pem file](https://blog.csdn.net/liuchenbaidu/article/details/136722001)

``` sh
python3 -m src.app --vad-args '{"auth_token": "vad token here"}'
```
