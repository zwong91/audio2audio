# funaudiollm-app repo

Welcome to the funaudiollm-app repository! This project hosts two exciting applications leveraging advanced audio understand and speech generation models to bring your audio experiences to life:

**Voice Chat** :  This application is designed to provide an interactive and natural chatting experience, making it easier to adopt sophisticated AI-driven dialogues in various settings.

**Voice Translation**: Break down language barriers with our real-time voice translation tool. This application seamlessly translates spoken language on the fly, allowing for effective and fluid communication between speakers of different languages.

For Details, visit [FunAudioLLM Homepage](https://fun-audio-llm.github.io/), [CosyVoice Paper](https://fun-audio-llm.github.io/pdf/CosyVoice_v1.pdf), [FunAudioLLM Technical Report](https://fun-audio-llm.github.io/pdf/FunAudioLLM.pdf)

For `CosyVoice`, visit [CosyVoice repo](https://github.com/FunAudioLLM/CosyVoice) and [CosyVoice space](https://www.modelscope.cn/studios/iic/CosyVoice-300M).

For `SenseVoice`, visit [SenseVoice repo](https://github.com/FunAudioLLM/SenseVoice) and [SenseVoice space](https://www.modelscope.cn/studios/iic/SenseVoice).

## Install

**Clone and install**

- Clone the repo and submodules

``` sh
git clone --recursive URL
# If you failed to clone submodule due to network failures, please run following command until success
cd rt-audio
git submodule update --init --recursive

#1 pre_install.sh
# 安装 miniconda, PyTorch/CUDA 的 conda 环境
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash && source ~/miniconda3/bin/activate
conda config --set auto_activate_base false


# Chatts 
cd /workspace/rt-audio/ChatTTS
conda create -n chatts python=3.10  -y
conda activate chatts
pip install -r requirements.txt 


#2 CosyVoice (Optional)
cd /workspace/rt-audio/cosyvoice
conda create -n cosyvoice python=3.8  -y
conda activate cosyvoice
# pynini is required by WeTextProcessing, use conda to install it as it can be executed on all platform.
conda install -y -c conda-forge pynini==2.1.5
#pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
pip install -r requirements.txt

apt update
# If you encounter sox compatibility issues
# ubuntu
apt-get install sox libsox-dev  ffmpeg  -y


#3 SenseVoice
cd /workspace/rt-audio/sensevoice
pip install -r requirements.txt


#4  rt-audio
cd /workspace/rt-audio
pip install -r requirements.txt

```

## Basic Usage

**prepare**

[openai](https://platform.openai.com/) api token.

[pem file](https://blog.csdn.net/liuchenbaidu/article/details/136722001)

**voice chat**

``` sh
cd voice_chat
OPENAI_API_KEY="YOUR-OPENAI-API-TOKEN" python app.py >> ./log.txt
```

<https://YOUR-IP-ADDRESS:5555/>

**voice translation**

``` sh
cd voice_translation
OPENAI_API_KEY="YOUR-OPENAI-API-TOKEN" python app.py >> ./log.txt
```

<https://YOUR-IP-ADDRESS:60002/>
