# rt-audio repo

Welcome to the rt-audio repository! This project hosts two exciting applications leveraging advanced audio understand and speech generation models to bring your audio experiences to life:

**Voice Chat** :  This application is designed to provide an interactive and natural chatting experience, making it easier to adopt sophisticated AI-driven dialogues in various settings.

**Voice Translation**: Break down language barriers with our real-time voice translation tool. This application seamlessly translates spoken language on the fly, allowing for effective and fluid communication between speakers of different languages.

For Details, visit [FunAudioLLM Homepage](https://fun-audio-llm.github.io/), [CosyVoice Paper](https://fun-audio-llm.github.io/pdf/CosyVoice_v1.pdf), [FunAudioLLM Technical Report](https://fun-audio-llm.github.io/pdf/FunAudioLLM.pdf)

For `CosyVoice`, visit [CosyVoice repo](https://github.com/FunAudioLLM/CosyVoice) and [CosyVoice space](https://www.modelscope.cn/studios/iic/CosyVoice-300M).

For `SenseVoice`, visit [SenseVoice repo](https://github.com/FunAudioLLM/SenseVoice) and [SenseVoice space](https://www.modelscope.cn/studios/iic/SenseVoice).

## Install

**Clone and install**

- Clone the repo and submodules

``` sh

#0  source code

apt update
# If you encounter sox compatibility issues
# ubuntu
apt-get install espeak-ng sox libsox-dev  ffmpeg  git-lfs -y

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


#2. Chatts 
cd /workspace/rt-audio/ChatTTS
conda create -n chatts python=3.10  -y
conda activate chatts
pip install -r requirements.txt 


#3 SenseVoice
cd /workspace/rt-audio/sensevoice
pip install -r requirements.txt

#4 XTTS
cd /workspace/rt-audio/XTTS_v2
pip install -e .[all]

#5  rt-audio
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
