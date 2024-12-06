import torch
import asyncio
import os
from io import BytesIO
import sys
import time
from uuid import uuid4
from typing import Tuple
from .tts_interface import TTSInterface

import numpy as np
from scipy.io.wavfile import write

sys.path.insert(1, "../vc")

from src.xtts.TTS.api import TTS
from src.xtts.TTS.tts.configs.xtts_config import XttsConfig    
from src.xtts.TTS.tts.models.xtts import Xtts

class XTTS(TTSInterface):
    def __init__(self, voice: str = 'liuyifei'):
        # 使用 os.path 确保路径正确拼接
        target_wav = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../rt-audio/vc")), "liuyifei.wav")
        print("Loading model...")    
        config = XttsConfig()   
        config.load_json("tts_models/multilingual/multi-dataset/xtts_v2/config.json") 
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir="tts_models/multilingual/multi-dataset/xtts_v2", use_deepspeed=True)
        model.cuda()
        self.model = model
        print("Computing speaker latents...")
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[target_wav])
        self.gpt_cond_latent = gpt_cond_latent
        self.speaker_embedding = speaker_embedding

    async def text_to_speech(self, text: str) -> Tuple[bytes, str, str]:
        audio_buffer = BytesIO()
        chunks = self.model.inference_stream(
            text,
            "zh-cn",  
            self.gpt_cond_latent,    
            self.speaker_embedding
        )   
        wav_chuncks = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                print(f"Time to first chunck: {time.time() - t0}")
            print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
            wav_chuncks.append(chunk)
        wav = torch.cat(wav_chuncks, dim=0)
        # 创建 BytesIO 对象
        wav_io = BytesIO()
        wav_audio = wav.squeeze().unsqueeze(0).cpu()
        sample_rate = 24000  # 默认采样率24k
        write(wav_io, sample_rate, wav_audio) 
        wav_io.seek(0)
        audio_data = wav_io.read()
        end_time = time.time()
        print(f"XTTSv2 text_to_speech time: {end_time - start_time:.4f} seconds")
        return audio_data, text, os.path.basename(speech_file_path)
