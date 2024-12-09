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

import edge_tts
from src.xtts.TTS.api import TTS
from src.xtts.TTS.tts.configs.xtts_config import XttsConfig    
from src.xtts.TTS.tts.models.xtts import Xtts

class XTTS(TTSInterface):
    def __init__(self, voice: str = 'zh-CN-XiaoxiaoNeural'):
        self.voice = voice
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).to(device)

    async def text_to_speech(self, text: str, rate: int = 0, pitch: int = 20, volume: int = 110) -> Tuple[bytes, str, str]:
        audio_buffer = BytesIO()
        """使用 x_tts 库将文本转语音"""
        start_time = time.time()

        temp_file = f"/tmp/audio_{uuid4().hex[:8]}.wav"
        rate_str = f"{rate:+d}%"
        pitch_str = f"{pitch:+d}Hz"
        volume_str = f"{volume:+d}%"
        communicate = edge_tts.Communicate(text=text, voice=self.voice, pitch=pitch_str, volume=volume_str)
        await communicate.save(temp_file)

        # 使用 os.path 确保路径正确拼接
        target_wav = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../rt-audio/vc")), "liuyifei.wav")
        speech_file_path = f"/tmp/audio_{uuid4().hex[:8]}.wav"

        tts_task = asyncio.create_task(
            asyncio.to_thread(
                self.tts.voice_conversion_to_file,
                source_wav=temp_file,
                target_wav=target_wav,
                file_path=speech_file_path
            )
        )
        await tts_task

        # 将生成的音频文件读入内存缓冲区
        with open(speech_file_path, 'rb') as f:
            audio_buffer.write(f.read())

        audio_buffer.seek(0)
        audio_data = audio_buffer.read()

        end_time = time.time()
        print(f"XTTS text_to_speech time: {end_time - start_time:.4f} seconds")

        return audio_data, text

class XTTS_v2(TTSInterface):
    def __init__(self, voice: str = 'liuyifei'):
        # 使用 os.path 确保路径正确拼接
        target_wav = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../rt-audio/vc")), "liuyifei.wav")
        print("Loading model...")
        config = XttsConfig()
        config.load_json("./config.json")
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir="XTTS-v2", use_deepspeed=True)
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
