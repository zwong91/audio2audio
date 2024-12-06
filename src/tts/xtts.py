import torch
import asyncio
import os
from io import BytesIO
import sys
import time
from uuid import uuid4
from typing import Tuple
import edge_tts
from .tts_interface import TTSInterface

import numpy as np
from scipy.io.wavfile import write

sys.path.insert(1, "../vc")

from src.xtts.TTS.api import TTS

class XTTS(TTSInterface):
    def __init__(self, voice: str = 'zh-CN-XiaoxiaoNeural'):
        self.voice = voice
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).to(device)

    async def text_to_speech(self, text: str, rate: int = 0, pitch: int = 20, volume: int = 110) -> Tuple[bytes, str, str]:
        audio_buffer = BytesIO()
        """使用 x_tts 库将文本转语音"""
        start_time = time.time()
        
        temp_file = f"/tmp/audio_{uuid4().hex[:8]}.mp3"
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

        return audio_data, text, os.path.basename(speech_file_path)
    

class XTTSv2(TTSInterface):
    def __init__(self, voice: str = ''):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    async def text_to_speech(self, text: str) -> Tuple[bytes, str, str]:
        start_time = time.time()

        # 使用 os.path 确保路径正确拼接
        target_wav = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../rt-audio/vc")), "liuyifei.wav")
        speech_file_path = f"/tmp/audio_{uuid4().hex[:8]}.wav"

        # 生成音频文件
        tts_task = asyncio.create_task(
            asyncio.to_thread(
                self.tts.tts,
                text=text,
                speaker_wav=target_wav,
                language="en"
            )
        )
        wav_data = await tts_task

        # 创建 BytesIO 对象
        wav_io = BytesIO()

        # wav_data[0] 是音频数据，wav_data[1] 是采样率
        wav_audio = wav_data[0]
        sample_rate = wav_data[1] if len(wav_data) > 1 else 16000  # 默认采样率16000

        # 如果返回的是 NumPy 数组，则写入 WAV 文件格式
        if isinstance(wav_audio, np.ndarray):
            write(wav_io, sample_rate, wav_audio)
        # 如果返回的是字节流，直接写入
        elif isinstance(wav_audio, (bytes, bytearray)):
            wav_io.write(wav_audio)
        
        # 将游标移到开始位置
        wav_io.seek(0)
        audio_data = wav_io.read()
        end_time = time.time()
        print(f"XTTSv2 text_to_speech time: {end_time - start_time:.4f} seconds")

        return audio_data, text, os.path.basename(speech_file_path)