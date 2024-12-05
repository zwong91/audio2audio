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

sys.path.insert(1, "../vc")

from src.xtts.TTS.api import TTS

class XTTS(TTSInterface):
    def __init__(self, voice: str = 'zh-CN-XiaoxiaoNeural'):
        self.voice = voice
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).to(device)

    async def text_to_speech(self, text: str) -> Tuple[bytes, str, str]:
        audio_buffer = BytesIO()
        """使用 x_tts 库将文本转语音"""
        start_time = time.time()
        
        temp_file = f"/tmp/audio_{uuid4().hex[:8]}.mp3"
        communicate = edge_tts.Communicate(text=text, voice=self.voice)
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
        wav = await tts_task

        # 将生成的音频文件读入内存缓冲区
        audio_buffer = BytesIO()
        audio_buffer.write(wav)
        audio_buffer.seek(0)
        audio_data = audio_buffer.read()

        end_time = time.time()
        print(f"XTTSv2 text_to_speech time: {end_time - start_time:.4f} seconds")

        return audio_data, text, os.path.basename(speech_file_path)