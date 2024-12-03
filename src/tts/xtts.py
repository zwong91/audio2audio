import torch
import os
import sys
from uuid import uuid4
from typing import Tuple
from .tts_interface import TTSInterface

sys.path.insert(1, "../vc")

from src.xtts.TTS.api import TTS

class XTTS(TTSInterface):
    def __init__(self, voice: str = ''):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    async def text_to_speech(self, text: str) -> Tuple[str, str]:
      
        # 使用 os.path 确保路径正确拼接
        target_wav = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../rt-audio/vc")), "liuyifei.wav")
        speech_file_path = f"/tmp/audio_{uuid4()}.wav"
        # 调用语音转换方法
        self.tts.tts_with_vc_to_file(
            text,
            speaker_wav=target_wav,
            language="zh-CN", 
            file_path=speech_file_path
        )
        return os.path.basename(speech_file_path), text
