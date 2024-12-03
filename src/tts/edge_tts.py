import os
import sys
from uuid import uuid4
from typing import Tuple
import edge_tts
from .tts_interface import TTSInterface

class EdgeTTS(TTSInterface):
    def __init__(self, voice: str = 'zh-CN-XiaoxiaoNeural'):
        self.voice = voice

    async def text_to_speech(self, text: str) -> Tuple[str, str]:
        """使用 edge_tts 库将文本转语音"""
        temp_file = f"/tmp/audio_{uuid4()}.mp3"
        communicate = edge_tts.Communicate(text=text, voice=self.voice)
        await communicate.save(temp_file)

        return os.path.basename(temp_file), text
