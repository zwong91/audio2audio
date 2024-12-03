import os
import time
from uuid import uuid4
from typing import Tuple
import edge_tts
from .tts_interface import TTSInterface

class EdgeTTS(TTSInterface):
    #edge-tts --list-voices
    def __init__(self, voice: str = 'zh-CN-XiaoxiaoNeural'):
        self.voice = voice

    async def text_to_speech(self, text: str, rate: str = '50%', pitch: str = '-50Hz', volume: int = 70) -> Tuple[str, str]:
        start_time = time.time()
        """使用 edge_tts 库将文本转语音"""
        temp_file = f"/tmp/audio_{uuid4().hex[:8]}.mp3"
        communicate = edge_tts.Communicate(text=text, voice=self.voice)
        await communicate.save(temp_file)

        end_time = time.time()
        print(f"EdgeTTS text_to_speech time: {end_time - start_time:.4f} seconds")

        return os.path.basename(temp_file), text
