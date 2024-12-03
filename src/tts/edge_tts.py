import os
import time
from uuid import uuid4
from typing import Tuple
import edge_tts
from io import BytesIO
from .tts_interface import TTSInterface

class EdgeTTS(TTSInterface):
    def __init__(self, voice: str = 'zh-CN-XiaoxiaoNeural'):
        self.voice = voice

    async def text_to_speech(self, text: str, rate: str = '50%', pitch: str = '-50Hz', volume: int = 70) -> Tuple[bytes, str, str]:
        start_time = time.time()
        """使用 edge_tts 库将文本转语音"""
        temp_file = f"/tmp/audio_{uuid4().hex[:8]}.mp3"
        communicate = edge_tts.Communicate(text=text, voice=self.voice)
        await communicate.save(temp_file)

        # 将音频文件保存到内存缓冲区
        audio_file = BytesIO()
        with open(temp_file, 'rb') as f:
            audio_file.write(f.read())
        audio_file.seek(0)

        end_time = time.time()
        print(f"EdgeTTS text_to_speech time: {end_time - start_time:.4f} seconds")

        return audio_file.read(), text, os.path.basename(temp_file)