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
        # 直接将音频数据写入内存
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_file.write(chunk["data"])
        
        audio_file.seek(0)

        end_time = time.time()
        print(f"EdgeTTS text_to_speech time: {end_time - start_time:.4f} seconds")
        # 生成一个虚拟的文件名，用于标识音频流
        virtual_filename = f"audio_{uuid4().hex[:8]}.mp3"
        return audio_file.read(), text, virtual_filename