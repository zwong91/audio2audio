import time
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
        
        communicate = edge_tts.Communicate(text=text, voice=self.voice)
        audio_bytes = BytesIO()
        async for chunk in communicate.stream():
            audio_bytes.write(chunk)
        audio_bytes.seek(0)

        end_time = time.time()
        print(f"EdgeTTS text_to_speech time: {end_time - start_time:.4f} seconds")

        return audio_bytes.read(), text, "in_memory_audio.mp3"
