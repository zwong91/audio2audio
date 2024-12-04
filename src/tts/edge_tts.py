import time
from uuid import uuid4
from typing import Tuple
import edge_tts
from io import BytesIO
from .tts_interface import TTSInterface

class EdgeTTS(TTSInterface):
    def __init__(self, voice: str = 'zh-CN-XiaoxiaoNeural'):
        self.voice = voice

    async def text_to_speech(self, text: str, rate: int = 50, pitch: int = -50, volume: int = 70) -> Tuple[bytes, str, str]:
        start_time = time.time()
        
        """使用 edge_tts 库将文本转语音"""
        audio_file = BytesIO()

        rate_str = f"{rate:+d}%"
        pitch_str = f"{pitch:+d}Hz"

        # 初始化 Communicate 对象，设置语音参数
        communicate = edge_tts.Communicate(text=text, voice=self.voice, rate=rate, pitch=pitch, volume=volume)
        
        # 异步读取音频数据并将其写入 BytesIO 流
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_file.write(chunk["data"])
        
        # 将 BytesIO 文件指针重置到开头，以便后续读取
        audio_file.seek(0)
        
        end_time = time.time()
        print(f"EdgeTTS text_to_speech time: {end_time - start_time:.4f} seconds")
        
        # 生成一个虚拟的文件名，用于标识音频流
        virtual_filename = f"audio_{uuid4().hex[:8]}.mp3"
        
        # 返回音频数据的字节流和相关信息
        return audio_file.read(), text, virtual_filename
