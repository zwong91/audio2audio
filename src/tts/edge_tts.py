import time
from uuid import uuid4
from typing import Tuple
import edge_tts
from io import BytesIO
from .tts_interface import TTSInterface

class EdgeTTS(TTSInterface):
    def __init__(self, voice: str = 'zh-CN-XiaoxiaoNeural'):
        self.voice = voice

    async def text_to_speech(self, text: str, rate: int = 0, pitch: int = 10, volume: int = 90) -> Tuple[bytes, str, str]:
        start_time = time.time()

        """使用 edge_tts 库将文本转语音"""
        
        # 直接初始化内存缓冲区（BytesIO）
        audio_buffer = BytesIO()

        rate_str = f"{rate:+d}%"
        pitch_str = f"{pitch:+d}Hz"
        volume_str = f"{volume:+d}%"
        try:
            # 初始化 Communicate 对象，设置语音、语速、音调和音量参数
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.voice,
                rate=rate_str,
                pitch=pitch_str,
                volume=volume_str
            )

            # 将音频数据流直接写入 BytesIO 缓冲区
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    # 根据edge_tts的文档，chunk["data"] 是音频流的字节数据
                    audio_buffer.write(chunk["data"])

            # 重置 BytesIO 文件指针，以便后续读取
            audio_buffer.seek(0)

            end_time = time.time()
            print(f"EdgeTTS text_to_speech time: {end_time - start_time:.4f} seconds")

            # 生成一个虚拟的文件名，用于标识音频流
            virtual_filename = f"audio_{uuid4().hex[:8]}.mp3"
            audio_data = audio_buffer.getvalue()
            print(f"音频数据大小: {len(audio_data)} 字节")
            # 返回音频数据的字节流、原始文本和生成的虚拟文件名
            return audio_data, text, virtual_filename
        finally:
            audio_buffer.close()
