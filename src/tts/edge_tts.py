import os
from uuid import uuid4
import edge_tts
from tts_interface import TTSInterface

class EdgeTTS(TTSInterface):
    def __init__(self, voice: str = 'zh-CN-XiaoxiaoNeural'):
        self.voice = voice

    async def text_to_speech(self, text: str) -> Tuple[str, str]:
        """使用 edge_tts 库将文本转语音"""
        speech_file_path = f"/tmp/audio_{uuid4()}.mp3"
        communicate = edge_tts.Communicate(text=text, voice=self.voice)
        await communicate.save(speech_file_path)
        return os.path.basename(speech_file_path), text