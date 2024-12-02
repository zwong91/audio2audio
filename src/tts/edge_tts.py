import os
import sys
from uuid import uuid4
from typing import Tuple
import edge_tts
from .tts_interface import TTSInterface

sys.path.insert(1, "../xtts")
from src.xtts.TTS.api import TTS

class EdgeTTS(TTSInterface):
    def __init__(self, voice: str = 'zh-CN-XiaoxiaoNeural'):
        self.voice = voice
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).to(device)

    async def text_to_speech(self, text: str) -> Tuple[str, str]:
        """使用 edge_tts 库将文本转语音"""
        temp_file = f"/tmp/audio_{uuid4()}.wav"
        communicate = edge_tts.Communicate(text=text, voice=self.voice)
        await communicate.save(temp_file)

        speech_file_path = f"/tmp/audio_{uuid4()}.wav"
        self.tts.voice_conversion_to_file(source_wav=temp_file, target_wav="../vc/liuyifei.wav", file_path=speech_file_path)
        return os.path.basename(speech_file_path), text
