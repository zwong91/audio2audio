import os
import sys
from uuid import uuid4
from typing import Tuple
import edge_tts
from .tts_interface import TTSInterface

class EdgeTTS(TTSInterface):
    def __init__(self, voice: str = 'zh-CN-XiaoxiaoNeural'):
        self.voice = voice

    async def list_voices():
        """列出所有可用的语音"""
        voices = await edge_tts.list_voices()
        for voice in voices:
            print(f"Voice Name: {voice['Name']}, Gender: {voice['Gender']}, Language: {voice['Locale']}")

    async def text_to_speech(self, text: str, rate: str = '50%', pitch: str = '10%', volume: int = 70) -> Tuple[str, str]:
        """使用 edge_tts 库将文本转语音，并设置语速、音调和音量"""
        temp_file = f"/tmp/audio_{uuid4().hex[:8]}.mp3"
        communicate = edge_tts.Communicate(
            text=text, 
            voice=self.voice,
            rate=rate,
            pitch=pitch,
            volume=volume
        )
        await communicate.save(temp_file)

        return os.path.basename(temp_file), text
