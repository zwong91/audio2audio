from .edge_tts import EdgeTTS
from .xtts import XTTS_v2
class TTSFactory:
    @staticmethod
    def create_tts_pipeline(asr_type, **kwargs):
        if asr_type == "edge":
            return EdgeTTS(**kwargs)
        elif asr_type == "xtts-v2":
            return XTTS_v2(**kwargs)
        else:
            raise ValueError(f"Unknown TTS pipeline type: {tts_type}")
