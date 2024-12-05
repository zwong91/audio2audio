from .edge_tts import EdgeTTS
from .xtts import XTTS, XTTSv2
class TTSFactory:
    @staticmethod
    def create_tts_pipeline(asr_type, **kwargs):
        if asr_type == "edge":
            return EdgeTTS(**kwargs)
        elif asr_type == "xtts":
            return XTTS(**kwargs)
        elif asr_type == "xttsv2":
            return XTTSv2(**kwargs)
        else:
            raise ValueError(f"Unknown TTS pipeline type: {tts_type}")
