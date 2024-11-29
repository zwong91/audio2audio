from .edge_tts import EdgeTTS

class TTSFactory:
    @staticmethod
    def create_tts_pipeline(asr_type, **kwargs):
        if asr_type == "edge":
            return EdgeTTS(**kwargs)
        else:
            raise ValueError(f"Unknown TTS pipeline type: {tts_type}")
