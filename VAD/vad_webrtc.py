import webrtcvad

'''
WebRTC VAD 的音频帧长度:
WebRTC VAD 期望输入的音频帧长度为 10ms、20ms 或 30ms 的音频数据，
对于 16kHz 采样率，这对应于 160、320 或 480 个采样点。
'''
class WebRTCVAD:
    def __init__(self, sample_rate=16000, frame_size=320, bytes_per_sample=2, idle_time=0.5) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.bytes_per_sample = bytes_per_sample
        self.idle_cut = (self.sample_rate * idle_time) / self.frame_size  # chunk audio if no voice for idle_time seconds
        self.last_voice_activity = 0
        self.vad = webrtcvad.Vad(3)  # 使用 WebRTC VAD，敏感度设置为 3

    def convert_buffer_size(self, audio_frame):
        while len(audio_frame) < (self.frame_size * self.bytes_per_sample):
            audio_frame = audio_frame + b'\x00'
        return audio_frame
    
    def voice_activity_detection(self, audio_frame):
        converted_frame = self.convert_buffer_size(audio_frame)
        is_speech = self.vad.is_speech(converted_frame, sample_rate=self.sample_rate)
        if is_speech:
            self.last_voice_activity = 0
            return "1"
        else:
            if self.last_voice_activity >= self.idle_cut:
                self.last_voice_activity = 0
                return "X"
            else:
                self.last_voice_activity += 1
                return "_"