import torch
import torchaudio
import asyncio
import os
from io import BytesIO
import sys
import time
from uuid import uuid4
from typing import Tuple
from .tts_interface import TTSInterface

import numpy as np

sys.path.insert(1, "../vc")

import edge_tts
from src.xtts.TTS.api import TTS
from src.xtts.TTS.tts.configs.xtts_config import XttsConfig    
from src.xtts.TTS.tts.models.xtts import Xtts

class XTTS(TTSInterface):
    def __init__(self, voice: str = 'zh-CN-XiaoxiaoNeural'):
        self.voice = voice
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).to(device)

    async def text_to_speech(self, text: str, rate: int = 0, pitch: int = 20, volume: int = 110) -> Tuple[bytes, str]:
        audio_buffer = BytesIO()
        """使用 x_tts 库将文本转语音"""
        start_time = time.time()

        temp_file = f"/tmp/audio_{uuid4().hex[:8]}.wav"
        rate_str = f"{rate:+d}%"
        pitch_str = f"{pitch:+d}Hz"
        volume_str = f"{volume:+d}%"
        communicate = edge_tts.Communicate(text=text, voice=self.voice, pitch=pitch_str, volume=volume_str)
        await communicate.save(temp_file)

        # 使用 os.path 确保路径正确拼接
        target_wav = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../rt-audio/vc")), "liuyifei.wav")
        speech_file_path = f"/tmp/audio_{uuid4().hex[:8]}.wav"

        tts_task = asyncio.create_task(
            asyncio.to_thread(
                self.tts.voice_conversion_to_file,
                source_wav=temp_file,
                target_wav=target_wav,
                file_path=speech_file_path
            )
        )
        await tts_task

        # 将生成的音频文件读入内存缓冲区
        with open(speech_file_path, 'rb') as f:
            audio_buffer.write(f.read())

        audio_buffer.seek(0)
        audio_data = audio_buffer.read()

        end_time = time.time()
        print(f"XTTS text_to_speech time: {end_time - start_time:.4f} seconds")

        return audio_data, text

class XTTS_v2(TTSInterface):
    def __init__(self, voice: str = 'liuyifei'):
        device = "cuda"
        # 使用 os.path 确保路径正确拼接
        target_wav = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../rt-audio/vc")), "liuyifei.wav")
        print("Loading model...")
        config = XttsConfig()
        config.load_json("XTTS-v2/config.json")
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir="XTTS-v2")#, use_deepspeed=True)
        self.model.to(device)

        print("Computing speaker latents...")
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=[target_wav])
        self.gpt_cond_latent = gpt_cond_latent
        self.speaker_embedding = speaker_embedding

    def wav_postprocess(self, wav):
        """Post process the output waveform"""
        if isinstance(wav, list):
            wav = torch.cat(wav, dim=0)
        wav = wav.clone().detach().cpu().numpy()
        wav = np.clip(wav, -1, 1)
        wav = (wav * 32767).astype(np.int16)
        return wav

    async def text_to_speech(self, text: str, language: str) -> Tuple[bytes, str]: 
        start_time = time.time()
        chunks = self.model.inference_stream(
            text,
            language,
            self.gpt_cond_latent,
            self.speaker_embedding,
            stream_chunk_size=1024,
        )

        for i, chunk in enumerate(chunks):
            print(type(chunk))
            processed_chunk = self.wav_postprocess(chunk)
            processed_bytes = processed_chunk.tobytes()
            yield processed_bytes
    
    async def text_to_speech_(self, text: str, language: str) -> Tuple[bytes, str]: 
        start_time = time.time()
        chunks = self.model.inference_stream(
            text,
            language,
            self.gpt_cond_latent,
            self.speaker_embedding,
            stream_chunk_size=1024,
        )
        wav_chunks = []
        for i, chunk in enumerate(chunks):
            wav_chunks.append(chunk)
        wav = torch.cat(wav_chunks, dim=0)
        wav_audio = wav.squeeze().unsqueeze(0).cpu()

        # Saving to a file on disk
        #file_path = f"/tmp/audio_{uuid4().hex[:8]}.wav"
        #torchaudio.save(file_path, wav_audio, 22050, format="wav")

        # Saving to a temporary file or directly converting to a byte array
        with torch.no_grad():
            # Use torchaudio to save the tensor to a buffer (or file)
            # Using a buffer to save the audio data as bytes
            buffer = BytesIO()
            torchaudio.save(buffer, wav_audio, 22050, format="wav")  # Adjust sample rate if needed
            audio_data = buffer.getvalue()

        end_time = time.time()
        print(f"XTTSv2 text_to_speech time: {end_time - start_time:.4f} seconds")
        return audio_data, text
