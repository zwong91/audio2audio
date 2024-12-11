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

from langdetect import detect
import glob

sys.path.insert(1, "../vc")

from src.xtts.TTS.api import TTS
from src.xtts.TTS.tts.configs.xtts_config import XttsConfig    
from src.xtts.TTS.tts.models.xtts import Xtts

from src.xtts.TTS.utils.generic_utils import get_user_data_dir
from src.xtts.TTS.utils.manage import ModelManager

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
        
        # model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        # logging.info("⏳Downloading model")
        # ModelManager().download_model(model_name)
        # model_path = os.path.join(
        #     get_user_data_dir("tts"), model_name.replace("/", "--")
        # )

        # config = XttsConfig()
        # config.load_json(os.path.join(model_path, "config.json"))
        # self.model = Xtts.init_from_config(config)
        # self.model.load_checkpoint(config, checkpoint_dir=model_path, eval=True)
        # self.model.to(device)

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

    async def text_to_speech(self, text: str, vc_uid: str, gen_file: bool) -> Tuple[bytes, str]: 
        start_time = time.time()
        language = detect(text)

        # 构造目标路径，获取匹配的 .wav 文件
        target_wav_pattern = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../rt-audio/vc")), f"{vc_uid}*.wav")
        target_wav_files = glob.glob(target_wav_pattern)  # 使用 glob 扩展通配符

        if not target_wav_files:
            target_wav = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../rt-audio/vc")), "liuyifei.wav")
            print(f"No WAV files found matching pattern, use default: {target_wav_pattern}")

        print("Computing speaker latents...")

        # 调用模型函数，传递匹配的文件列表
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=target_wav_files)
        print("Target WAV files:", target_wav_files)

        chunks = self.model.inference_stream(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            # Streaming
            stream_chunk_size=4096,
            overlap_wav_len=1024,
            # GPT inference
            temperature=0.7,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=50,
            top_p=0.85,
            do_sample=True,
            speed=1.2,
            enable_text_splitting=False,
        )
        wav_chunks = []
        output_path = f"/asset/audio_{uuid4().hex[:8]}.wav"
        for i, chunk in enumerate(chunks):
            wav_chunks.append(chunk)
            #processed_chunk = self.wav_postprocess(chunk)
            #processed_bytes = processed_chunk.tobytes()
            #yield processed_bytes, output_path
  
        wav = torch.cat(wav_chunks, dim=0)
        wav_audio = wav.squeeze().unsqueeze(0).cpu()

        # Saving to a file on disk
        if gen_file:
            torchaudio.save(output_path, wav_audio, 22050, format="wav")

        # Saving to a temporary file or directly converting to a byte array
        with torch.no_grad():
            # Use torchaudio to save the tensor to a buffer (or file)
            # Using a buffer to save the audio data as bytes
            buffer = BytesIO()
            torchaudio.save(buffer, wav_audio, 22050, format="wav")  # Adjust sample rate if needed
            audio_data = buffer.getvalue()

        end_time = time.time()
        print(f"XTTSv2 text_to_speech time: {end_time - start_time:.4f} seconds")
        return audio_data, output_path

