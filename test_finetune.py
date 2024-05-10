from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, AutoFeatureExtractor, set_seed
import torch
import soundfile as sf
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"

repo_id = "parler-tts/parler_tts_mini_v0.1"

model = ParlerTTSForConditionalGeneration.from_pretrained("/home/ys/project/parler-tts/output_dir_training", torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(repo_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)


prompt = """
Hi, I have a server with 8 RTX3090 GPUs, and I’m encountering a CUDA error exclusively when my code is executed on a particular GPU. Specifically, the issue arises only when I set CUDA_VISIBLE_DEVICES=0 . There are no such problems when I use CUDA_VISIBLE_DEVICES=1 , CUDA_VISIBLE_DEVICES=2 , etc.
Based on this, I suspect there might be a hardware issue with my first GPU (GPU 0). However, I’m finding it challenging to confirm this suspicion. Could you suggest any methods or steps to determine whether this is indeed a hardware-related issue?
"""
description = ""

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()

audio_array_int16 = (audio_arr * np.iinfo(np.int16).max).astype(np.int16)

SAMPLE_RATE = feature_extractor.sampling_rate
SEED = 41

file_path = "output.wav"
sf.write(file_path, audio_array_int16, SAMPLE_RATE)