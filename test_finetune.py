import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf

import wave

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained(
    "/home/ys/project/parler-tts/output_dir_training", torch_dtype=torch.float32
).to(device)

tokenizer = AutoTokenizer.from_pretrained("/home/ys/project/parler-tts/output_dir_training")
feature_extractor = AutoFeatureExtractor.from_pretrained("/home/ys/project/parler-tts/output_dir_training")

prompt = "Halo Pagi, aku Budi, dari Solo."
description = "Indonesian Male with clear voice"

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids, do_sample=True, temperature=1.0)
audio_arr = generation.cpu().numpy().squeeze()

#audio_array_int16 = np.frombuffer(audio_arr, dtype=np.int16)

audio_array_int16 = (audio_arr * np.iinfo(np.int32).max).astype(np.int16)

SAMPLE_RATE = feature_extractor.sampling_rate
SEED = 412

file_path = "output.wav"
sf.write(file_path, audio_array_int16, SAMPLE_RATE, subtype="PCM_24")

# model.push_to_hub("parler-tts-mini-indo")
# tokenizer.push_to_hub("parler-tts-mini-indo")
# feature_extractor.push_to_hub("parler-tts-mini-indo")
