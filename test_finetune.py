from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, AutoFeatureExtractor, set_seed
import torch
import soundfile as sf
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"

repo_id = "parler-tts/parler_tts_mini_v0.1"
feature_extractor_repo_id = "ylacombe/dac_44khZ_8kbps"
tokenizer_repo_id = "google/flan-t5-base"

model = ParlerTTSForConditionalGeneration.from_pretrained("/home/ys/project/parler-tts/output_dir_training_id_male", torch_dtype=torch.float32).to(device)
tokenizer = AutoTokenizer.from_pretrained("/home/ys/project/parler-tts/output_dir_training_id_male")
feature_extractor = AutoFeatureExtractor.from_pretrained("/home/ys/project/parler-tts/output_dir_training_id_male")


prompt = "Sejak itu mereka hidup damai di Bulan."
description = "an indonesian male"

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids, do_sample=True, temperature=1.0)
audio_arr = generation.cpu().numpy().squeeze()

audio_array_int16 = (audio_arr * np.iinfo(np.int16).max).astype(np.int16)

SAMPLE_RATE = feature_extractor.sampling_rate
SEED = 41

file_path = "output.wav"
sf.write(file_path, audio_array_int16, SAMPLE_RATE)

# model.push_to_hub("parler-tts-mini-indo-male")
# tokenizer.push_to_hub("parler-tts-mini-indo-male")