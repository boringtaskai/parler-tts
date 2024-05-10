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
The database DATABASE_URL on Heroku app pontoon-sumbangsuara has exceeded its allocated storage capacity. Immediate action is required.
The database contains 10,708 rows, exceeding the Mini plan limit of 10,000. INSERT privileges to the database have been automatically revoked. This will cause service failures in most applications dependent on this database.
To enable access to your database, migrate the database to a Basic ($9/month) or higher database plan:
https://devcenter.heroku.com/articles/updating-heroku-postgres-databases
If you are unable to upgrade the database, you should reduce the number of records stored in it.
"""
description = "'Jenny's speech is very clear, and she speaks in a very monotone voice, really slowly and with minimal variation in speed.'"

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()

audio_array_int16 = (audio_arr * np.iinfo(np.int16).max).astype(np.int16)

SAMPLE_RATE = feature_extractor.sampling_rate
SEED = 41

file_path = "output.wav"
sf.write(file_path, audio_array_int16, SAMPLE_RATE)

# model.push_to_hub("parler-tts-mini-Jenny-test")
# tokenizer.push_to_hub("parler-tts-mini-Jenny-test")