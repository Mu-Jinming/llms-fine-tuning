import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct")
out = pipe(messages)
print(out)