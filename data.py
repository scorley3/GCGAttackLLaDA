from dotenv import load_dotenv
import os
from huggingface_hub import login
from datasets import load_dataset


def get_dataset():
    load_dotenv()
    huggingface_token = os.getenv("HF_TOKEN")
    token = huggingface_token
    login(token)
    ds = load_dataset("walledai/AdvBench")
    return ds['train']['prompt'], ds['train']['target']
