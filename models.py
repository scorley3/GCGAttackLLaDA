import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import gc



def load_llada_mini(device="cuda"):
    print("\n[Loading LLADA Mini model...]")
    tokenizer = AutoTokenizer.from_pretrained("inclusionAI/LLaDA2.0-mini-preview", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("inclusionAI/LLaDA2.0-mini-preview", trust_remote_code=True, dtype="float16")
    return model, tokenizer

# Quantization config for 8-bit to use with not A100
# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     llm_int8_threshold=6.0  # optional, helps with stability
# )

def load_llada_base(device="cuda"):
    print("\n[Loading LLADA Base model...]")
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        trust_remote_code=True
    )
    print("[LLADA Base] Tokenizer loaded.")
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        dtype="float16",
        device_map=device,
        trust_remote_code=True
    )
    print("[LLADA Base] Model loaded onto device.")
    return model, tokenizer


def load_qwen(device="cuda"):
    print("\n[Loading Qwen model...]")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B",
        dtype="float16",
        device_map=device,
    )
    return model, tokenizer


def load_llada(device="cuda"):
    print("\n[Loading LLADA model...]")
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        dtype="float16",
        device_map=device,
        trust_remote_code=True
    )
    return model, tokenizer

def unload_model(model):
    print("[Unloading model from GPU to free VRAM...]")
    del model
    gc.collect()
    torch.cuda.empty_cache()