import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import gc

def get_qwen_model():
    model_name_qwen = "Qwen/Qwen2.5-3B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_qwen = AutoModelForCausalLM.from_pretrained(
        model_name_qwen,
        torch_dtype="auto",
        device_map="auto"
    )
    model_qwen.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_qwen)

    return tokenizer, model_qwen

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
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        dtype="float16",
        device_map=device,
        trust_remote_code=True
    )
    return model, tokenizer


def load_qwen(device="cuda"):
    print("\n[Loading Qwen model...]")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
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