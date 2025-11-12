from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
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
