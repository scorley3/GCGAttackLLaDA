from transformers import AutoModelForCausalLM, AutoTokenizer

def get_qwen_model():
    model_name_qwen = "Qwen/Qwen2.5-3B-Instruct"

    model_qwen = AutoModelForCausalLM.from_pretrained(
        model_name_qwen,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_qwen)

    return tokenizer, model_qwen
