from transformers import AutoModel, AutoTokenizer
from transformers import BitsAndBytesConfig

# Quantization config for 8-bit to use with not A100
# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     llm_int8_threshold=6.0  # optional, helps with stability
# )
def get_llada_model():
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    model = AutoModel.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct',
        #quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    return tokenizer, model