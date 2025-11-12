import torch
import random 
from data import get_dataset
from evaluate import evaluate, evaluate_experimental
from gcg_single import gcg_single_attack_loss 
from gcg_qwen import qwen_gcg_single_attack 
from llada import get_llada_model
from qwen import get_qwen_model
from LLaDA_main import generate
from LLaDA_main import get_log_likelihood


prompts, targets = get_dataset()

llada_tokenizer, llada_model = get_llada_model()

qwen_tokenizer, qwen_model = get_qwen_model()
