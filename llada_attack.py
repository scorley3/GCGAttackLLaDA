import torch, gc
import torch.nn.functional as F
from evaluate import evaluate
from data import get_dataset
import argparse
import json
from pathlib import Path
from typing import List, Any

def main(prompts_start=0, prompts_end=50, iterations=40, k=256, batch_size=128,use_qwen=True, suffix_len=20, seed_llada=True): #prompts, targets, iters, k, batch_size, use_qwen=True, device="cuda", suffix_len=30
    mask_id = 126336
    gc.collect()
    torch.cuda.empty_cache()
    prompts, targets = get_dataset()

    evaluate(prompts[prompts_start:prompts_end], targets[prompts_start:prompts_end], iterations, k, batch_size=batch_size, use_qwen=use_qwen,seed_llada=seed_llada, device="cuda", suffix_len=suffix_len)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run GCG adversarial attacks on LLADA model."
    )

    parser.add_argument("--prompts_start", type=int, default=0, help="Start index for dataset prompts")
    parser.add_argument("--prompts_end", type=int, default=20, help="End index for dataset prompts")
    parser.add_argument("--iterations", type=int, default=5, help="Number of attack iterations")
    parser.add_argument("--k", type=int, default=256, help="Top-k tokens considered for substitution")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for random substitutions")
    parser.add_argument("--use_qwen", type=bool, default=False, help="Whether to attack Qwen model first")
    parser.add_argument("--suffix_len", type=int, default=20, help="Length of suffix to optimize")
    parser.add_argument("--seed_llada", type=bool, default=True, help="Whether to seed LLADA with Qwen adversarial prompt")

    args = parser.parse_args()
    
    main(
        prompts_start=args.prompts_start,
        prompts_end=args.prompts_end,
        iterations=args.iterations,
        k=args.k,
        batch_size=args.batch_size,
        use_qwen=args.use_qwen,
        suffix_len=args.suffix_len,
        seed_llada=args.seed_llada
    )