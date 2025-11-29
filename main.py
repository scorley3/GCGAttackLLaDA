import torch, gc
import torch.nn.functional as F
from evaluate import evaluate, inpainting_evaluate
from data import get_dataset
import argparse
#prompts, targets, iters, k, batch_size, plot_path, use_qwen, seed_llada, device, suffix_len,log_path, prefill_string
def main(prompts_start, prompts_end, iterations, k, batch_size,use_qwen, suffix_len, seed_llada, prefill_string, log_path, plot_path, mc_num, change_prefix, inpainting): #prompts, targets, iters, k, batch_size, use_qwen=True, device="cuda", suffix_len=30
    gc.collect()
    torch.cuda.empty_cache()
    prompts, targets = get_dataset()
    if inpainting:
        inpainting_evaluate(prompts[prompts_start:prompts_end], targets[prompts_start:prompts_end], iterations, prompt_length=suffix_len)
    else:
        evaluate(prompts[prompts_start:prompts_end], targets[prompts_start:prompts_end], iterations, k, batch_size=batch_size, use_qwen=use_qwen,seed_llada=seed_llada, device="cuda", suffix_len=suffix_len, log_path=log_path, prefill_string=prefill_string, plot_path=plot_path, mc_num=mc_num, change_prefix=change_prefix)



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes','true','t','1'):
        return True
    elif v.lower() in ('no','false','f','0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run GCG adversarial attacks on Qwen and/or LLADA models."
    )

    parser.add_argument("--prompts_start", type=int, default=0, help="Start index for dataset prompts (gcg + inpainting)")
    parser.add_argument("--prompts_end", type=int, default=50, help="End index for dataset prompts (gcg + inpainting)")
    parser.add_argument("--iterations", type=int, default=40, help="Number of attack iterations (gcg + inpainting)")
    parser.add_argument("--k", type=int, default=256, help="Top-k tokens considered for substitution (gcg)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for random substitutions (gcg)")
    parser.add_argument("--use_qwen", type=str2bool, default=False, help="Whether to attack Qwen model first (gcg)")
    parser.add_argument("--suffix_len", type=int, default=15, help="Length of suffix to optimize (gcg + inpainting)")
    parser.add_argument("--seed_llada", type=str2bool, default=False, help="Whether to seed LLADA with Qwen adversarial prompt (gcg)")
    parser.add_argument("--prefill_string", type=str2bool, default=False, help="String to prefill in LLADA prompts (gcg)")
    parser.add_argument("--log_path", type=str, default="attack_log.json", help="Path to log attack results (gcg)")
    parser.add_argument("--plot_path", type=str, default="Loss_plot_prompt", help="Path to plot results (gcg)")
    parser.add_argument("--mc_num", type=int, default=64, help="Number of Monte Carlo samples for LLADA evaluation (gcg)")
    parser.add_argument("--change_prefix", type=str2bool, default=False, help="Update prefix as well as suffix (gcg)")
    parser.add_argument("--inpainting", type=str2bool, default=False, help="Run inpainting attack (inpainting)")
    args = parser.parse_args()
    
    main(
        prompts_start=args.prompts_start,
        prompts_end=args.prompts_end,
        iterations=args.iterations,
        k=args.k,
        batch_size=args.batch_size,
        use_qwen=args.use_qwen,
        suffix_len=args.suffix_len,
        seed_llada=args.seed_llada,
        prefill_string=args.prefill_string,
        log_path=args.log_path,
        plot_path=args.plot_path,
        mc_num=args.mc_num,
        change_prefix=args.change_prefix,
        inpainting=args.inpainting
    )