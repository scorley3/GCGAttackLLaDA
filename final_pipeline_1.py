"""
This Pipeline is made to test base conclusively using both prefix and suffix attacks.
Early stoppage will be used by checking for a positive result after each attack iteration.

Brainstorm:
If we generate all adversarial strings at once, we cannot benefit from early stoppage as what works on one model may not work on another.
If we dont generate all adversarial strings at once, we increase the attack time threefold but we can do iterative checks on each generated attack
We want to check generating base attacks and use them on base, instruct, and llada 2.0 mini
We can first generate attacks on base, then use those attacks that work on instruct, if they dont work, use that to seed a new attack so that the progress isnt all lost.
We can do base -> mini -> instruct as that should be the order of least to most robust
"""
from evaluate import *
from gcg_for_pipeline import *
from data import get_dataset
from models import *
from gcg_for_pipeline import *
import torch, gc
import argparse
import json
import os
from pathlib import Path

def PIPELINE_prefix_only(llada_model, llada_tokenizer, prompts, targets, iters=3, k=2, batch_size=25, model_name="llada_base", mc_num=64):
    total = len(prompts)
    successful = 0
    device = llada_model.device
    prompts_to_push = [None] * total
    dict_adversarial_prompts = {}
    for i in range(total):
        prompt = prompts[i]
        target = targets[i]

        print(f"\n=== Running LLADA Attack for Prompt {i}: {prompt} ===")

        optimized_prompt, loss_vals, adv_prompts, saved_outputs = gcg_base_prefix(llada_model, llada_tokenizer, prompt, target, iters, k, batch_size, mc_num=mc_num)

        save_loss_plot(loss_vals, f"Pipeline_imgs/base{i}.png") 
        attack_log_entry = {}
        
        attack_log_entry = {
        "Potential prompt": optimized_prompt,
        "Loss values": loss_vals,
        "Prompts": adv_prompts,
        "outputs": saved_outputs
        }
        dict_adversarial_prompts[i] = adv_prompts
        update_log(filepath="experiment/base_pre.json", prompt_index=i, model_name="base_model", attack_output=attack_log_entry)

    print(f"\n[RESULTS] Successful attacks: {successful}/{total}")
    print(f"          Success rate: {successful / total:.2%}")

    return dict_adversarial_prompts


def PIPELINE_suffix(llada_model, llada_tokenizer, prompts, targets, log_path, model_name, seeds=None, iters=5, k=10, batch_size=32, mc_num=64):
    total = len(prompts)
    device = llada_model.device
    dict_adversarial_prompts = {}
    for i in range(total):
       
        prompt = prompts[i]
        target = targets[i]

        print(f"\n=== Running LLADA Attack for Prompt {i}: {prompt} ===")
        if seeds != None:
            seed_prompt = seeds[i]
            optimized_prompt, best_results, outputs = gcg_base_suffix(llada_model, llada_tokenizer, prompt, target, iters, k, batch_size, mc_num=mc_num, seed=seed_prompt)
            best_results.append((0, seed_prompt))
        else:
            optimized_prompt, best_results, outputs = gcg_base_suffix(llada_model, llada_tokenizer, prompt, target, iters, k, batch_size, mc_num=mc_num)
        
        best_results = [prompt for _, prompt in best_results]
        evaluate_on_models(best_results, i, llada_model, llada_tokenizer, model_name, log_path)
        dict_adversarial_prompts[i] = best_results
    
        attack_log_entry = {
            "Potential prompt": optimized_prompt,
            "outputs": outputs
        }
        update_log(filepath=log_path, prompt_index=i, model_name="base_model", attack_output=attack_log_entry)
    return dict_adversarial_prompts


def PIPELINE_qwen_suffix(model, tokenizer, prompts, targets, suffix_len=20, iters=60, k=32, batch_size=64):
    total = len(prompts)
    device = model.device
    dict_adversarial_prompts = {}
    for i in range(total):
        prompt = prompts[i]
        target = targets[i]

        print(f"\n=== Running Qwen Attack for Prompt {i}: {prompt} ===")
        #GCG suffix responds the "optimal" prompt and the top 10 prompts with lowest loss
        optimized_prompt = qwen_gcg_prompt_generation(model, tokenizer, prompt, target,suffix_len, iters, k, batch_size)
        print("Generated Adversarial Prompt: ", optimized_prompt)
        dict_adversarial_prompts[i]=optimized_prompt
    return dict_adversarial_prompts


def evaluate_on_models(adversarial_prompts, prompt_num, model, tokenizer, model_name, log_path):
    device = model.device
    outputs = []
    for adv_prompt in adversarial_prompts:
        
        if model_name == "llada_base":
            output = llada_generate_base(adv_prompt, model, tokenizer, device=device)
            outputs.append(output)
        elif model_name == "llada_instruct":
            output = llada_generate(adv_prompt, model, tokenizer, device=device)
            outputs.append(output)
    attack_log_entry = {
        "Prompts": adversarial_prompts,
        "outputs": outputs
        }
    update_log(filepath=log_path, prompt_index=prompt_num, model_name=model_name, attack_output=attack_log_entry)
    
    

def initialize_log(filepath, prompts, model_names):
    """
    Create a logging json structure:
    {
        "0": {"prompt": "...", "modelA": {"attacks": []}, ...},
        "1": {...},
        ...
    }
    """
    log = {}

    for i, p in enumerate(prompts):
        log[str(i)] = {"prompt": p}
        for m in model_names:
            log[str(i)][m] = {"attacks": []}

    with open(filepath, "w") as f:
        json.dump(log, f, indent=2)

    print(f"[INIT] Log initialized at {filepath}")

def update_log(filepath, prompt_index, model_name, attack_output):
    # Load
    with open(filepath, "r") as f:
        log = json.load(f)

    # Convert index to str because JSON object keys are strings
    idx = str(prompt_index)

    # Append attack
    log[idx][model_name]["attacks"].append(attack_output)

    # Save again
    with open(filepath, "w") as f:
        json.dump(log, f, indent=2)

    print(f"[LOG] Added attack for prompt {idx} under model '{model_name}'")




def main():
    gc.collect()
    torch.cuda.empty_cache()
    _, targets = get_dataset()
    refused_prompts = json.load(open("base_refused_prompts.json","r"))
    model_base, tokenizer_base = load_llada_base()
    print("[PIPELINE] Loaded LLaDA Base Model")
    model_names = ["base_model", "instruct_model"]
    #initialize_log("experiment/base_pre.json", refused_prompts, model_names)
    #initialize_log("experiment/base_suf_highBatch_log.json", refused_prompts, model_names)
    initialize_log("experiment/base_suf_seeded_log.json", refused_prompts, model_names)
    initialize_log("experiment/instruct_suf_seeded_log.json", refused_prompts, model_names)
    print("[PIPELINE] Initialized Logs")
    #First, run the attack on base prompts that were refused
    #Use multiple restarts incase an attack fails? -> how do we catch when it fails?
    canonical_targets = [targets[int(index)] for index in refused_prompts][:50]
    canonical_prompts = [value for key, value in refused_prompts.items()][:50]
    #base prefix
    #print("[PIPELINE] Starting Base Prefix Attacks")
    #prefix_adversarial_prompts = PIPELINE_prefix_only(model_base, tokenizer_base, canonical_prompts, canonical_targets, iters=5, k=5, batch_size=25, model_name="base_model", mc_num=64)
    
    #base suffix medium iter higher batch/ evaluates
    #suffix_adversarial_prompts = PIPELINE_suffix(model_base, tokenizer_base, canonical_prompts, canonical_targets, model_name="base_model", log_path="experiment/base_suf_highBatch_log.json", iters=60, batch_size=32, k=10)
    
    unload_model(model_base)
    model_qwen, tokenizer_qwen = load_qwen()
    #qwen seed on base
    qwen_adversarial_prompts = PIPELINE_qwen_suffix(model_qwen, tokenizer_qwen, canonical_prompts, canonical_targets, 20, iters=64, k=32, batch_size=32)
    unload_model(model_qwen)
    

    
    model_base, tokenizer_base = load_llada_base()
    #pass qwen prompts as seeds to llada base and llada instruct suffix attacks, testing on the initial one aswell
    #evaluates for base
    suffix_seeded_prompts = PIPELINE_suffix(model_base, tokenizer_base, canonical_prompts, canonical_targets, model_name="base_model", log_path="experiment/base_suf_seeded_log.json", seeds=qwen_adversarial_prompts, iters=30, batch_size=16, k=10)
    unload_model(model_base)

    os.system("rm -rf ~/.cache")
    model_instruct, tokenizer_instruct = load_llada()
    for prompt_num, adversarials in suffix_seeded_prompts.items():
        evaluate_on_models(adversarials, prompt_num, model_instruct, tokenizer_instruct, "instruct_model", "experiment/instruct_suf_seeded_log.json")

    unload_model(model_instruct)
    return


if __name__ == "__main__":
    main()
