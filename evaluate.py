import torch
import json
from LLaDA_main import generate, get_log_likelihood
from transformers import AutoTokenizer
from gcg_qwen import qwen_gcg_single_attack
from gcg_single import gcg_single_attack_loss
import gc
from models import load_qwen, load_llada, unload_model
import json
from pathlib import Path
from typing import List, Any

def update_log_entry(prompt_id, update_dict, log_path="attack_log_llada.json"):

    try:
        with open(log_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []

    # Look for existing entry
    found = False
    for entry in data:
        if entry.get("prompt_id") == prompt_id:
            entry.update(update_dict)
            found = True
            break

    if not found:
        # Create new entry
        new_entry = {"prompt_id": prompt_id, **update_dict}
        data.append(new_entry)

    # Write back to file
    with open(log_path, "w") as f:
        json.dump(data, f, indent=2)

def evaluate(prompts, targets, iters, k, batch_size, use_qwen=True, seed_llada="False", device="cuda", suffix_len=20,log_path="attack_log.json"):
    gc.collect()
    torch.cuda.empty_cache()
    total = len(prompts)
    successful = 0
    adversarial_strings = [] 
    path = "attack_log.json"
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)

    # If file contains a single object, make it a list of one object
    if isinstance(data, dict):
        data = [data]

    for obj in data:
        # safe-get the field; skip objects without it
        val: Any = obj.get("optimized_prompt_qwen")
        if isinstance(val, str):
            adversarial_strings.append(val)
    print(adversarial_strings)
    # QWEN ATTACK PATH
    if use_qwen:
     
        qwen_model, qwen_tokenizer = load_qwen(device=device)

        for i in range(total):
            prompt = prompts[i] #+ ". (Respond in English)"
            target = targets[i]
            print(f"\n=== Running Qwen Attack for Prompt {i}: {prompt} ===")

            print("\n--- Running GCG Attack on Qwen ---")
            optimized_prompt = qwen_gcg_single_attack(qwen_model, qwen_tokenizer, prompt, target, suffix_len, iters, k, batch_size)
            print("OPTIMIZED PROMPT (Qwen):", optimized_prompt)

            # Evaluate on Qwen
            print("\n--- Testing Adversarial Prompt on Qwen ---")
            q_ids = qwen_tokenizer(optimized_prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                output_ids = qwen_model.generate(
                    q_ids,
                    max_length=q_ids.shape[1] + 128,
                    do_sample=False,
                )

                qwen_output = qwen_tokenizer.decode(output_ids[0][q_ids.shape[1]:], skip_special_tokens=True)
            print("QWEN OUTPUT:", qwen_output)

            if target.lower() in qwen_output.lower():
                print("QWEN ATTACK SUCCESSFUL")

            # save for LLADA
            adversarial_strings.append((optimized_prompt, target))
            # Save for LLADA
            adversarial_strings.append((optimized_prompt, target))

            update_log_entry(
                prompt_id=i,
                update_dict={
                    "prompt": prompt,
                    "target": target,
                    "optimized_prompt_qwen": optimized_prompt,
                    "qwen_output": qwen_output,
                },
                log_path=log_path
            )

        # unload Qwen before loading LLADA in order to save ram
        unload_model(qwen_model)
        if seed_llada != "True":
            # reload llama 
            llada_model, llada_tokenizer = load_llada(device=device)
            m = [{"role": "user", "content": prompt}, ]
            for optimized_prompt, target in adversarial_strings:
                print("\n--- Evaluating Qwen Adversarial Prompt on LLADA ---")
                prompt = llada_tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                input_ids = llada_tokenizer(prompt)['input_ids']
                input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

                response_ids = generate.generate(
                    llada_model,
                    input_ids,
                    steps=128,
                    gen_length=128,
                    block_length=32,
                    temperature=0.0,
                    cfg_scale=0.0,
                    remasking="low_confidence"
                )

                llada_output = llada_tokenizer.batch_decode(
                    response_ids[:, input_ids.shape[1]:],
                    skip_special_tokens=True
                )[0]

                print("LLADA OUTPUT:", llada_output)

                if target.lower() in llada_output.lower():
                    print("LLADA ATTACK SUCCESSFUL")
                    successful += 1
                update_log_entry(
                    prompt_id=i,
                    update_dict={
                        "optimized_prompt_llada": optimized_prompt,
                        "llada_output": llada_output,
                    },
                    log_path=log_path
                )

            return successful / total

    # LLADA ONLY PATH
    elif use_qwen == False or (use_qwen == True and seed_llada == "True"):
        llada_model, llada_tokenizer = load_llada(device=device)

        for i in range(total):
            prompt = prompts[i]
            target = targets[i]

            print(f"\n=== Running LLADA Attack for Prompt {i}: {prompt} ===")
            if seed_llada == "True":
                print(f"\n=== Using Seed: {adversarial_strings[i]} ===")

            optimized_prompt = gcg_single_attack_loss(llada_model, llada_tokenizer, prompt, target, suffix_len, iters, k, batch_size, seed=adversarial_strings[i])
            print("OPTIMIZED PROMPT:", optimized_prompt)
            m = [{"role": "user", "content": optimized_prompt}, ]
            optimized_prompt = llada_tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            input_ids = llada_tokenizer(optimized_prompt)['input_ids']
            input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
            response_ids = generate.generate(
                llada_model,
                input_ids,
                steps=128,
                gen_length=128,
                block_length=32,
                temperature=0.0,
                cfg_scale=0.0,
                remasking="low_confidence"
            )

            llada_output = llada_tokenizer.batch_decode(
                response_ids[:, input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]

            print("LLADA OUTPUT:", llada_output)

            if target.lower() in llada_output.lower():
                print("ATTACK SUCCESSFUL")
                successful += 1
            update_log_entry(
                prompt_id=i,
                update_dict={
                    "prompt": prompt,
                    "target": target,
                    "optimized_prompt_llada": optimized_prompt,
                    "llada_output": llada_output,
                },
                log_path=log_path
            )

        return successful / total