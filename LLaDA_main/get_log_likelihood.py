import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import numpy as np

def forward_process(batch, prompt_index, mask_id):
    b, l = batch.shape

    target_len = (l - prompt_index.sum()).item()
    k = torch.randint(1, target_len + 1, (), device=batch.device)

    x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
    x = ((x - 1) % target_len) + 1
    assert x.min() >= 1 and x.max() <= target_len

    indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)
    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(target_len)]

    is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)
    noisy_batch = torch.where(is_mask, mask_id, batch)

    # Return the masked batch and the mask ratio
    return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)


def get_logits(model, batch, prompt_index, cfg_scale, mask_id):
    if cfg_scale > 0.:
        assert len(prompt_index) == batch.shape[1]
        prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
        un_batch = batch.clone()
        un_batch[prompt_index] = mask_id
        batch = torch.cat([batch, un_batch])

    input = batch
    logits = model(input).logits

    if cfg_scale > 0.:
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
    return logits


@ torch.no_grad()
def get_log_likelihood(model, prompt, answer, mc_num=128, batch_size=16, cfg_scale=0., mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (l1).
        answer: A tensor of shape (l2).
        mc_num: Monte Carlo estimation times.
                As detailed in Appendix B.5. Since MMLU, CMMLU, and C-EVAL only require the likelihood of a single token, a
                single Monte Carlo estimate is sufficient for these benchmarks. For all other benchmarks, we find that 128
                Monte Carlo samples are adequate to produce stable results.
        batch_size: Mini batch size.
        cfg_scale: Unsupervised classifier-free guidance scale.
        mask_id: The toke id of [MASK] is 126336.
    '''
    seq = torch.concatenate([prompt, answer])[None, :]
    seq = seq.repeat((batch_size, 1)).to(model.device)
    prompt_index = torch.arange(seq.shape[1], device=model.device) < len(prompt)

    loss_ = []

    checkpoints = {15, 32, 48, 64, 80, 96, 112, 128}
    checkpoint_losses = {}
    for iteration in range(mc_num // batch_size):
        perturbed_seq, p_mask = forward_process(seq, prompt_index, mask_id)
        mask_index = perturbed_seq == mask_id

        logits = get_logits(model, perturbed_seq, prompt_index, cfg_scale, mask_id)

        loss = F.cross_entropy(logits[mask_index], seq[mask_index], reduction='none') / p_mask[mask_index]
        loss = loss.sum() / batch_size

        loss_.append(loss.item())
    #     mc_position = iteration * batch_size
    #     if mc_position in checkpoints:
    #         running_avg = sum(loss_) / len(loss_)
    #         checkpoint_losses[mc_position] = running_avg
    # prompt_preview = " ".join(map(str, prompt.tolist()))[:40] + "..."

    # xs = list(checkpoint_losses.keys())
    # ys = list(checkpoint_losses.values())

    # plt.figure()
    # plt.plot(xs, ys)
    # plt.xlabel("Monte Carlo Iterations")
    # plt.ylabel("Loss")
    # plt.title(f"Loss vs Iterations ({prompt_preview})")
    # plt.grid(True)
    # plt.show()

    return - sum(loss_) / len(loss_)



def inpainting(model, tokenizer, target, num_steps, prompt_length, 
               input_prompt=None, mask_id=126336, temperature=1.0):
    """Implements the inpainting algorithm from "Diffusion LLMs are Natural Adversaries"""
    device = model.device
    
    # Get filtered vocabulary
    special_ids = set(tokenizer.all_special_ids)
    filtered_vocab = [
        i for i in range(100, model.config.vocab_size - 100)
        if i not in special_ids
    ]

    target_tokens = tokenizer(target, return_tensors="pt")["input_ids"].to(device)
    target_length = target_tokens.shape[1] # how many tokens to inpaint
    
    # Initialize with filtered vocab
    prompt_tokens = torch.tensor(
        np.random.choice(filtered_vocab, size=(1, prompt_length)),
        device=device
    )

    # if input_prompt is not None:
    #     temp = tokenizer(input_prompt, return_tensors="pt")["input_ids"].to(device)
    #     prefix_len = min(temp.shape[1], prompt_length)
    #     prompt_tokens[:, :prefix_len] = temp[:, :prefix_len]

    seq = torch.cat([prompt_tokens, target_tokens], dim=1) # add targets to the end
    
    prompt_index = torch.cat([
        torch.ones(prompt_length, dtype=torch.bool, device=device),
        torch.zeros(target_length, dtype=torch.bool, device=device)
    ]).unsqueeze(0)
    
    masked_seq, _ = forward_process(seq, prompt_index, mask_id) # do masking on empty tokens

    for t in reversed(range(1, num_steps + 1)):
        # Stochastic target masking
        if t > 1:
            mask_prob = 0.15 * (t / num_steps)
            target_mask = torch.rand(1, target_length, device=device) < mask_prob
            input_target = target_tokens.clone()
            input_target[target_mask] = mask_id
        else:
            input_target = target_tokens
        
        # Forward pass with masked target
        likelihoods = []
        logit_results = []
        for _ in range(8):
             logits = model(torch.cat([masked_seq[:, :prompt_length], input_target], dim=1)).logits
             prompt_logits = logits[:, :prompt_length, :]
             prompt_probs = F.softmax(prompt_logits / temperature, dim=-1)
             likelihoods.append(
                 F.cross_entropy(prompt_logits.reshape(-1, prompt_logits.shape[-1]), target_tokens[:,:prompt_length].reshape(-1))
             )
             logit_results.append(prompt_logits)
        prompt_logits = logit_results[torch.argmax(torch.tensor(likelihoods))]
        # input_seq = torch.cat([masked_seq[:, :prompt_length], input_target], dim=1)
        # logits = model(input_seq).logits
        
        # Filter vocabulary
        prompt_logits = logits[:, :prompt_length, :].clone()
        vocab_mask = torch.ones(prompt_logits.shape[-1], dtype=torch.bool, device=device)
        vocab_mask[filtered_vocab] = False
        prompt_logits[:, :, vocab_mask] = -float('inf')
        
        # Sample with temperature
        probs = F.softmax(prompt_logits / temperature, dim=-1)
        predicted_prompt = torch.multinomial(
            probs.view(-1, probs.shape[-1]), 
            num_samples=1
        ).view(1, prompt_length)
        
        # Unmask schedule
        if t == 1:
            num_to_unmask = prompt_length
        else:
            unmask_ratio = (num_steps - t + 1) / num_steps
            num_to_unmask = max(1, min(int(unmask_ratio * prompt_length), prompt_length))
        
        # Select by confidence
        confidences = probs.max(dim=-1).values
        keep_indices = torch.topk(confidences, num_to_unmask, dim=1).indices
        
        # Update
        updated_prompt = update_with_selection(
            masked_seq[:, :prompt_length],
            predicted_prompt,
            keep_indices,
            mask_id
        )
        
        # Always restore clean target
        masked_seq = torch.cat([updated_prompt, target_tokens], dim=1)
        
        if t % 100 == 0:
            try:
                preview = tokenizer.decode(updated_prompt[0].cpu().numpy(), skip_special_tokens=False)
                num_masked = (updated_prompt == mask_id).sum().item()
                print(f"  Step {t}: {num_masked}/{prompt_length} masked | {preview[:60]}")
            except:
                pass
    
    return tokenizer.batch_decode(
        masked_seq[:, :prompt_length], 
        skip_special_tokens=True
    )[0]


def update_with_selection(current_seq, predicted_tokens, keep_indices, mask_token_id):
    """Preserve previously unmasked tokens, only remask low-confidence ones"""
    batch_size, seq_len = current_seq.shape
    updated_seq = current_seq.clone()
    
    # Create update mask for selected positions
    update_mask = torch.zeros_like(current_seq, dtype=torch.bool)
    batch_indices = torch.arange(batch_size, device=current_seq.device).unsqueeze(1)
    update_mask[batch_indices, keep_indices] = True
    
    # Track already unmasked positions
    already_unmasked = (current_seq != mask_token_id)
    
    # Update selected positions with predictions
    updated_seq[update_mask] = predicted_tokens[update_mask]
    
    # Remask positions that weren't selected AND weren't already unmasked
    updated_seq[~update_mask & ~already_unmasked] = mask_token_id
    
    return updated_seq









def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)

    # this prompt and answer is from Hellaswag dataset
    prompt = 'Roof shingle removal: A man is sitting on a roof. He'
    answer = ' is using wrap to wrap a pair of skis.'

    prompt = torch.tensor(tokenizer(prompt)['input_ids']).to(device)
    answer = torch.tensor(tokenizer(answer)['input_ids']).to(device)
    print(get_log_likelihood(model, prompt, answer, mc_num=128))


if __name__ == '__main__':
    main()