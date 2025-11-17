from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import torch
import gc



def qwen_gcg_single_attack(model, tokenizer, x, target, suffix_len, T, k, B):
  model.eval()
  device = model.device
  prefix_ids = tokenizer(x, return_tensors="pt")["input_ids"].to(device) # tokenize prompt string
  #target_ids = tokenizer(target, return_tensors="pt")["input_ids"].to(device) # tokenize target string
  target_ids = tokenizer(target, return_tensors="pt",add_special_tokens=False)["input_ids"].to(device)
  prefix_len = prefix_ids.shape[1]
  suffix_tokens = torch.randint(low=100, high=model.config.vocab_size - 100, size=(1, suffix_len),device=device)
  # run T iterations of token substitution
  for t in range(T):
    input_ids = torch.cat([prefix_ids, suffix_tokens], dim=1)
    # get vocab embeddings from the model
    embeddings = model.get_input_embeddings()
    
    #modifiable_ids = tokenizer(I, return_tensors="pt")["input_ids"].to(device)

    # get input embeddings so we can get gradients
    input_embeddings = embeddings(input_ids)
    input_embeddings.retain_grad()
    input_embeddings.requires_grad_()
    
    model_output = model(inputs_embeds=input_embeddings)
    output_logits = model_output.logits
    min_len = min(target_ids.shape[1], output_logits.shape[1] - (prefix_len - 1))
    model_pred_slice = output_logits[:, prefix_len - 1 : prefix_len - 1 + min_len, :] # from last token of prompt to the size of target
    # pred_slice = output_logits[:, prefix_len-1:, :]
    # pred_slice = pred_slice[:, :target_ids.shape[1], :]
    # model_pred_slice = pred_slice
    
    # compute per-token loss, ignoring padded positions
    target_temp = target_ids[:model_pred_slice.size(0)].reshape(-1)
    target_temp = target_temp[:min_len]
    original_loss = F.cross_entropy(
            model_pred_slice.reshape(-1, model_pred_slice.size(-1)), #reshape to size (flatten(batch x seq_len), vocab size)
            target_temp
        )

    model.zero_grad()
    original_loss.backward()
    grads = input_embeddings.grad.detach().clone()[0]

    # delete anything that still holds graph references
    del model_output, output_logits, original_loss
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    Xi_vals = torch.empty((suffix_len, k))
    Xi_indices = torch.empty((suffix_len, k))

    # compute scores for each modifiable token
    for idx in range(suffix_len):
      # get one-hot gradient for element i
      grad = grads[idx]
      candidate_scores = torch.matmul(embeddings.weight, -(grad))

      # compute top-k promising tokens
      Xi_vals[idx], Xi_indices[idx] = torch.topk(candidate_scores, k)
    #B random subsitutions
    candidates = torch.zeros([B, len(input_ids[0])], dtype=torch.long).to(device)
    losses = torch.zeros(B)

    for b in range(B):
      temp = input_ids.clone()[0]
      
      # Select random element from I
      i = prefix_len + torch.randint(suffix_len, (1,)).item()

      # Select random element from Xi
      token_idx = Xi_indices[i - prefix_len][torch.randint(k, (1,)).item()]
      temp[i] = token_idx
      candidates[b] = temp
      # retrieve token

      # compute output of model for adversarial prompt
      with torch.no_grad():
        adv_output = model(candidates[b].unsqueeze(0))
        adv_logits = adv_output.logits[:, prefix_len - 1 : prefix_len - 1 + min_len, :]
        losses[b] = F.cross_entropy(adv_logits.reshape(-1, adv_logits.shape[-1]), target_temp)
      # per-candidate cleanup
      del adv_output, adv_logits

    # return adversarial prompt with minimum loss
    x_ids = candidates[torch.argmin(losses)]
    suffix_tokens = x_ids[prefix_len:].unsqueeze(0)
    
    del input_ids
    del input_embeddings, grads
    del Xi_vals, Xi_indices, candidates, losses
    torch.cuda.empty_cache()
    gc.collect()
  final_ids = torch.cat([prefix_ids, suffix_tokens],dim=1)
  return tokenizer.decode(final_ids[0], skip_special_tokens=True)