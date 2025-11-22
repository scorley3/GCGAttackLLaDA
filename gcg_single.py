from transformers import AutoModel, AutoTokenizer
from LLaDA_main import generate
from LLaDA_main import get_log_likelihood
import torch.nn.functional as F
import torch
import gc



def gcg_single_attack_loss(model, tokenizer, x, target, suffix_len, T, k, B, mc_num, seed="None"):
  model.eval()
  device = model.device
  embeddings = model.model.transformer.wte
  prefix_ids = tokenizer(x, return_tensors="pt")["input_ids"].to(device)
  target_ids = tokenizer(target, return_tensors="pt")["input_ids"].to(device)
  prefix_len = prefix_ids.shape[1]
  loss_vals = []
  if seed != "None":
    suffix_tokens = tokenizer(seed, return_tensors="pt")["input_ids"].to(device)
  #randomly seed suffix
  else:
    suffix_tokens = torch.randint(low=100, high=model.config.vocab_size - 100, size=(1, suffix_len),device=device)
  
  for t in range(T):
    input_ids = torch.cat([prefix_ids, suffix_tokens], dim=1)

    input_embeddings = embeddings(input_ids)
    input_embeddings.retain_grad()
    input_embeddings.requires_grad_()


    model_output = model(inputs_embeds=input_embeddings)

    output_logits = model_output.logits
    model_pred_slice = output_logits[:, prefix_len - 1 : prefix_len - 1 + target_ids.shape[1], :] # from last token of prompt to the size of target

    # compute per-token loss, ignoring padded positions
    original_loss = F.cross_entropy(
            model_pred_slice.reshape(-1, model_pred_slice.size(-1)), #reshape to size (flatten(batch x seq_len), vocab size)
            target_ids.reshape(-1) #flatten
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

    candidates = torch.zeros([B, len(input_ids[0])], dtype=torch.long, device=device)
    losses = torch.zeros(B,device=device)

    for b in range(B):
      temp = input_ids.clone()[0]

      # Select random element from I
      i = prefix_len + torch.randint(suffix_len, (1,)).item()

      # Select random element from Xi
      token_idx = Xi_indices[i - prefix_len][torch.randint(k, (1,)).item()]
      temp[i] = token_idx
      candidates[b] = temp
      with torch.no_grad():
        losses[b] = -1 * get_log_likelihood.get_log_likelihood(model, candidates[b], target_ids.squeeze(0), mc_num=mc_num, batch_size=16, cfg_scale=0., mask_id=126336)
    loss_vals.append(torch.min(losses))
    x_ids = candidates[torch.argmin(losses)]
    suffix_tokens = x_ids[prefix_len:].unsqueeze(0)
    # retrieve token
    del input_ids, input_embeddings, grads, candidates, losses, Xi_indices, Xi_vals
    torch.cuda.empty_cache()  
    gc.collect()

    # return adversarial prompt with minimum loss
  final_ids = torch.cat([prefix_ids, suffix_tokens], dim=1)
  return tokenizer.decode(final_ids[0], skip_special_tokens=True), [l.cpu().item() for l in loss_vals]