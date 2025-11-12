from transformers import AutoModel, AutoTokenizer
from LLaDA_main import generate
from LLaDA_main import get_log_likelihood
import torch.nn.functional as F
import torch
from project import model_qwen, qwen_tokenizer, device 
def gcg_single_attack_loss(model, x, target, I, T, k, B):
  model.eval()
  tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
  device = model.device
  mask_id = 126336
  # run T iterations of token substitution
  for t in range(T):
    # get vocab embeddings from the model
    embeddings = model.model.transformer.wte # set of embeddings for full vocabulary
    input_ids = tokenizer(x, return_tensors="pt")["input_ids"].to(device) # tokenize prompt string
    target_ids = tokenizer(target, return_tensors="pt")["input_ids"].to(device) # tokenize target string
    modifiable_ids = tokenizer(I, return_tensors="pt")["input_ids"].to(device)

    # get input embeddings so we can get gradients
    input_ids = input_ids.to(device)
    input_embeddings = embeddings(input_ids)
    input_embeddings.retain_grad()
    input_embeddings.requires_grad_()
    model_output = model(inputs_embeds=input_embeddings)

    output_logits = model_output.logits
    query = output_logits.view(-1, output_logits.size(-1)) # [batch*seq_len, vocab_size]
    resp = target_ids.view(-1) # [batch*seq_len]

    q = query.shape[0]
    r = resp.shape[0]

    # pad tensors with 0s if necessary for alignment
    ###
    #concat query output and response target and mask query and target
    #part respectively then compare them based on logits that are there, ignoring any positions that dont match
    ###
    if q < r:
      query = F.pad(query, (0, 0, 0, r - q), value=0)  # logits padding can be zeros
    elif r < q:
      resp = F.pad(resp, (0, 0, 0, q - r), value=mask_id)  # target padding with mask_id

    # compute per-token loss, ignoring padded positions
    original_loss = F.cross_entropy(query, resp,  ignore_index=mask_id)

    model.zero_grad()
    original_loss.backward()


    grads = input_embeddings.grad.squeeze(0) # TODO: modify to deal with I smaller than x
    n = len(input_ids[0])
    X = input_embeddings.clone().detach()
    Xi_vals = torch.empty((len(modifiable_ids[0]), k))
    Xi_indices = torch.empty((len(modifiable_ids[0]), k))

    # compute scores for each modifiable token
    for idx in range(len(modifiable_ids[0])):
      # get one-hot gradient for element i
      grad = grads[idx]
      candidate_scores = torch.matmul(embeddings.weight, -(grad))

      # compute top-k promising tokens
      Xi_vals[idx], Xi_indices[idx] = torch.topk(candidate_scores, k)

    x_iter = torch.zeros([B, len(input_ids[0])])
    losses = torch.zeros(B)

    for b in range(B):
      x_iter[b][0:n] = input_ids[0][0:n]

      # Select random element from I
      i = torch.randint(len(input_ids[0]), (1,)).item()

      # Select random element from Xi
      token_idx = Xi_indices[i][torch.randint(len(Xi_indices[i]), (1,)).item()]

      # retrieve token
      adv_token = tokenizer.convert_ids_to_tokens([token_idx.item()], skip_special_tokens=True)

      # substitute token
      x_iter[b][i] = token_idx.item()


      # compute output of model for adversarial prompt
      adv_output = generate.generate(model, x_iter[b].unsqueeze(0), steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')

      # retrieve token IDs for adversarial output
      adv_ids = tokenizer.batch_decode(adv_output[:, x_iter[b].unsqueeze(0).shape[1]:], skip_special_tokens=True)[0]
      adv_ids = tokenizer(adv_ids, return_tensors="pt")["input_ids"].to(device)

      # compute loss between adversarial IDs and IDs of target prompt
      loss = -1 * get_log_likelihood.get_log_likelihood(model, adv_ids.squeeze(0), target_ids.squeeze(0), mc_num=128, batch_size=16, cfg_scale=0., mask_id=126336)
      losses[b] = loss

    # return adversarial prompt with minimum loss
    x_ids = x_iter[torch.argmin(losses)]
    x = tokenizer.decode(x_ids.long().tolist(), skip_special_tokens=True)
  return x