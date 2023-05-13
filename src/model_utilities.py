import torch
import numpy as np
from torch.nn import functional as F

def get_model_perplexity(model, tokenizer, sentence):
    model.eval()
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss=model(tensor_input, labels = tensor_input)[0]

    return np.exp(loss.cpu().detach().numpy())
def add_model_perplexity(prompts, model, tokenizer):
    for row in prompts:
        moral_p = -get_model_perplexity(model, tokenizer, row['moral prompt'])
        nonmoral_p = -get_model_perplexity(model, tokenizer, row['nonmoral prompt'])

        difference = moral_p - nonmoral_p
        row['perplexity difference'] = difference
    return prompts,'perplexity difference'



def get_batch_log_prob(lines, model, tokenizer, use_cuda  = False):
    tokenizer.pad_token = tokenizer.eos_token
    tok_moral = tokenizer.batch_encode_plus(lines, return_tensors='pt', padding='max_length',add_special_tokens=True)
    input_ids = tok_moral['input_ids']
    attention_mask = tok_moral['attention_mask']
    lines_len = torch.sum(tok_moral['attention_mask'], dim=1)
    if use_cuda:
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        torch.cuda.empty_cache()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss, logits = outputs[:2]
        probs = []
        for line_ind in range(len(lines)):
            line_log_prob = 0.0
            for token_ind in range(lines_len[line_ind] - 1):
                token_prob = F.softmax(logits[line_ind, token_ind], dim=0)
                token_id = input_ids[line_ind, token_ind + 1]
                line_log_prob += torch.log(token_prob[token_id])
            average_log_prob = line_log_prob / (lines_len[line_ind] - 1)
            probs.append(average_log_prob.cpu().detach().numpy())

    return probs


def get_batch_last_token_log_prob(lines, model, tokenizer, use_cuda  = False, end_with_period = True):
    tokenizer.pad_token = tokenizer.eos_token
    tok_moral = tokenizer.batch_encode_plus(lines, return_tensors='pt', padding='max_length',add_special_tokens=True)
    input_ids = tok_moral['input_ids']
    attention_mask = tok_moral['attention_mask']
    lines_len = torch.sum(tok_moral['attention_mask'], dim=1)


    remove_from_end = 3 if end_with_period else 2
    tokens_wanted = [ll - remove_from_end for ll in lines_len]
    if use_cuda:
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        torch.cuda.empty_cache()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, return_dict = True)
        logits = outputs['logits']
        logits = logits.detach().cpu()

        logits_zeros = torch.zeros_like(logits)


        for i, line in enumerate(lines):
            logits_zeros[i][tokens_wanted[i]][input_ids[i][tokens_wanted[i] + 1]] = 1

        logits_probs = torch.log(torch.softmax(logits, dim = -1))
        zeroed = logits_probs * logits_zeros
        log_probs = zeroed.sum(2).sum(1)

    return log_probs


def get_batch_log_prob_matrix(lines, model, tokenizer, use_cuda  = False):
    tokenizer.pad_token = tokenizer.eos_token
    tok_moral = tokenizer.batch_encode_plus(lines, return_tensors='pt', padding='max_length',add_special_tokens=True)
    lines_len = torch.sum(tok_moral['attention_mask'], dim=1)

    input_ids = tok_moral['input_ids']
    attention_mask = tok_moral['attention_mask']

    if use_cuda:
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        torch.cuda.empty_cache()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, return_dict = True)
        logits = outputs['logits']
        logits = logits.detach().cpu()


        logits_zeros = torch.zeros_like(logits)
        for i, line in enumerate(lines):
            for j in range(lines_len[i] - 1):
                token_ind = input_ids[i][j + 1]
                attention = attention_mask[i][j]
                if attention == 1:
                    logits_zeros[i][j][token_ind] = 1


        logits_probs = torch.log(torch.softmax(logits, dim = -1))
        zeroed = logits_probs * logits_zeros
        zeroed_sum = zeroed.sum(2)
        lengths = logits_zeros.sum(1).sum(1)
        log_probs = zeroed_sum.sum(1) / lengths



    return log_probs


def get_lines_log_prob(lines, model, tokenizer, use_cuda = False):
    lines = [tokenizer.eos_token + line for line in lines]
    logprobs = get_batch_log_prob(lines, model, tokenizer, use_cuda)
    return logprobs

def get_lines_log_prob_last_token(lines, model, tokenizer, use_cuda = False):
    lines = [tokenizer.eos_token + line for line in lines]
    logprobs = get_batch_last_token_log_prob(lines, model, tokenizer, use_cuda)
    return logprobs

def get_lines_log_prob_matrix(lines, model, tokenizer, use_cuda = False):
    lines = [tokenizer.eos_token + line for line in lines]
    logprobs = get_batch_log_prob_matrix(lines, model, tokenizer, use_cuda)
    return logprobs


def get_sbert_embeddings(sentences, model):
    embedding = model.encode(sentences)
    return embedding

