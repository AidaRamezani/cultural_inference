import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from transformers import BertModel

BERT_ORIG = 'bert-large-nli-mean-tokens'


def get_function(function):
    def get_function_with_model_name(modelname):
        return function(modelname)
    return get_function_with_model_name

def get_bert_model(model_name = 'bert-base-uncased'):
    model = BertModel.from_pretrained(model_name)
    return model

def get_tokenizer(model_name = 'bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def gpt2_finetuned_model(model_name, test_type, train_data_name, use_cuda = True,device = None):
    model = GPT2LMHeadModel.from_pretrained(f'data/finetuning/outputs/{model_name}_{test_type}/{train_data_name}/')
    if use_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device(device)
        model.cuda(device)
    return model


def get_gpt2_model(modelname = 'gpt2', use_cuda = True, device = 'cuda:0'):
    if modelname not in ['gpt2','gpt2-medium','gpt2-large','gpt2-xl']:
        model = GPT2LMHeadModel.from_pretrained(f'data/finetuning/outputs/{modelname}/')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    else:
        model = GPT2LMHeadModel.from_pretrained(modelname)
        tokenizer = GPT2Tokenizer.from_pretrained(modelname)

    if use_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device(device)
        model.cuda(device)

    return tokenizer, model

def get_sbert_model(model_name = BERT_ORIG, use_cuda = False, device = 'cuda:0'):

    if use_cuda and torch.cuda.is_available():
        model = SentenceTransformer(model_name,device = device)
    else:
        model = SentenceTransformer(model_name,device = 'cpu')
    return model


