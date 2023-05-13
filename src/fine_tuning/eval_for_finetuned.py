import numpy as np
import pandas as pd
import torch.cuda
import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from src.probing_experiments.compare_prompt_responses.compare_gpt2 import  \
    compare_gpt2_wvs_token_pairs,compare_gpt2_pew_token_pairs, compare_gpt2_pair_prompts_mort_user_study
from src.fine_tuning.my_datasets import EOS,BOS,PAD
from src.my_models import gpt2_finetuned_model

DEVICE = "cpu"
EPOCH_NUM = 1
TOP_K = 20
TOP_P = 0.9

WVS_TEXT_TO_RATING_MAPPING = {'never justifiable': 1, 'not justifiable': 3,
                              'somewhat justifiable': 5.5, 'justifiable': 8, 'always justifiable': 10}


def store_gpt2_model_on_pew(model_name, train_data_name, test_type, use_cuda = True):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.eos_token = EOS
    tokenizer.bos_token = BOS
    tokenizer.pad_token = PAD

    model = gpt2_finetuned_model(model_name, test_type, train_data_name, use_cuda)
    compare_gpt2_pew_token_pairs(model_name=f'gpt2_{test_type}_on_{train_data_name}',
                     excluding_topics=[], model = model, tokenizer = tokenizer, use_cuda=True)


def store_gpt2_model_on_wvs(model_name, train_data_name, test_type, use_cuda = True):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.eos_token = EOS
    tokenizer.bos_token = BOS
    tokenizer.pad_token = PAD

    model = gpt2_finetuned_model(model_name, test_type, train_data_name, use_cuda)

    compare_gpt2_wvs_token_pairs(model_name=f'gpt2_{test_type}_on_{train_data_name}',
                     excluding_topics=[], model = model, tokenizer = tokenizer, use_cuda = use_cuda)


def evaluate_model_on_user_study(model_name, train_data_name, strategy, use_cuda = True):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.eos_token = EOS
    tokenizer.bos_token = BOS
    tokenizer.pad_token = PAD

    model = gpt2_finetuned_model(model_name, strategy, train_data_name, use_cuda)

    compare_gpt2_pair_prompts_mort_user_study(style='mv_at_end',
                                              modelname=f'gpt2_{strategy}_on_{train_data_name}',
                                              model=model, tokenizer = tokenizer, use_cuda=use_cuda)


if __name__ == '__main__':
    use_cuda = False
    if torch.cuda.is_available():
        torch.cuda.set_device(DEVICE)
        torch.cuda.empty_cache()
        use_cuda = True
        DEVICE = "cuda:0"


    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default='gpt2',
                        choices= ['gpt2','gpt2-medium','gpt2-large','gpt2-xl'] )
    parser.add_argument('--train', type = str, default='wvs',
                        choices= ['wvs','pew'],help = 'train data' )
    parser.add_argument('--strategy', type = str,
                        choices= ['country_based','topic based','random'],
                        required=True)
    args = parser.parse_args()

    store_gpt2_model_on_pew(args.model,args.train,args.strategy,use_cuda) #Evaluating on PEW
    store_gpt2_model_on_wvs(args.model,args.train,args.strategy,use_cuda) #Evaluating on WVS
    evaluate_model_on_user_study(args.model, args.train, args.strategy) #Evaluting on globalAMT









