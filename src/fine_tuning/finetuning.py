import numpy as np
import torch
from src.fine_tuning.funtuning_split import  RANDOM_SEED, load_dataset
from src.fine_tuning.my_datasets import EOS, BOS, PAD, SurveyDataForGPT2
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TrainingArguments, Trainer
import argparse

EPOCH_NUM = 1
TOP_K = 20
TOP_P = 0.9
SAMPLE_NUM = 100
DATA_SOURCE = 'from_people'


def finetune_gpt2(model_name = 'gpt2', train_data_name = 'wvs', test_data_name = 'pew', data_source = 'from_people',
                  sample_num = 100, save_model = True):

    assert 'gpt2' in model_name

    torch.manual_seed(RANDOM_SEED)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.eos_token = EOS
    tokenizer.bos_token = BOS
    tokenizer.pad_token = PAD


    datasets, cross_test_dataset = \
        load_dataset(tokenizer, train_data_name, test_data_name,SurveyDataForGPT2,data_source, sample_num)

    for test_type, (train_data,dev_data, test_data, test_df) in datasets.items():

        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.resize_token_embeddings(len(tokenizer))
        training_arguments = TrainingArguments(f'data/finetuning/outputs/{model_name}_{test_type}',
                                               num_train_epochs=EPOCH_NUM,
                                               logging_steps= 10,
                                               load_best_model_at_end=True,
                                               per_device_train_batch_size=8,
                                               per_device_eval_batch_size=8,
                                               warmup_steps=100,
                                               weight_decay=0.01,
                                               logging_dir='data/finetuning/logs/',
                                               evaluation_strategy='epoch',
                                               save_strategy = 'epoch'
                                               )
        model.cuda(device = device)


        trainer = Trainer(model=model, args = training_arguments, train_dataset=train_data, eval_dataset=dev_data,
                data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                            'attention_mask': torch.stack([f[1] for f in data]),
                                            'labels': torch.stack([f[0] for f in data])})

        trainer.train()

        if save_model:
            torch.save(model.state_dict(),
                       f'data/finetuning/outputs/{model_name}_{test_type}/model_{train_data_name}.pt')
            trainer.save_model(f'data/finetuning/outputs/{model_name}_{test_type}/{train_data_name}/')





if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default='gpt2',
                        choices= ['gpt2','gpt2-medium','gpt2-large','gpt2-xl'] )
    parser.add_argument('--train', type = str, default='wvs',
                        choices= ['wvs','pew'] )
    parser.add_argument('--test', type = str, default='pew',
                        choices= ['wvs','pew'] )

    args = parser.parse_args()
    finetune_gpt2(args.model, args.train, args.test,DATA_SOURCE, SAMPLE_NUM)


















