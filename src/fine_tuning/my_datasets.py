from torch.utils.data import Dataset
import torch


EOS ='<|endoftext|>'
BOS ='<|endoftext|>'
PAD ='<|endoftext|>'

class SurveyData(Dataset):
    def __init__(self, df):
        self.prompts = df['prompt']
        self.country = df['country']
        self.topic = df['topic']
        self.rating = df['rating text']
        self.input_ids = []
        self.attn_masks = []
        self.labels = []

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

class SurveyDataForGPT2(SurveyData):


    def __init__(self, df, tokenizer, max_length):
        super().__init__(df)

        tokenizer.pad_token = tokenizer.eos_token

        for txt in self.prompts:
            encodings_dict = tokenizer(
                BOS + txt + EOS, truncation=True,max_length=max_length,
                padding="max_length", add_special_tokens = True)

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))












