import torch
from torch.utils.data.dataset import Dataset
import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import os

    
class TransformerLMDataset(Dataset):
    def __init__(self, tokenizer, data, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self._convert_seqs_to_token()
    
    def _convert_seqs_to_token(self):
        texts = self.tokenizer(self.data.text.astype(str).values.tolist())
        concat_tokens = [t for tokens in texts.input_ids for t in tokens]
        # turncate concat_tokens length to multiple of self.max_len
        concat_tokens = concat_tokens[:self.max_len * (len(concat_tokens) // self.max_len)]
        # reshape to [-1, self.max_len]
        fixed_len_seqs = []
        for i in range(0, len(concat_tokens), self.max_len):
            fixed_len_seqs.append(concat_tokens[i:i+self.max_len])
        self.dataset = fixed_len_seqs

    def __getitem__(self, index):
        return self.dataset[index], 0, 0, None

    def collate_fn(self, data):
        #List of sentences and frames [B,]
        input_ids, speaker_ids, stays = zip(*data)
        speaker_ids = torch.LongTensor(speaker_ids).view(-1, 1)
        input_ids = torch.LongTensor(input_ids)
        
        return input_ids, None, None, 0, stays, None, speaker_ids

    def __len__(self):
        return len(self.dataset)
