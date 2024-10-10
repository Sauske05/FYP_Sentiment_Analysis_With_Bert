# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 08:06:16 2024

@author: Arun Joshi
"""
from torch.utils.data import Dataset
from tokenizer import Tokenizer


class CustomDataset(Dataset):
    def __init__(self, query_data, label, max_length:int):
        super().__init__()
        self.query_data = query_data
        self.label = label
        self.tokenizer = Tokenizer()
        self.max_length = max_length
        '''
        label_dictionary = {}
        for index, value in enumerate(label.unique()):
            label_dictionary[value] = index

        self.label =self.label.map(label_dictionary)
        
        print(label_dictionary)
        '''
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        tokenized_input = self.tokenizer.tokenize(self.query_data[index], self.max_length)
        input_ids = tokenized_input['input_ids']
        input_mask_ids = tokenized_input['attention_mask']
        
        return {
            
            'input_ids' : input_ids,
            'input_mask_ids' : input_mask_ids,
            'label' : self.label[index],
            'raw_text': self.query_data[index],
            'raw_text_len' : len(self.query_data[index].split())
            }
    
    #target is to the return the dict: