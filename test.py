# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 07:44:09 2024

@author: LENOVO
"""


#Testing the tokenizer class
def test_tokenizer():
    '''
    A dictionary is created with key values of 'input_ids',
    'token_type_ids', 'attention_mask'
    '''
    from tokenizer import Tokenizer


    tokenizer_obj = Tokenizer()
    test_sentence = 'Hello, How are you?'

    tokenized_input = tokenizer_obj.tokenize(test_sentence, 100)
    return tokenized_input, tokenizer_obj
    #print(tokenized_input)

#test_tokenizer()


#Testing the dataset class

def test_dataset():
    import pandas as pd

    df = pd.read_csv("hf://datasets/TungHamHoc/sentiment-mental-health/sentiment-mental-health.csv")

    #df.status, df.statement
    
    from dataset import CustomDataset
    dataset_obj = CustomDataset(df.statement, df.status, 100)
    ds_0 = dataset_obj.__getitem__(0)
    print(ds_0)
    
    print(dataset_obj.__len__())
    
#test_dataset()

#Testing the Input Embeddings

def test_input_embed():
    from model import InputEmbedding
    #import torch
    #from tokenizer import Tokenizer
    #tokenizer = Tokenizer()
    #test_sentence = 'Hello, How are you?'
    tokenized_input, tokenizer = test_tokenizer()
    
    embedding_layer = InputEmbedding(128,tokenizer.get_vocabsize(), 0)
    input_embedding = embedding_layer(tokenized_input['input_ids'])
    print(tokenized_input['input_ids'].unsqueeze(-1).shape)
    print(input_embedding.shape)
    return input_embedding
    
#input_embedding = test_input_embed()
#print(input_embedding)