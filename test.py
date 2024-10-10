# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 07:44:09 2024

@author: Arun Joshi
"""


# Testing the tokenizer class
def test_tokenizer():
    '''
    A dictionary is created with key values of 'input_ids',
    'token_type_ids', 'attention_mask'
    '''
    from tokenizer import Tokenizer

    tokenizer_obj = Tokenizer()
    test_sentence = ['Hello, How are you?', 'I am fine']

    tokenized_input = tokenizer_obj.tokenize(test_sentence, 100)
    return tokenized_input, tokenizer_obj
    # print(tokenized_input)
'''
tokenized_input, _ = test_tokenizer()
text = tokenized_input['input_ids']
print(f'Inital data: {text.shape}')
'''

# Testing the dataset class

def test_dataset():
    import pandas as pd

    df = pd.read_csv(
        "hf://datasets/TungHamHoc/sentiment-mental-health/sentiment-mental-health.csv")

    # df.status, df.statement

    from dataset import CustomDataset
    dataset_obj = CustomDataset(df.statement, df.status, 100)
    ds_0 = dataset_obj.__getitem__(0)
    print(ds_0)

    # print(dataset_obj.__len__())
    return ds_0


ds = test_dataset()

# Testing the Input Embeddings


def test_input_embed():
    from model import InputEmbedding
    # import torch
    # from tokenizer import Tokenizer
    # tokenizer = Tokenizer()
    # test_sentence = 'Hello, How are you?'
    tokenized_input, tokenizer = test_tokenizer()

    embedding_layer = InputEmbedding(128, tokenizer.get_vocabsize(), 0)
    input_embedding = embedding_layer(ds['input_ids'])
    # print(tokenized_input['input_ids'].unsqueeze(-1).shape)
    # print(input_embedding.shape)
    return input_embedding

# input_embedding = test_input_embed()
# print(input_embedding)


# Testing Postional Encoding
def test_postional_layer(input_embed):
    from model import PositionalEncoding
    from configure import config
    import torch
    positional_layer = PositionalEncoding(128, config()['seq_length'])

    #test_data = torch.randn(1, 100, 128)

    pos_embeded_x = positional_layer(input_embed)

    return pos_embeded_x

# pos_embed_x = test_postional_layer()

# print(f'Shape of pos_embed_x : {pos_embed_x.shape}')
# print(pos_embed_x)

# Testing MultiHead Attention:


def test_multiheadAttention():
    from model import MultiHeadAttention
    from configure import config
    import torch
    test_data = torch.randn(1, 100, 128)

    attn_mask = ((ds['input_mask_ids'].T @
                 ds['input_mask_ids']).expand(1, 4, 100, 100))
    attention_layer = MultiHeadAttention(config()['h'], config()['d_model'])

    return attention_layer(test_data, test_data, test_data, attn_mask)


# contextual_x = test_multiheadAttention()

# print(f'Shape of the contextual tensor : {contextual_x.shape}')
# print(contextual_x)


# Testing Residual Connection Layer for Attention:
def residual_connection():
    from model import ResidualConnection
    from configure import config
    from model import MultiHeadAttention

    import torch

    test_data = torch.randn(1, 100, 128)
    attn_mask = ((ds['input_mask_ids'].T @
                 ds['input_mask_ids']).expand(1, 4, 100, 100))
    multi_head_layer = MultiHeadAttention(config()['h'], config()['d_model'])

    res_layer = ResidualConnection()

    output = res_layer(test_data, lambda test_data: multi_head_layer(
        test_data, test_data, test_data, attn_mask))

    return output


#output = residual_connection()

#print(f'Shape of the output after residual connection via multi head: {output.shape}')


#Testing Residual Connection for Feed Forward Layer:
def res_connection_feed_fr():
    from model import ResidualConnection
    from configure import config
    from model import FeedForwardNet
    
    import torch
    
    x = torch.randn(1,100,128)
    
    res_layer = ResidualConnection()
    
    feed_forward_layer  = FeedForwardNet(config()['d_model'], config()['d_ff'])
    
    return res_layer(x, feed_forward_layer)


#output = res_connection_feed_fr()

#print(f'Shape of Feed Forward Res Layer: {output.shape}')


#Testing for softmax
def softmax():
    from torch import nn as nn
    import torch
    softmax_layer = nn.Softmax(dim = -1)
    x = torch.randn(3,5)
    print(x)
    return softmax_layer(x)
    
#val = softmax()
#print(val)

#Testing for Projection Layer:
def projection_layer():
    from model import ProjectionLayer
    proj_layer = ProjectionLayer(128,4)
    import torch
    x = torch.randn(1,100,128)
    
    final_label = proj_layer(x)
    return final_label


'''
final_labels = projection_layer()

print(final_labels)

import torch
final_class = torch.argmax(final_labels, dim = 1)
print(final_class.item()) 
'''


#Testing the sentiment_model:
    
def sentiment_model():
    #h:int, d_model:int,d_ff:int,unique_labels:int, input_mask ,input_embeddings,
    from model import SentimentModel
    from configure import config
    input_embedding_initial = test_input_embed()
    input_embedding = test_postional_layer(input_embedding_initial)
    
    mask = ((ds['input_mask_ids'].T @
                 ds['input_mask_ids']).expand(1, 4, 100, 100))
    sentiment_obj = SentimentModel(config()['h'], config()['d_model'], 
                                   config()['d_ff'], config()['labels'])
    return sentiment_obj(input_embedding, mask)

output = sentiment_model()

print(output)



def cuda_test():
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #return device
    print(device)
    
#cuda_test()
