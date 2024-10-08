# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 07:46:16 2024

@author: Arun Joshi
"""
from torch import nn 
import math
#import numpy as np
import torch
class InputEmbedding(nn.Module):
    def __init__(self, d_model:int, vocab_size:int, padding_idx:int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.embedding_layer = nn.Embedding(self.vocab_size, self.d_model, self.padding_idx)
    def forward(self, x):
        input_embedding = self.embedding_layer(x)
        return input_embedding * math.sqrt(self.d_model) 
    '''Following the attention is all you need paper for the implemention of 
        math.sqrt(d_model) in the input_embedding. 
        Link for the page is  : https://arxiv.org/pdf/1706.03762 -> Page no 5.'''
        
'''
class PostionalEncoding(nn.Module):
    def __init__(self, seq_len:int, d_model:int, dropout:float = 0.1 ,n:int = 10000):
        super().__init__()
        self.dropout = dropout
        self.d_model = d_model
        self.seq_len = seq_len
        self.n = n
        self.dropout_layer = nn.Dropout(self.dropout)
        
    
    def forward(self,x):
        P = np.zeros((self.seq_len, self.d_model))
        for k in range(self.seq_len):
            for i in np.arange(int(self.d_model/2)):
                denominator = np.power(self.n, 2*i/self.d_model)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        P = torch.from_numpy(P)
        P = P.expand(x.shape(0), self.seq_len, self.d_model)
        x = x + P
        return self.dropout(x)
'''

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len:int,dropout: float = 0.1, ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) #-> (max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model) #-> (max_len,1,d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, h:int, d_model:int):
        super().__init__()
        self.h = h
        self.d_model = d_model
        self.d_k = d_model // h
        assert d_model % h == 0, 'Make sure d_model is divisible by h'
        self.linear_layer1 = nn.Linear(d_model, d_model, bias = False)
        self.linear_layer2 = nn.Linear(d_model, d_model, bias = False)
        self.linear_layer3 = nn.Linear(d_model, d_model, bias = False)
        self.linear_layer4 = nn.Linear(d_model,d_model, bias = False)
    
    def forward(self,q,k,v,src_mask):
        query = self.linear_layer1(q) # (Batch, seq_len, d_model)
        key = self.linear_layer2(k) # (Batch, seq_len, d_model)
        value = self.linear_layer3(v) # (Batch, seq_len, d_model)
        
        #We need to reshape the q, k,v tensors in the shape (Batch, h, seq_len, d_k) first
        # Since d_k * h == d_model, we can reshape our tensor as (Batch, seq_len, h, d_k) and rearragne the dimensions
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(value.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        
        
        x = self.attention_block(query, key, value, src_mask)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        return self.linear_layer4(x)
        
    def attention_block(self, query, key, value, mask):
        #the masking will also be of shape (batch,h,seq_len,seq_len)
        attention_scores = torch.matmul(query, key.transpose(-2,-1))/math.sqrt(self.d_k)
        #Attention_scores will be of (batch_size, h, seq_len,seq_len) -> (1,4,100,100)
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e-9)
        attention_scores = torch.softmax(attention_scores, dim = -1)
        return attention_scores @ value # -> (1, head, seq_len, d_k)


class LayerNormalization(nn.Module):
    def __init__(self):
        pass
    
    def forward(self):
        pass
    
class ResidualConnection(nn.Module):
    def __init__(self):
        pass
    
    def forward(self):
        pass
        
class FeedForwardNet(nn.Module):
    def __init__(self):
        pass
    
    def forward(self):
        pass
    
class ProjectionLayer(nn.Module):
    def __init__(self):
        pass
    
    def forward(self):
        pass
        
        
    