# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 07:46:16 2024

@author: Arun Joshi
"""
from torch import nn 
import math
import numpy as np
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