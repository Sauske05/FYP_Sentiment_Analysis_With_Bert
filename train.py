# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 06:53:17 2024

@author: Arun Joshi
"""

from configure import config
from model import InputEmbedding
from model import PositionalEncoding
from data import dataloader
from tokenizer import Tokenizer
from model import SentimentModel
from torch.optim import Adam
import torch.nn as nn
import torch
from tqdm import tqdm
#Define the encoding layer
'''
def encode():
    #Load our dataloaders
    tokenizer = Tokenizer()
    positional_layer = PositionalEncoding(128,100)
    embedding_layer = InputEmbedding(128, tokenizer.get_vocabsize(), 0)
    train_dataloader, test_dataloader = dataloader()
    for data in train_dataloader:
        data['input_ids'] = embedding_layer(data['input_ids'].transpose(-1,-2))
        data['input_ids'] = positional_layer(data['input_ids'].squeeze()
                                             ) # -> (Batch, seq_len, 1, d_model) -> (Batch, seq_len, d_model)
        #print(data['input_ids'].shape)
    return train_dataloader
'''
#train_dataloader1 = encode()

        
#Definine the training loop
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Before encoding:
    train_dataloader, test_datatloader = dataloader()
    model = SentimentModel(4,128,512,4)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr = 0.01)
    optimizer = optimizer
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn
    #for data in train_dataloader:
        #test_data = data
        #break
    #After encode
    epoches = 5
    progress_bar = tqdm(total = epoches, desc = 'Processing')
    for epoch in range(epoches):
        correct_predictions = 0
        loss_sum = 0
        for index, test_data in enumerate(train_dataloader):
            input_data = test_data['input_ids'].to(device)
            target = test_data['label'].to(device)
            mask = test_data['input_mask_ids'].transpose(-1,-2)
    #print(f'Mask shape: {mask.shape}')
    #print(f'Mask shape : {mask.T.shape}')
            attn_mask = ((mask@
                      mask.transpose(-1,-2)).unsqueeze(1).expand(-1, 4, -1, -1)).to(device)
    
    
            output = model(input_data, attn_mask)
            #target =targe.to(device)
    
            loss = loss_fn(output, target)
            loss.backward()
            loss_sum += loss
    
            optimizer.step()
            actual_output = torch.argmax(output, dim = 1)
            correct_predictions += (actual_output == target).sum().item()
            optimizer.zero_grad()
            
            if (index % 1000 == 0):
                print(f'Epoch : {epoch}/{epoches}, Batch {index} / len(train_dataloader) -> Loss : {loss.item()}')
        print(f'Loss in epoch {epoch} : {loss_sum/ len(train_dataloader)}')
        print(f'Epoch {epoch} -> Accuracy : {correct_predictions/len(train_dataloader)}')
        progress_bar.update(1)
    progress_bar.close()
    return output, target


output, target = train()
print(output)
print(target)

    
    
    
    
    
