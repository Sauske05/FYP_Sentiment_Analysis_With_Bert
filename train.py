# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 06:53:17 2024

@author: Arun Joshi
"""

#import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


from configure import config
#from model import InputEmbedding
#from model import PositionalEncoding
from data import dataloader
#from tokenizer import Tokenizer
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
    #device = torch.device('cpu')
    #Before encoding:
    train_dataloader, test_dataloader = dataloader()
    model = SentimentModel(config()['h'],config()['d_model'],config()['d_ff'],config()['labels']).to(device)
    optimizer = Adam(model.parameters(), lr = 1e-5)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    #for data in train_dataloader:
        #test_data = data
        #break
    #After encode
    epoches = 5
    progress_bar_epoch = tqdm(total = epoches, desc = 'Processing')
    #progress_bar_batch = tqdm(total = len(train_dataloader), desc  = 'Batch Process')
    for epoch in range(epoches):
        correct_predictions = 0
        loss_sum = 0
        test_loss_sum = 0
        test_predictions = 0
        model.train()
        for index, train_data in enumerate(train_dataloader):
            input_data = train_data['input_ids'].to(device)
            target = train_data['label'].to(device)
            mask = train_data['input_mask_ids'].transpose(-1,-2)
    #print(f'Mask shape: {mask.shape}')
    #print(f'Mask shape : {mask.T.shape}')
            attn_mask = ((mask@
                      mask.transpose(-1,-2)).unsqueeze(1).expand(-1, 4, -1, -1)).to(device)
    
            #attn_mask shape : -> (Batch, head, seq_len, seq_len)
            output = model(input_data, attn_mask)
            #target =torch.tensor([1,2])
    
            loss = loss_fn(output, target)
            loss.backward()
            loss_sum += loss
            #print("Output shape:", output.shape)
            #print("Target shape:", target.shape)

            optimizer.step()
            actual_output = torch.argmax(output, dim = 1)
            correct_predictions += (actual_output == target).sum().item()
            optimizer.zero_grad()
            
            if (index % 1000 == 0):
                print(f'Epoch : {epoch+1}/{epoches}, Batch {index} / {len(train_dataloader)} -> Loss : {loss.item()}')
            #progress_bar_batch.update(1)
        #Validation loop
        model.eval()
        with torch.no_grad():
            for index, test_data in enumerate(test_dataloader):
                X_test = test_data['input_ids'].to(device)
                y_test = test_data['label'].to(device)
                
                test_mask = test_data['input_mask_ids'].transpose(-1,-2)
                test_attn_mask = ((test_mask @
                                   test_mask.transpose(-1,-2)).unsqueeze(1).expand(-1,4,-1,-1)).to(device)
                
                test_output = model(X_test, test_attn_mask)
                
                test_loss = loss_fn(test_output, y_test)
                
                test_loss_sum += test_loss
                
                actual_test_output = torch.argmax(test_output, dim = 1)
                test_predictions += (actual_test_output == y_test).sum().item()
                
                if (index % 1000 == 0):
                    print(f'Epoch : {epoch+1}/{epoches}, Batch {index} / {len(test_dataloader)} -> Test Loss : {test_loss.item()}')
                
        print(f'Train Loss in epoch {epoch+1} : {loss_sum/ len(train_dataloader)}')
        print(f'Test Loss in epoch {epoch+1} : {test_loss_sum/ len(test_dataloader)}')
        #progress_bar_batch.close()
        print(f'Epoch {epoch+1} -> Train Accuracy : {correct_predictions/ (len(train_dataloader) * 4)}')
        print(f'Epoch {epoch+1} -> Test Accuracy : {test_predictions/ (len(test_dataloader) * 4)}')
        progress_bar_epoch.update(1)
    progress_bar_epoch.close()
    return output, target


output, target = train()
print(output)
print(target)

    
    
    
    
    
