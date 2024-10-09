# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 19:48:01 2024

@author: LENOVO
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


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
# Define the training loop
def train():
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Before encoding:
    train_dataloader, test_dataloader = dataloader()
    model = SentimentModel(4, 128, 512, 6).to(device)  # Keep 6 classes
    optimizer = Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    epoches = 5
    progress_bar = tqdm(total=epoches, desc='Processing')
    
    for epoch in range(epoches):
        correct_predictions = 0
        loss_sum = 0
        
        for index, test_data in enumerate(train_dataloader):
            input_data = test_data['input_ids'].to(device)
            target = test_data['label'].to(device)  # Move target to the device
            mask = test_data['input_mask_ids'].transpose(-1, -2).to(device)
            
            # Ensure attention mask is calculated correctly
            attn_mask = (mask @ mask.transpose(-1, -2)).unsqueeze(1).expand(-1, 4, -1, -1)

            # Forward pass
            output = model(input_data, attn_mask)

            # Check unique target labels
            print("Unique target labels:", target.unique())
            
            # Check for out-of-bounds labels
            if target.max() >= 4:  # Ensure target labels are in range [0, 3]
                raise ValueError(f"Target labels must be in the range [0, 3], but found {target.max().item()}")

            # Calculate loss
            loss = loss_fn(output, target)
            loss.backward()
            loss_sum += loss.item()

            # Update weights
            optimizer.step()
            actual_output = torch.argmax(output, dim=1)
            correct_predictions += (actual_output == target).sum().item()
            optimizer.zero_grad()  # Reset gradients
            
            # Log progress
            if (index % 1000 == 0):
                print(f'Epoch: {epoch}/{epoches}, Batch {index} / len(train_dataloader) -> Loss: {loss.item()}')

        # Calculate and print epoch statistics
        print(f'Loss in epoch {epoch}: {loss_sum / len(train_dataloader)}')
        print(f'Epoch {epoch} -> Accuracy: {correct_predictions / len(train_dataloader)}')
        progress_bar.update(1)
    
    progress_bar.close()
    return output, target

output, target = train()
