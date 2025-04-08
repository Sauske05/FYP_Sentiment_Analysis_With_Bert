# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 06:53:17 2024

@author: Arun Joshi
"""

#import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import mlflow


#mlflow.set_tracking_uri('http://localhost:5000')
#mlflow.set_experiment('Sentiment_Classification_First_Log')

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentModel(config()['h'], config()['d_model'], config()['d_ff'], config()['labels']).to(device)
def train():

    # Before encoding:
    train_dataloader, test_dataloader = dataloader()

    optimizer = Adam(model.parameters(), lr=1e-5)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    epoches = 20
    progress_bar_epoch = tqdm(total=epoches, desc='Processing')

    # Start a new MLflow run
    with mlflow.start_run():
        # Log model parameters as MLflow parameters
        mlflow.log_param("learning_rate", 1e-5)
        mlflow.log_param("epochs", epoches)
        mlflow.log_param("batch_size", len(train_dataloader.dataset) // len(train_dataloader))
        
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

                attn_mask = ((mask @ mask.transpose(-1,-2)).unsqueeze(1).expand(-1, 4, -1, -1)).to(device)
                output = model(input_data, attn_mask)

                loss = loss_fn(output, target)
                loss.backward()
                loss_sum += loss

                optimizer.step()
                actual_output = torch.argmax(output, dim=1)
                correct_predictions += (actual_output == target).sum().item()
                optimizer.zero_grad()
                
                if index % 1000 == 0:
                    print(f'Epoch : {epoch+1}/{epoches}, Batch {index} / {len(train_dataloader)} -> Loss : {loss.item()}')

            # Log training metrics to MLflow
            mlflow.log_metric("train_loss", loss_sum / len(train_dataloader), step = epoch)
            mlflow.log_metric("train_accuracy", correct_predictions / (len(train_dataloader) * 4), step = epoch)

            # Validation loop
            model.eval()
            with torch.no_grad():
                for index, test_data in enumerate(test_dataloader):
                    X_test = test_data['input_ids'].to(device)
                    y_test = test_data['label'].to(device)

                    test_mask = test_data['input_mask_ids'].transpose(-1,-2)
                    test_attn_mask = ((test_mask @ test_mask.transpose(-1,-2)).unsqueeze(1).expand(-1,4,-1,-1)).to(device)
                    
                    test_output = model(X_test, test_attn_mask)
                    test_loss = loss_fn(test_output, y_test)
                    test_loss_sum += test_loss
                    
                    actual_test_output = torch.argmax(test_output, dim=1)
                    test_predictions += (actual_test_output == y_test).sum().item()

                    if index % 1000 == 0:
                        print(f'Epoch : {epoch+1}/{epoches}, Batch {index} / {len(test_dataloader)} -> Test Loss : {test_loss.item()}')

            # Log test metrics to MLflow
            mlflow.log_metric("test_loss", test_loss_sum / len(test_dataloader), step = epoch)
            mlflow.log_metric("test_accuracy", test_predictions / (len(test_dataloader) * 4), step = epoch)

            print(f'Train Loss in epoch {epoch+1} : {loss_sum/len(train_dataloader)}')
            print(f'Test Loss in epoch {epoch+1} : {test_loss_sum/len(test_dataloader)}')
            print(f'Epoch {epoch+1} -> Train Accuracy : {correct_predictions/(len(train_dataloader) * 4)}')
            print(f'Epoch {epoch+1} -> Test Accuracy : {test_predictions/(len(test_dataloader) * 4)}')
            progress_bar_epoch.update(1)

        progress_bar_epoch.close()

        # Log the trained model to MLflow
        mlflow.pytorch.log_model(model, "sentiment_model")
        mlflow.pytorch.log_artifact(model.state_dict(), 'sentiment_model_state_dict')

    return output, target

# output, target = train()
# print(output)
# print(target)
# torch.save(model.state_dict(), 'model_state_dict.pth')

    
train_dataloader, test_dataloader = dataloader()

for index, train_data in enumerate(train_dataloader):
    input_data = train_data['input_ids'].to(device)
    target = train_data['label'].to(device)
    mask = train_data['input_mask_ids'].transpose(-1, -2)
    print(mask.size())
    attn_mask = ((mask @ mask.transpose(-1, -2)).unsqueeze(1).expand(-1, 4, -1, -1)).to(device)
    print(attn_mask.size())
    print(input_data.size())
    print(target.size())
    break
    
    
