# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 06:55:06 2024

@author: LENOVO
"""

import pandas as pd
from torch.utils.data import DataLoader
from dataset import CustomDataset
from sklearn.model_selection import train_test_split
#from tqdm import tqdm


def clean_data():
    df1 = pd.read_csv(
        "hf://datasets/TungHamHoc/sentiment-mental-health/sentiment-mental-health.csv")
    df2 = pd.read_csv("hf://datasets/AhmedSSoliman/sentiment-analysis-for-mental-health-Combined-Data/sentiment-analysis-for-mental-health-Combined Data.csv")

    df = pd.concat([df1,df2])
    df = df.drop(columns = ['Unnamed: 0'], axis = 1)
    print(f'Shape of df before dropping null val: {df.shape}')
    df = df.dropna()
    df.to_csv('clean_data.csv')

#clean_data()
#import pandas as pd
def raw_data():
    df = pd.read_excel('mod_data.xlsx')
    df = df.dropna()
    # Find the indices of rows where 'status' equals 'Anxiety' or 'Personality disorder'
    #rows_to_drop = df[(df['status'] == 'Anxiety') | (df['status'] == 'Personality disorder')].index

    # Drop those rows from the DataFrame
    # = df.drop(rows_to_drop)
    print(f'Shape of df after dropping null val: {df.shape}')
    raw_text = df['statement']
    labels = df['status']
    
    label_dictionary = {}
    for index, label in enumerate(labels.unique()):
        label_dictionary[label] = index

    labels =labels.map(label_dictionary)
    print('Label Dict of the data.py file:')
    print(label_dictionary)
    return raw_text, labels
    

raw_texts, labels= raw_data()

print(f'Shape of raw_texts: {raw_texts.shape}')
print(f'Shape of labels: {labels.shape}')
print(f'Unique values of labels : {labels.value_counts()}')
print(f'Null values of raw_text: {raw_texts.isnull().sum()}')
print(f'Null values of labels : {labels.isnull().sum()}')


def dataloader():
    raw_texts, labels  = raw_data()
    X_train, X_test, y_train, y_test = train_test_split(
        raw_texts, labels, random_state = 42, test_size=0.2)
    # Reset indices to avoid KeyError issues during access by index
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    train_dataset = CustomDataset(X_train, y_train, 100)
    test_dataset = CustomDataset(X_test, y_test, 100)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    return train_dataloader, test_dataloader



train_dataloader, test_dataloader = dataloader()




'''
# Inspect the batches and their shapes
for i, batch_dict in enumerate(train_dataloader):
    print(batch_dict['input_ids'].transpose(-1,-2).shape) # -> Shape of (batch,seq_len, 1)
    print(batch_dict['inp_mask_ids'].transpose(-1,-2).shape)
    print(batch_dict['input_mask_ids'].transpose(-1,-2))
    print(batch_dict['label'])
    print(batch_dict['raw_text'])
    print(batch_dict['raw_text_len'])
    break
'''

'''
for index, batch in enumerate(train_dataloader):
    print(batch['input_ids'].shape[0])
    print(batch['label'].shape[0])
'''


#last_batch = list(train_dataloader)[-1]
#print(last_batch['input_ids'].shape)
#print(last_batch['label'].shape)