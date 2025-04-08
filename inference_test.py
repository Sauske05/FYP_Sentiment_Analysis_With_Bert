import torch
from typing import Dict
from model import SentimentModel
from configure import config

from tokenizer import Tokenizer
model = SentimentModel(config()['h'], config()['d_model'], config()['d_ff'], config()['labels']).to('cuda')
  # Your model architecture
PATH:str = 'model_state_dict.pth'
model.load_state_dict(torch.load(PATH, weights_only=True))
# Model class must be defined somewhere
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#model = torch.load('model.pth', weights_only=False).to(device)
model.eval()
print(model)

input = 'I am feeling sad!'
tokenizer_obj = Tokenizer()

token_input:Dict = tokenizer_obj.tokenize([input], 100)
print(token_input)
input_ids:torch.tensor= token_input['input_ids'].to(device)
attention_mask:torch.tensor  = token_input['attention_mask']
attention_mask = (attention_mask.T @ attention_mask).expand(1,4,100,100).to(device)
#print(input_ids.shape)
#print(attention_mask.shape)


y_pred = model(input_ids, attention_mask)
print(y_pred.shape)
print(y_pred)