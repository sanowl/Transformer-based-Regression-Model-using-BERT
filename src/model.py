import torch
import torch.nn as nn
from transformers import BertModel , BertConfig

class TransformerRegression(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_features=1, dropout_rate=0.1):
        super(TransformerRegression, self).__init__()
        self.config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.config.hidden_size, num_features)
    def forward(self,x):
      output = self.bert(x)
      pooled_output=output[1] # Use the pooled output (the first token's hidden state)
      x = self.dropout (pooled_output)
      x= self.fc(x)
      

