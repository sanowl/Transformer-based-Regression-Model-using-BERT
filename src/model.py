import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class TransformerRegression(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_features=1, dropout_rate=0.1):
        super(TransformerRegression, self).__init__()
        self.config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.config.hidden_size, num_features)
        
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.to(self.device)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Use the pooled output (the first token's hidden state)
        x = self.dropout(pooled_output)
        x = self.fc(x)
        return x
