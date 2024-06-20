import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertConfig, BertTokenizer

# Define the transformer-based regression model with regularization
class TransformerRegression(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_features=1):
        super(TransformerRegression, self).__init__()
        self.config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.config.hidden_size, num_features)
    
    def forward(self, x):
        outputs = self.bert(x)
        pooled_output = outputs[1]  # Use the pooled output (the first token's hidden state)
        x = self.fc(pooled_output)
        return x

# Function to compute the regularization term
def compute_regularization(model, lambda_reg):
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param**2)
    return lambda_reg * reg_loss

# Function to train the transformer model
def train_transformer(model, train_loader, num_epochs, learning_rate, lambda_reg):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            reg_loss = compute_regularization(model, lambda_reg)
            total_loss = loss + reg_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Data Preparation
def prepare_data_for_transformer(tokenizer):
    # Example dataset for transformer
    texts = ["This is a sentence.", "This is another sentence.", "Yet another sentence.", "Sentence number four.", "The last sentence."]
    X_train = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")["input_ids"]
    y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    return train_loader

# Example usage
model_name = 'bert-base-uncased'
num_features = 1
learning_rate = 0.001
num_epochs = 10
lambda_reg = 0.01

tokenizer = BertTokenizer.from_pretrained(model_name)
train_loader = prepare_data_for_transformer(tokenizer)

model = TransformerRegression(model_name, num_features)
train_transformer(model, train_loader, num_epochs, learning_rate, lambda_reg)
