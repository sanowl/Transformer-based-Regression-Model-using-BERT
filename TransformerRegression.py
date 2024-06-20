import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertConfig, BertTokenizer
import numpy as np
import matplotlib.pyplot as plt

class TransformerRegression(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_features=1, dropout_rate=0.1):
        super(TransformerRegression, self).__init__()
        self.config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.config.hidden_size, num_features)
    
    def forward(self, x):
        outputs = self.bert(x)
        pooled_output = outputs[1]  # Use the pooled output (the first token's hidden state)
        x = self.dropout(pooled_output)
        x = self.fc(x)
        return x

def compute_regularization(model, lambda_reg):
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param**2)
    return lambda_reg * reg_loss

def train_transformer(model, train_loader, val_loader, num_epochs, learning_rate, lambda_reg, clip_value, verbose):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            reg_loss = compute_regularization(model, lambda_reg)
            total_loss = loss + reg_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            
            epoch_loss += total_loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_losses.append(val_loss / len(val_loader))

        if verbose:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    # Plotting loss curves
    plt.figure()
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def prepare_data_for_transformer(tokenizer):
    texts = ["This is a sentence.", "This is another sentence.", "Yet another sentence.", "Sentence number four.", "The last sentence."]
    X_train = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")["input_ids"]
    y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    # Split the dataset into train and validation
    val_dataset = TensorDataset(X_train, y_train)  # Using the same data for simplicity; replace with actual validation set
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    return train_loader, val_loader

# Example usage
model_name = 'bert-base-uncased'
num_features = 1
learning_rate = 0.001
num_epochs = 50
lambda_reg = 0.01
clip_value = 1.0
verbose = True

tokenizer = BertTokenizer.from_pretrained(model_name)
train_loader, val_loader = prepare_data_for_transformer(tokenizer)

model = TransformerRegression(model_name, num_features)
train_transformer(model, train_loader, val_loader, num_epochs, learning_rate, lambda_reg, clip_value, verbose)
