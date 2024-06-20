import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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
      model,train()
      epoch_loss = 0.0
     for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            reg_loss = compute_regularization(model, lambda_reg)
            total_loss = loss + reg_loss
      
    
    
  

  
  