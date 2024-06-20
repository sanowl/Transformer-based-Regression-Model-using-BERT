import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def compute_regularization(model, lambda_reg):
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param**2)
    return lambda_reg * reg_loss

def validate_model(model, val_loader):
    model.eval()
    val_loss = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for inputs, attention_mask, targets in val_loader:
            inputs, attention_mask, targets = inputs.to(model.device), attention_mask.to(model.device), targets.to(model.device)
            outputs = model(inputs, attention_mask)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    return val_loss / len(val_loader)

def train_transformer(model, train_loader, val_loader, num_epochs, learning_rate, lambda_reg, clip_value, verbose, accumulation_steps=1, early_stopping_patience=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    patience = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        for i, (inputs, attention_mask, targets) in enumerate(train_loader):
            inputs, attention_mask, targets = inputs.to(model.device), attention_mask.to(model.device), targets.to(model.device)
            outputs = model(inputs, attention_mask)
            loss = criterion(outputs, targets)
            reg_loss = compute_regularization(model, lambda_reg)
            total_loss = loss + reg_loss
            
            total_loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += total_loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        
        # Validation
        val_loss = validate_model(model, val_loader)
        val_losses.append(val_loss)

        if verbose:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')  # Ensure this filename is consistent
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                print("Early stopping triggered")
                break

    # Plotting loss curves
    plt.figure()
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_validation_loss.png')  # Save the plot instead of showing it
    plt.close()
