import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from transformers import BertTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from tqdm import tqdm

# Assuming these are defined in separate files
from model import TransformerRegression
from train import train_transformer, validate_model
from prepare_data import prepare_data_for_transformer

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Initialize the tokenizer and prepare the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
X_train, attention_mask, y_train, _, _ = prepare_data_for_transformer(tokenizer)

def data_loader_func(train_idx, val_idx, batch_size=16):
    X_train_fold = X_train[train_idx]
    attention_mask_fold = attention_mask[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    attention_mask_val_fold = attention_mask[val_idx]
    y_val_fold = y_train[val_idx]
    
    train_dataset = TensorDataset(X_train_fold, attention_mask_fold, y_train_fold)
    val_dataset = TensorDataset(X_val_fold, attention_mask_val_fold, y_val_fold)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_transformer_improved(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-4, lambda_reg=0.01, clip_value=1.0, verbose=True, accumulation_steps=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=lambda_reg)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, (inputs, masks, labels) in progress_bar:
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
            
            outputs = model(inputs, attention_mask=masks)
            loss = torch.nn.functional.mse_loss(outputs.squeeze(), labels)
            
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            
            progress_bar.set_postfix({'loss': total_loss / (i + 1)})
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        val_loss = validate_model(model, val_loader)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        if verbose:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

def cross_validate_model(model_class, data_loader_func, k=5, num_epochs=10, learning_rate=1e-4, lambda_reg=0.01, batch_size=16):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []
    all_train_losses = []
    all_val_losses = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        print(f"Fold {fold + 1}/{k}")
        
        train_loader, val_loader = data_loader_func(train_idx, val_idx, batch_size)
        model = model_class().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        train_losses, val_losses = train_transformer_improved(
            model, train_loader, val_loader, 
            num_epochs=num_epochs, learning_rate=learning_rate, 
            lambda_reg=lambda_reg, clip_value=1.0, verbose=True, 
            accumulation_steps=4
        )
        
        final_val_loss = val_losses[-1]
        fold_results.append(final_val_loss)
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        
        print(f'Fold {fold + 1}, Final Validation Loss: {final_val_loss:.4f}')
    
    avg_loss = np.mean(fold_results)
    std_loss = np.std(fold_results)
    print(f'Average Validation Loss: {avg_loss:.4f} ± {std_loss:.4f}')
    
    # Plot learning curves
    plt.figure(figsize=(12, 6))
    for fold in range(k):
        plt.plot(all_train_losses[fold], label=f'Fold {fold+1} Train')
        plt.plot(all_val_losses[fold], label=f'Fold {fold+1} Val')
    plt.title('Learning Curves for All Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Execute cross-validation
cross_validate_model(TransformerRegression, data_loader_func, k=5, num_epochs=10, learning_rate=1e-4, lambda_reg=0.01, batch_size=16)