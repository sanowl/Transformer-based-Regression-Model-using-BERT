import numpy as np
from sklearn.model_selection import KFold
from model import TransformerRegression
from train import train_transformer, validate_model
from prepare_data import prepare_data_for_transformer
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset

# Initialize the tokenizer and prepare the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
X_train, y_train, _, _ = prepare_data_for_transformer(tokenizer)

# Function to create data loaders for train and validation sets
def data_loader_func(train_idx, val_idx):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]
    
    train_dataset = TensorDataset(X_train_fold, y_train_fold)
    val_dataset = TensorDataset(X_val_fold, y_val_fold)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    return train_loader, val_loader

# Function to perform cross-validation
def cross_validate_model(model_class, data_loader_func, k=5):
    kfold = KFold(n_splits=k, shuffle=True)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        train_loader, val_loader = data_loader_func(train_idx, val_idx)
        model = model_class().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Use gradient accumulation to simulate larger batch sizes
        train_transformer(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, lambda_reg=0.01, clip_value=1.0, verbose=True, accumulation_steps=4)
        
        val_loss = validate_model(model, val_loader)
        fold_results.append(val_loss)
        print(f'Fold {fold + 1}, Validation Loss: {val_loss:.4f}')
    
    avg_loss = np.mean(fold_results)
    print(f'Average Validation Loss: {avg_loss:.4f}')

# Execute cross-validation
cross_validate_model(TransformerRegression, data_loader_func, k=5)
