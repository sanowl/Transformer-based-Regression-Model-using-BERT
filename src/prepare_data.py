# prepare_data.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer

def prepare_data_for_transformer(tokenizer):
    texts = ["This is a sentence.", "This is another sentence.", "Yet another sentence.", "Sentence number four.", "The last sentence."]
    X_train = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")["input_ids"]
    y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    # Split the dataset into train and validation
    val_dataset = TensorDataset(X_train, y_train)  # Using the same data for simplicity; replace with actual validation set
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    return X_train, y_train, train_loader, val_loader
