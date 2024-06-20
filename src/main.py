# main.py
from model import TransformerRegression
from train import train_transformer
from prepare_data import prepare_data_for_transformer
from transformers import BertTokenizer

# Example usage
model_name = 'bert-base-uncased'
num_features = 1
learning_rate = 0.0001  # Reduced learning rate
num_epochs = 50
lambda_reg = 0.01
clip_value = 1.0
verbose = True

tokenizer = BertTokenizer.from_pretrained(model_name)
X_train, attention_mask, y_train, train_loader, val_loader = prepare_data_for_transformer(tokenizer)

model = TransformerRegression(model_name, num_features)
train_transformer(model, train_loader, val_loader, num_epochs, learning_rate, lambda_reg, clip_value, verbose, early_stopping_patience=10)
