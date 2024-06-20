# hyperparameter_tuning.py
import optuna
from model import TransformerRegression
from train import train_transformer, validate_model
from prepare_data import prepare_data_for_transformer
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_loader, val_loader = prepare_data_for_transformer(tokenizer)

def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    lambda_reg = trial.suggest_loguniform('lambda_reg', 1e-5, 1e-1)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    clip_value = trial.suggest_uniform('clip_value', 0.5, 2.0)

    model = TransformerRegression(model_name='bert-base-uncased', num_features=1, dropout_rate=dropout_rate)
    train_transformer(model, train_loader, val_loader, num_epochs=10, learning_rate=learning_rate, lambda_reg=lambda_reg, clip_value=clip_value, verbose=False)
    
    val_loss = validate_model(model, val_loader)
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print('Best hyperparameters:', study.best_params)
