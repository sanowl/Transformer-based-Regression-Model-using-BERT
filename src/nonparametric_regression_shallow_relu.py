import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments
from datasets import Dataset

class ShallowReLUNetwork(nn.Module):
    def __init__(self, input_dim: int, width: int):
        super(ShallowReLUNetwork, self).__init__()
        self.linear = nn.Linear(input_dim, width)
        self.activation = nn.ReLU()
        self.output = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)
        return self.output(x)

def kappa_norm(model: nn.Module) -> torch.Tensor:
    norm = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            norm += torch.sum(torch.abs(param) * torch.norm(param, p=2, dim=0))
    return norm

def holder_function(x: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return np.sum(np.abs(x) ** alpha, axis=1)

def generate_dataset(n_samples: int, input_dim: int, alpha: float = 0.5, noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    X = np.random.rand(n_samples, input_dim) * 2 - 1
    y = holder_function(X, alpha) + noise_level * np.random.randn(n_samples)
    return X, y

def local_rademacher_complexity(model: nn.Module, X: torch.Tensor, delta: float, n_samples: int = 1000) -> float:
    rademacher = torch.randint(0, 2, (n_samples, X.shape[0])).float() * 2 - 1
    complexity = 0
    for r in rademacher:
        outputs = model(X)
        complexity += torch.abs(torch.mean(r * outputs))
    return complexity.item() / n_samples

def constrained_optimization(model: nn.Module, X: torch.Tensor, y: torch.Tensor, M: float, num_epochs: int = 1000, lr: float = 0.01) -> nn.Module:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        
        kappa = kappa_norm(model)
        if kappa > M:
            loss += kappa - M
        
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Kappa: {kappa.item():.4f}")

    return model

def train_and_evaluate(X: np.ndarray, y: np.ndarray, input_dim: int, width: int, M: float) -> Tuple[nn.Module, float, float]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = torch.FloatTensor(X_train), torch.FloatTensor(X_test)
    y_train, y_test = torch.FloatTensor(y_train).unsqueeze(1), torch.FloatTensor(y_test).unsqueeze(1)

    model = ShallowReLUNetwork(input_dim, width)
    model = constrained_optimization(model, X_train, y_train, M)

    with torch.no_grad():
        y_pred = model(X_test)
        mse = nn.MSELoss()(y_pred, y_test).item()
        
    complexity = local_rademacher_complexity(model, X_test, delta=0.1)

    return model, mse, complexity

def plot_results(X: np.ndarray, y: np.ndarray, model: nn.Module):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], y, alpha=0.5, label='True')
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X)).numpy()
    plt.scatter(X[:, 0], y_pred, alpha=0.5, label='Predicted')
    plt.xlabel('First feature')
    plt.ylabel('Target')
    plt.legend()
    plt.title('True vs Predicted')

    plt.subplot(1, 2, 2)
    plt.scatter(y, y_pred, alpha=0.5)
    plt.xlabel('True values')
    plt.ylabel('Predicted values')
    plt.title('Prediction Scatter Plot')

    plt.tight_layout()
    plt.show()

def create_hf_dataset(X: np.ndarray, y: np.ndarray) -> Dataset:
    text_data = [' '.join(map(str, row)) for row in X]
    return Dataset.from_dict({"text": text_data, "label": y})

def train_and_evaluate_hf_model(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_dataset = create_hf_dataset(X_train, y_train)
    test_dataset = create_hf_dataset(X_test, y_test)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=1)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
    )

    trainer.train()

    eval_results = trainer.evaluate()
    mse = eval_results['eval_loss']

    X_test_tensor = torch.FloatTensor(X_test)
    complexity = local_rademacher_complexity(model, X_test_tensor, delta=0.1)

    return mse, complexity

def main():
    input_dim = 10
    width = 1000  # Over-parameterized
    M = 10
    n_samples = 1000
    alpha = 0.75

    X, y = generate_dataset(n_samples, input_dim, alpha)
    
    # Train and evaluate our shallow ReLU network
    model, mse, complexity = train_and_evaluate(X, y, input_dim, width, M)

    print("Shallow ReLU Network Results:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Local Rademacher Complexity: {complexity:.4f}")
    print(f"Final Kappa Norm: {kappa_norm(model).item():.4f}")

    plot_results(X, y, model)

    # Train and evaluate the Hugging Face model
    print("\nTraining Hugging Face DistilBERT model...")
    hf_mse, hf_complexity = train_and_evaluate_hf_model(X, y)

    print("\nHugging Face DistilBERT Results:")
    print(f"Mean Squared Error: {hf_mse:.4f}")
    print(f"Local Rademacher Complexity: {hf_complexity:.4f}")

    # Compare results
    print("\nComparison:")
    print(f"Shallow ReLU Network MSE: {mse:.4f}")
    print(f"DistilBERT MSE: {hf_mse:.4f}")
    print(f"Shallow ReLU Network Complexity: {complexity:.4f}")
    print(f"DistilBERT Complexity: {hf_complexity:.4f}")

if __name__ == "__main__":
    main()