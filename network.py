import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the shallow ReLU neural network
class ShallowReLUNet(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super(ShallowReLUNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_units, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Function to train the model
def train_model(model, train_loader, num_epochs, learning_rate, regularization_strength):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization_strength)
    
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Data Preparation
def prepare_data():
    # Example dataset
    X_train = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    return train_loader

# Example usage
input_dim = 1  # Example input dimension
hidden_units = 10
learning_rate = 0.001
regularization_strength = 0.01
num_epochs = 100

train_loader = prepare_data()

model = ShallowReLUNet(input_dim, hidden_units)
train_model(model, train_loader, num_epochs, learning_rate, regularization_strength)
