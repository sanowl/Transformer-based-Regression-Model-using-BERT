import torch
from transformers import BertTokenizer
from model import TransformerRegression
from prepare_data import prepare_data_for_transformer
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import torch.nn as nn

def evaluate_model(model, test_loader):
    model.eval()
    test_loss = 0.0
    criterion = nn.MSELoss()
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for inputs, attention_mask, targets in test_loader:
            inputs, attention_mask, targets = inputs.to(model.device), attention_mask.to(model.device), targets.to(model.device)
            outputs = model(inputs, attention_mask)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
    
    test_loss /= len(test_loader)
    mae = mean_absolute_error(all_targets, all_outputs)
    r2 = r2_score(all_targets, all_outputs)
    
    # Plotting predictions vs. actual values
    plt.figure()
    plt.scatter(all_targets, all_outputs, alpha=0.5)
    plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], color='red')  # Line y=x
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.savefig('actual_vs_predicted.png')
    plt.close()
    
    return test_loss, mae, r2

# Example usage
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
_, _, _, _, test_loader = prepare_data_for_transformer(tokenizer)  # Replace with actual test data loading

model = TransformerRegression(model_name, num_features=1)
model.load_state_dict(torch.load('best_model.pth'))

test_loss, mae, r2 = evaluate_model(model, test_loader)
print(f'Test Loss: {test_loss:.4f}')
print(f'Mean Absolute Error: {mae:.4f}')
print(f'R-squared: {r2:.4f}')
