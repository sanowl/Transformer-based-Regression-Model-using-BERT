

```markdown
# ShallowReLURegression

## Introduction

This project implements a regression model using the BERT transformer architecture. The model is trained to predict target values based on input text sequences.

## Model Architecture

- **Base Model**: `bert-base-uncased` pre-trained model from Hugging Face's transformers library.
- **Components**:
  - **BERT Model**: Extracts contextualized embeddings.
  - **Dropout Layer**: Regularization to prevent overfitting.
  - **Fully Connected Layer**: Maps BERT embeddings to target values.

## Data Preparation

- **Tokenization**: Tokenizes text using `BertTokenizer`.
- **Padding & Attention Masks**: Ensures uniform sequence length and generates attention masks.

## Training

- **Optimizer**: Adam optimizer.
- **Loss Function**: Mean Squared Error (MSE).
- **Regularization**: L2 regularization to prevent overfitting.
- **Early Stopping**: Implemented to halt training when validation loss stops improving.

### Training Command

```bash
python train.py --epochs 50 --learning_rate 2e-5 --lambda_reg 0.01 --clip_value 1.0 --early_stopping_patience 10
```

## Evaluation

- **Metrics**: Mean Absolute Error (MAE) and R-squared (R²).
- **Results**: Achieved a test loss of 0.0133, MAE of 0.1046, and R² of 0.9983.

### Evaluation Command

```bash
python evaluate.py
```

## Results

- **Test Loss**: 0.0133
- **Mean Absolute Error (MAE)**: 0.1046
- **R-squared (R²)**: 0.9983

## Visuals

- **Training & Validation Loss Plot**: `training_validation_loss.png`
- **Predictions vs Actual Values Plot**: `actual_vs_predicted.png`

## Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Training**:
   ```bash
   python train.py
   ```

3. **Run Evaluation**:
   ```bash
   python evaluate.py
   ```

## Conclusion

This BERT-based regression model demonstrates high performance on the test dataset, effectively capturing the relationship between text input and target values. Future work could include hyperparameter tuning, model optimization, and deployment considerations.
