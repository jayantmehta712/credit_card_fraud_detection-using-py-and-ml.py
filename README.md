# Credit Card Fraud Detection

A machine learning project built in Python that detects fraudulent credit card transactions. The pipeline handles class imbalance using oversampling, trains a Random Forest classifier, and evaluates it using industry-standard fraud detection metrics.

---

## Overview

Credit card fraud detection is an inherently imbalanced classification problem — fraudulent transactions typically make up a very small fraction of all records. This project addresses that challenge using `RandomOverSampler` from the `imbalanced-learn` library before training, ensuring the model does not simply learn to predict the majority class.

---

## Repository Structure

```
credit_card_fraud_detection-using-py-and-ml.py/
└── credit_card_fraud_detection using py and ml.py   # Main Python script
```

---

## Features

- **Data Loading** — Reads transaction data from a CSV file
- **Class Imbalance Handling** — Applies Random Oversampling to balance fraud vs. non-fraud classes
- **Feature Scaling** — Standardises features using `StandardScaler`
- **Model Training** — Trains a Random Forest classifier on the resampled data
- **Evaluation** — Reports confusion matrix, classification report, and ROC-AUC score
- **Model Persistence** — Saves the trained model and scaler as `.pkl` files using `pickle`

---

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- imbalanced-learn

Install all dependencies with:

```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

---

## Dataset Format

The script expects a CSV file with the following structure:

| Column        | Description                                      |
|---------------|--------------------------------------------------|
| `V1` – `V28`  | PCA-transformed transaction features             |
| `Time`        | Seconds elapsed between the transaction and the first transaction in the dataset |
| `Amount`      | Transaction amount                               |
| `Class`       | Target label — `1` for fraud, `0` for legitimate |

This format matches the commonly used [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

---

## Getting Started

### 1. Prepare your dataset

Place your CSV file in the project directory and ensure it contains a `Class` column as the target.

### 2. Run the script

```bash
python "credit_card_fraud_detection using py and ml.py"
```

Update the file path in the script where `load_data` is called:

```python
df = load_data('your_data.csv')  # replace with your actual file path
```

### 3. Output

The script prints evaluation metrics and saves:

```
fraud_model.pkl
scaler.pkl
```

---

## Pipeline Summary

```
Raw CSV Data
     |
     v
Drop irrelevant columns
     |
     v
Separate features (X) and target (y = 'Class')
     |
     v
Random Oversampling  (balance fraud vs. non-fraud)
     |
     v
Standard Scaling
     |
     v
Train/Test Split (80/20)
     |
     v
Random Forest Classifier
     |
     v
Evaluation: Confusion Matrix | Classification Report | ROC-AUC
```

---

## Evaluation Metrics

Given the imbalanced nature of fraud data, the following metrics are used:

- **Confusion Matrix** — Identifies true/false positives and negatives
- **Classification Report** — Precision, Recall, and F1-Score per class
- **ROC-AUC Score** — Measures the model's ability to distinguish between classes

---

## Model Used

| Model          | Configuration                          |
|----------------|----------------------------------------|
| Random Forest  | 100 estimators, random_state=42        |

---

## Author

**Jayant Mehta**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jayant-mehta-b2752a302/)

---

