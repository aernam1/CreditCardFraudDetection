# Credit Card Fraud Detection

Machine learning project detecting fraudulent credit card transactions using multiple classification algorithms.

## Dataset
- **284,807 transactions** (492 frauds = 0.172% fraud rate)
- Features: V1-V28 (PCA components), Time, Amount, Class
- Highly imbalanced → PR-AUC used as primary metric

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook credit_card_fraud_detection.ipynb
```

## Models Evaluated
1. Logistic Regression
2. Random Forest
3. **XGBoost** (Best: PR-AUC ~0.86, ROC-AUC ~0.97)
4. SVM
5. KNN

## Key Features
- Time-based features (Hour, HourOfDay, Day)
- Log-transformed Amount
- Interaction features (V14_Amount, V12_Amount)
- RobustScaler for preprocessing

## Results
- **Best Model**: XGBoost
- **Top Features**: V14, V12, V10, V4, V1, Amount
- **High-Risk Periods**: Analysis identifies specific hours/days with elevated fraud rates

## Requirements
See `requirements.txt` for dependencies (pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, jupyter)

## Files
- `credit_card_fraud_detection.ipynb` - Main analysis
- `creditcard.csv` - Dataset (download from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud))
