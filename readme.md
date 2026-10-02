# CodeAlpha_CreditScoring

Credit scoring model built during my Machine Learning internship at CodeAlpha.

## Objective
Predict whether a customer has good or bad credit using past financial data.

## Dataset
German Credit Data (1,000 customers, 20 features), loaded from OpenML.

## Approach
1. Checked for missing values (none found)
2. Dropped `personal_status` to avoid gender-based decisions
3. Feature engineering: `monthly_payment` (amount / duration) and `is_long_loan`
4. 80/20 stratified train-test split
5. Scaled numeric features and one-hot encoded categorical features
6. Trained Logistic Regression, Decision Tree, and Random Forest

## Results (test set)
| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.847 | 0.671 | 0.749 | 0.748 |
| Decision Tree | 0.844 | 0.579 | 0.686 | 0.696 |
| Random Forest | 0.831 | 0.771 | 0.800 | 0.784 |

Random Forest performed best overall. Top features: credit amount,
monthly payment (engineered), age, duration, and checking account status.

## Limitations
- Small test set (200 customers), so small score gaps may be noise
- `age` is still used as a feature, which needs care in real lending

## How to run
pip install -r requirements.txt
Open credit_scoring.ipynb and run all cells.