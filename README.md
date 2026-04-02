# Titanic Survival Prediction using Machine Learning

## Project Overview
This project predicts whether a Titanic passenger survived (binary classification). The model is Logistic Regression with a full data pipeline:
- data loading (from CSV or synthetic fallback)
- cleaning: missing value imputation
- feature engineering: encoding categorical features
- train/test split (80/20)
- model training + evaluation

## Why this matters
- Predicts survival likelihood based on passenger attributes.
- Demonstrates how data enables decisions by revealing which features drive outcomes.

## Dataset source
- Recommended: Kaggle Titanic dataset
  - https://www.kaggle.com/c/titanic/data
- Notebook also supports synthetic 100-row dataset if no file found.

## Notebook structure
1. Project title and objective
2. Problem statement
3. Dataset understanding
4. EDA (survival distribution, gender/class impact, age)
5. Cleaning (Age/Embarked treatment, drop irrelevant columns)
6. Feature engineering (Sex mapping, Embarked one-hot)
7. Train-test split (80/20)
8. Model selection: Logistic Regression
9. Training and evaluation (accuracy)
10. Results + insights (women, class, age trends)

## Addressing the NaN error in `LogisticRegression`
Error trigger:
- `ValueError: Input X contains NaN.`
- logistic regression cannot handle missing values directly.

Solution in notebook:
1. Fill missing values in explicit columns:
   - `Age` using median
   - `Embarked` using mode
2. Add catch-all imputation:
   - numeric: median (`SimpleImputer(strategy='median')`)
   - categorical: most frequent (`SimpleImputer(strategy='most_frequent')`)
3. Validate no missing values:
   - `df.isna().sum()`
   - assertion before fitting:
     - `assert not X_train.isna().any().any()`
     - `assert not y_train.isna().any()`

## Running the notebook (colab/local)
1. Put your `titanic.csv` in the same directory, or use synthetic data mode.
2. Run Cell 1 (load dataset).
3. Run Cell 2 (clean + encode + impute).
4. Run Cell 3 (split + train). Ensure assertions pass.
5. Run Cell 4 (evaluate + insights).

## Tips
- Re-run cells from the top if you edit preprocessing.
- If using real dataset, make sure `Survived`, `Pclass`, `Sex`, `Age`, `Fare`, `Embarked`, `SibSp`, `Parch` are present.
- If any remaining columns are object strings, convert them or include in imputation/encoding.

## Output interpretation
- Accuracy indicates correct predictions proportion.
- Coefficients show direction (+/-) and importance of features.
- Expected insights:
  - Females higher survival
  - 1st class higher survival
  - Younger passengers tend to survive more
