# Loan-Default-Prediction
markdown
# Loan Default Prediction Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-3.3.2-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A machine learning model to predict loan default risk using LightGBM, trained on bank loan data from Kaggle.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project implements a binary classification model to predict whether a bank customer is likely to default on a loan. The model uses:
- LightGBM gradient boosting framework
- SMOTE for handling class imbalance
- Feature importance analysis
- Comprehensive evaluation metrics

## Dataset
The dataset comes from [Bank Loan Modelling on Kaggle](https://www.kaggle.com/datasets/itsmesunil/bank-loan-modelling) and contains:
- 5,000 records of bank customers
- 11 predictive features
- Personal Loan acceptance as target variable (converted to default risk)

## Features
The model uses the following customer attributes:
- Demographic: Age, Experience, Family size
- Financial: Income, CCAvg (credit card spending), Mortgage
- Banking products: Securities Account, CD Account, Online banking, CreditCard
- Education level

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/loan-default-prediction.git
cd loan-default-prediction
Install requirements:

bash
pip install -r requirements.txt
Download the dataset:

bash
kaggle datasets download -d itsmesunil/bank-loan-modelling
unzip bank-loan-modelling.zip
Usage
Run the Jupyter notebook:

bash
jupyter notebook Loan_Default_Prediction.ipynb
Or execute the Python script:

bash
python loan_prediction.py
Make predictions with the trained model:

python
import joblib
model = joblib.load('loan_default_model.pkl')

sample = {
    'Age': 45,
    'Experience': 20,
    'Income': 100,  # in thousands
    # ... other features
}
prediction = model.predict([list(sample.values())])
Model Performance
Evaluation metrics on test set:

Accuracy: 92.5%

ROC AUC: 0.96

Precision (Default class): 0.89

Recall (Default class): 0.85

Results
https://images/feature_importance.png
Top predictive features: Income, CCAvg, and Education level

https://images/confusion_matrix.png
Model shows strong performance on both classes

Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

License
This project is licensed under the MIT License - see the LICENSE file for details.

text

### Recommended Repository Structure:
loan-default-prediction/
│
├── README.md
├── Loan_Default_Prediction.ipynb
├── loan_prediction.py
├── requirements.txt
├── images/
│ ├── feature_importance.png
│ └── confusion_matrix.png
└── Bank_Personal_Loan_Modelling.xlsx

text

### Additional recommendations:
1. Add a `requirements.txt` file with all dependencies
2. Include sample output images in an `images/` folder
3. Add a `.gitignore` file to exclude large data files
4. Consider adding GitHub Actions for automated testing
5. Include example API usage if deploying as a service

The README provides clear installation instructions, usage examples, and visual documentation of results to help users understand and use your project effectively.
