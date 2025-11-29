# FinWise – Financial Analysis & Prediction System
FinWise is a Streamlit-based personal finance platform using Machine Learning and Deep Learning.
It includes four major features: Fraud Detection, Stock Price Prediction, Loan Approval Prediction, and Future Savings Prediction.Built with RandomForest, XGBoost, and LSTM models, it offers clear, real-time insights for smarter financial decisions.
This project combines analytics + ML models to help users understand, predict, and analyze financial patterns.

## How to Run the Project
Use the **module format** to launch Streamlit:
```bash
python -m streamlit run app.py
```

# 📁 Project Structure
```bash
FinWise/
│── app.py                      # Main Streamlit Navigation (4 feature tabs)
│── fraud_detection.py          # Fraud Detection module
│── fraud_model.py              # Model creation script for fraud detection
│── loan_approval.py            # Loan approval prediction module
│── model_building_loan.py      # Model creation script for loan approval
│── stock_predictor.py          # Stock LSTM model & prediction logic
│── predict_savings.py          # Future savings prediction module
│── abc.py                      # Model creation script for savings prediction
│── profile.json
│── .gitignore
│── README.md
│── models/                     # (Download models into this folder)
│── data/                       # (Download datasets into this folder)
```
# 📥 Download Required Files
```
Due to large file size limits, models and datasets are NOT included in the GitHub repo.
Download everything from Google Drive:
```
🔗 Google Drive (Models + Datasets):
```
https://drive.google.com/drive/folders/15wrTdKvCtFP-7P716CXwQ-WfoUJGaYe6?usp=drive_link
```
After downloading:
```
Place all .pkl & .h5 model files inside the models/ folder
Place all .csv dataset files inside the data/ folder
```
##🧠 ML Features & Model Details
1️) Fraud Detection
```
File: fraud_detection.py
Dataset: transaction_dataset.csv
Model Creation Script: fraud_model.py
ML Model Used: RandomForestClassifier
```
This module predicts whether a financial transaction is fraudulent based on patterns learned from the dataset.

2️) Stock Price Predictor (LSTM)
```
File: stock_predictor.py
Dataset: No local dataset (data fetched from the internet)
Model Creation Script: The model is built inside the .py file
ML Model Used: LSTM (Long Short-Term Memory Neural Network)
```
This module predicts future stock prices based on time-series deep learning using LSTM networks.

3) Loan Approval Prediction
```
File: loan_approval.py
Dataset: loan_data_.csv
Model Creation Script: model_building_loan.py
ML Model Used: XGBoostClassifier
```
Predicts whether a user is eligible for a loan using income, credit history, and other financial factors.

4️) Future Savings Prediction
```
File: predict_savings.py
Dataset: synthetic_finance_data.csv
Model Creation Script: abc.py
ML Model Used: RandomForestRegressor
```
Estimates next-month savings based on spending behavior and income patterns.

5) Main File (app.py)
```
app.py is the central Streamlit application.
It contains:
Sidebar navigation
Four feature tabs
UI integration for all ML modules
Inputs + predictions for each model
This is the only file you run to use the entire application.
```
## ⚙️ Requirements
```
All requirment in "requirement.text" file
```
## Credits
Developed as part of an academic project showcasing financial analytics using AI/ML, combining user-friendly UI with powerful prediction models.

## Contact

Feel free to reach out if you want to suggest improvements, collaborate, or learn more about the project.
