# Indian Stock Market Analysis & Return Prediction (2000–2020)

A ready-to-run Streamlit mini-project for VS Code. It explores Indian stock market ups/downs, plots prices and returns, trains a regression model, evaluates predictions on a chronological holdout set, and estimates the next trading day's return.

## Features
- Upload CSV files or load CSV files placed in `data/` (including nested folders).
- Filter dates (default 2000–2020) and select a ticker.
- Price chart, daily-return distribution, and monthly returns.
- Feature engineering with lagged returns, moving-average ratios, volatility, price range, volume change and calendar features.
- Random Forest or Histogram Gradient Boosting model.
- Chronological 80/20 train/test split, MAE, RMSE, R² and directional accuracy.
- Download cleaned data and test predictions.

## 1. Get the dataset
Download the dataset from Kaggle: https://www.kaggle.com/datasets/sagara9595/stock-data

Extract the CSV files into this project's `data/` folder. You may keep subfolders. The Kaggle site may require you to sign in and accept its dataset terms. The CSV should contain at least a date column and a closing-price column. Common names such as `Date`, `Close`, `Adj Close`, `Symbol`, `Ticker`, `Open`, `High`, `Low`, and `Volume` are supported.

Alternatively, start the app and upload CSV files using the sidebar. If your dataset has a different schema, adjust the column aliases in `app.py`.

## 2. Run in VS Code (Windows PowerShell)
Open this folder in VS Code, then run these commands in the VS Code terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt terminal and run:

```bat
.venv\Scripts\activate.bat
pip install -r requirements.txt
streamlit run app.py
```

The browser will open at a local Streamlit address, usually `http://localhost:8501`.

## 3. Notes
- Default dates are 2000-01-01 through 2020-12-31 to match the assignment. Change the dates in the sidebar if the data has a different span.
- Model target: next trading day's simple return = next close / current close - 1.
- The final prediction is an educational estimate, not financial advice. It does not guarantee future returns.
- If your CSV contains several stocks in one file, it should have a ticker/symbol column. If each stock is in a separate CSV, the filename is used as its ticker when no symbol column exists.

## Project structure
```
Indian_Stock_Market_Analysis/
├── app.py
├── requirements.txt
├── README.md
├── data/       # put Kaggle CSV files here
└── outputs/    # optional output folder
```
