# 1. Imports
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")


# 2. Config (easy to change without touching logic)
CONFIG = {
    "ticker": "AAPL",
    "start": "2020-01-01",
    "end": "2025-01-01",
    "train_ratio": 0.8,
    "arima_order": (5, 1, 0),
    "features": ["Lag1", "Lag2", "MA5", "MA10"]
}


# 3. Load Data
def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df = df[["Close"]].dropna()
    print(f"Loaded {len(df)} rows for {ticker}")
    return df


# 4. Feature Engineering
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # Avoid SettingWithCopyWarning
    df["Lag1"]   = df["Close"].shift(1)
    df["Lag2"]   = df["Close"].shift(2)
    df["MA5"]    = df["Close"].rolling(5).mean()
    df["MA10"]   = df["Close"].rolling(10).mean()
    df["Target"] = df["Close"].shift(-1)
    df.dropna(inplace=True)
    return df


# 5. Split Data
def split_data(df: pd.DataFrame, train_ratio: float):
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].copy()   # .copy() on slices
    test  = df.iloc[split_idx:].copy()
    print(f"Train: {len(train)} rows | Test: {len(test)} rows")
    return train, test


# 6. Train Linear Regression (with scaling)
def train_lr(train: pd.DataFrame, features: list):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train[features])

    model = LinearRegression()
    model.fit(X_train_scaled, train["Target"])
    return model, scaler


# 7. Evaluate — now returns a dict instead of just printing
def evaluate(y_true: pd.Series, y_pred: pd.Series, name: str) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    metrics = {"Model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "R²": round(r2, 4)}
    return metrics


# 8. Plot — much more readable output
def plot_results(test: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(test.index, test["Target"],   label="Actual",           color="black",  linewidth=1.5)
    ax.plot(test.index, test["LR_Pred"],  label="Linear Regression", color="blue",   linewidth=1, linestyle="--")
    ax.plot(test.index, test["ARIMA"],    label="ARIMA",             color="red",    linewidth=1, linestyle="--")
    ax.plot(test.index, test["Naive"],    label="Naive Baseline",    color="gray",   linewidth=1, linestyle=":")

    ax.set_title(f"{ticker} Stock Price Forecasting", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{ticker}_forecast.png", dpi=150)  # ✅ Save for portfolio/README
    plt.show()
    print("Plot saved.")


# 9. Main Pipeline
def main():
    ticker  = CONFIG["ticker"]
    features = CONFIG["features"]

    # Load & engineer features
    df = load_data(ticker, CONFIG["start"], CONFIG["end"])
    df = create_features(df)
    train, test = split_data(df, CONFIG["train_ratio"])

    # --- Linear Regression ---
    model, scaler = train_lr(train, features)
    X_test_scaled   = scaler.transform(test[features])
    test["LR_Pred"] = model.predict(X_test_scaled)

    # --- Naive Baseline ---
    test["Naive"] = test["Lag1"]

    # --- ARIMA ---
    arima_model = ARIMA(train["Close"], order=CONFIG["arima_order"]).fit()
    forecast = arima_model.forecast(steps=len(test))
    test["ARIMA"] = forecast.values  # align by position since we use iloc split

    # --- Evaluate all models ---
    results = []
    results.append(evaluate(test["Target"], test["Naive"],   "Naive Baseline"))
    results.append(evaluate(test["Target"], test["LR_Pred"], "Linear Regression"))
    results.append(evaluate(test["Target"], test["ARIMA"],   "ARIMA"))

    # ✅ Print a clean summary table
    summary = pd.DataFrame(results).set_index("Model")
    print("\n===== Model Comparison =====")
    print(summary.to_string())

    # --- Plot ---
    plot_results(test, ticker)


if __name__ == "__main__":
    main()