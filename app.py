
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------- CONFIGURATION --------------------

st.set_page_config(
    page_title="Indian Stock Market Analysis",
    page_icon="📈",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"


# -------------------- DATA LOADING --------------------

def normalize_columns(df):
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace("_", " ")
        for c in df.columns
    ]

    aliases = {
        "date": ["date", "datetime", "timestamp"],
        "open": ["open"],
        "high": ["high"],
        "low": ["low"],
        "close": ["close", "adj close", "adjusted close"],
        "volume": ["volume", "vol"],
        "ticker": ["ticker", "symbol", "stock", "company"],
    }

    rename = {}
    for canonical, names in aliases.items():
        for name in names:
            if name in df.columns:
                rename[name] = canonical
                break

    return df.rename(columns=rename)


def read_stock_csv(file, ticker_name=None):
    try:
        df = normalize_columns(pd.read_csv(file))

        if "date" not in df.columns or "close" not in df.columns:
            return None

        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "", regex=False),
                    errors="coerce",
                )

        df = df.dropna(subset=["date", "close"])
        df = df[df["close"] > 0].copy()

        if df.empty:
            return None

        if "ticker" not in df.columns:
            df["ticker"] = ticker_name or Path(str(file)).stem.upper()

        if "open" not in df.columns:
            df["open"] = df["close"]

        if "high" not in df.columns:
            df["high"] = df["close"]

        if "low" not in df.columns:
            df["low"] = df["close"]

        if "volume" not in df.columns:
            df["volume"] = np.nan

        df["ticker"] = df["ticker"].fillna(
            ticker_name or "UNKNOWN"
        ).astype(str).str.upper().str.strip()

        return df.sort_values("date")

    except Exception:
        return None


def load_dataset(uploaded_files):
    frames = []

    if uploaded_files:
        for file in uploaded_files:
            df = read_stock_csv(file, Path(file.name).stem.upper())
            if df is not None:
                frames.append(df)
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        for path in glob.glob(str(DATA_DIR / "**" / "*.csv"), recursive=True):
            df = read_stock_csv(path, Path(path).stem.upper())
            if df is not None:
                frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True, sort=False)


# -------------------- FEATURE ENGINEERING --------------------

def make_features(raw):
    d = raw.sort_values("date").copy()

    d["daily_return"] = d["close"].pct_change()
    d["target_return"] = d["close"].shift(-1) / d["close"] - 1

    d["return_lag_1"] = d["daily_return"].shift(1)
    d["return_lag_2"] = d["daily_return"].shift(2)
    d["return_lag_3"] = d["daily_return"].shift(3)
    d["return_lag_5"] = d["daily_return"].shift(5)
    d["return_lag_10"] = d["daily_return"].shift(10)

    d["ma_5_ratio"] = d["close"] / d["close"].rolling(5).mean() - 1
    d["ma_20_ratio"] = d["close"] / d["close"].rolling(20).mean() - 1

    d["volatility_5"] = d["daily_return"].rolling(5).std()
    d["volatility_20"] = d["daily_return"].rolling(20).std()

    d["range_ratio"] = (
        (d["high"] - d["low"]) / d["close"]
    )

    # Index datasets may contain zero or unavailable volume.
    # Treat undefined volume changes as neutral rather than
    # discarding thousands of otherwise useful rows.
    volume = pd.to_numeric(d["volume"], errors="coerce")
    valid_volume = volume.where(volume > 0)

    d["volume_change"] = (
        valid_volume.pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    d["month"] = d["date"].dt.month
    d["day_of_week"] = d["date"].dt.dayofweek

    feature_cols = [
        "return_lag_1",
        "return_lag_2",
        "return_lag_3",
        "return_lag_5",
        "return_lag_10",
        "ma_5_ratio",
        "ma_20_ratio",
        "volatility_5",
        "volatility_20",
        "range_ratio",
        "volume_change",
        "month",
        "day_of_week",
    ]

    d[feature_cols] = d[feature_cols].replace(
        [np.inf, -np.inf], np.nan
    )

    return d, feature_cols


# -------------------- DASHBOARD --------------------

st.title("📈 Indian Stock Market Analysis & Return Prediction")
st.caption(
    "Historical analysis of Indian market data from 2000–2020 "
    "with experimental next-day return prediction. "
    "For educational purposes only, not investment advice."
)

with st.sidebar:
    st.header("Dataset")

    uploads = st.file_uploader(
        "Upload stock CSV files",
        type=["csv"],
        accept_multiple_files=True,
    )

    st.caption("Or place CSV files in the project's data folder.")

    start_date = st.date_input(
        "Start date",
        value=pd.Timestamp("2000-01-01").date(),
    )

    end_date = st.date_input(
        "End date",
        value=pd.Timestamp("2020-12-31").date(),
    )

    if start_date > end_date:
        st.error("Start date must be before end date.")
        st.stop()


all_data = load_dataset(uploads)

if all_data.empty:
    st.error(
        "No usable CSV data found. Place a CSV file in the data folder "
        "or upload one using the sidebar. The CSV must contain Date and Close."
    )
    st.stop()


all_data = all_data[
    (all_data["date"].dt.date >= start_date)
    & (all_data["date"].dt.date <= end_date)
].copy()

if all_data.empty:
    st.error("No rows match the selected date range.")
    st.stop()


tickers = sorted(all_data["ticker"].dropna().unique().tolist())

if not tickers:
    st.error("No stock symbols were found in the dataset.")
    st.stop()

selected = st.sidebar.selectbox("Select stock / ticker", tickers)

df = (
    all_data[all_data["ticker"] == selected]
    .sort_values("date")
    .drop_duplicates(subset=["date"], keep="last")
    .copy()
)

if len(df) < 35:
    st.warning("Few rows are available. Model evaluation may be unreliable.")


# -------------------- SUMMARY METRICS --------------------

first_close = float(df["close"].iloc[0])
last_close = float(df["close"].iloc[-1])

total_return = (
    (last_close / first_close - 1) * 100
    if first_close > 0 else 0
)

daily_returns = df["close"].pct_change().dropna()

positive_days = (
    (daily_returns > 0).mean() * 100
    if len(daily_returns) else 0
)

m1, m2, m3, m4 = st.columns(4)

m1.metric("Selected Stock", selected)
m2.metric("Latest Close", f"{last_close:,.2f}")
m3.metric("Total Price Return", f"{total_return:+.2f}%")
m4.metric("Positive Trading Days", f"{positive_days:.2f}%")


# -------------------- HISTORICAL PRICE --------------------

st.subheader("1. Historical Stock Price")

fig = px.line(
    df,
    x="date",
    y="close",
    title=f"{selected} — Closing Price",
    labels={"date": "Date", "close": "Closing Price"},
)

st.plotly_chart(fig, use_container_width=True)


# -------------------- MARKET UPS AND DOWNS --------------------

st.subheader("2. Returns and Market Ups/Downs")

df["daily_return_pct"] = df["close"].pct_change() * 100

left, right = st.columns(2)

with left:
    st.plotly_chart(
        px.histogram(
            df.dropna(subset=["daily_return_pct"]),
            x="daily_return_pct",
            nbins=60,
            title="Daily Return Distribution (%)",
        ),
        use_container_width=True,
    )

with right:
    monthly = (
        df.set_index("date")["close"]
        .resample("ME")
        .last()
        .pct_change()
        .mul(100)
        .dropna()
        .rename("monthly_return_pct")
        .reset_index()
    )

    st.plotly_chart(
        px.bar(
            monthly,
            x="date",
            y="monthly_return_pct",
            title="Monthly Returns (%)",
            color="monthly_return_pct",
            color_continuous_scale="RdYlGn",
        ),
        use_container_width=True,
    )


# -------------------- MACHINE LEARNING --------------------

st.subheader("3. Predict Next-Day Stock Return")

featured, feature_cols = make_features(df)

model_data = (
    featured.dropna(subset=feature_cols + ["target_return"])
    .copy()
)

with st.expander("Model data diagnostics"):
    st.write("Total historical rows:", len(df))
    st.write("Rows available for training and testing:", len(model_data))
    st.write("Feature columns:", feature_cols)

    missing_counts = featured[feature_cols].isna().sum()
    st.write("Missing values by feature:")
    st.dataframe(missing_counts.rename("Missing rows").to_frame())

if len(model_data) < 40:
    st.error(
        f"Only {len(model_data)} usable rows remain. "
        "Check the date range, closing prices, and CSV format."
    )
    st.stop()


model_name = st.selectbox(
    "Select machine learning model",
    ["Random Forest", "Histogram Gradient Boosting"],
)

test_fraction = st.slider(
    "Test data percentage",
    min_value=10,
    max_value=30,
    value=20,
    step=5,
)

split = int(len(model_data) * (1 - test_fraction / 100))

if split < 20 or len(model_data) - split < 5:
    st.error("Insufficient training or test records.")
    st.stop()

train = model_data.iloc[:split]
test = model_data.iloc[split:]

if model_name == "Random Forest":
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
else:
    model = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.06,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )

model.fit(train[feature_cols], train["target_return"])

preds = model.predict(test[feature_cols])
actual = test["target_return"].to_numpy()

mae = mean_absolute_error(actual, preds)
rmse = np.sqrt(mean_squared_error(actual, preds))
r2 = r2_score(actual, preds)

direction_accuracy = (
    (np.sign(actual) == np.sign(preds)).mean() * 100
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("MAE", f"{mae * 100:.4f}%")
c2.metric("RMSE", f"{rmse * 100:.4f}%")
c3.metric("R² Score", f"{r2:.4f}")
c4.metric("Direction Accuracy", f"{direction_accuracy:.2f}%")


# -------------------- PREDICTION CHART --------------------

results = pd.DataFrame({
    "Date": test["date"].to_numpy(),
    "Actual Return (%)": actual * 100,
    "Predicted Return (%)": preds * 100,
})

st.plotly_chart(
    px.line(
        results,
        x="Date",
        y=["Actual Return (%)", "Predicted Return (%)"],
        title="Actual vs Predicted Next-Day Returns",
    ),
    use_container_width=True,
)


# -------------------- NEXT-DAY ESTIMATE --------------------

latest_rows = featured.dropna(subset=feature_cols)

if not latest_rows.empty:
    latest_row = latest_rows.iloc[[-1]]

    next_return = float(
        model.predict(latest_row[feature_cols])[0]
    )

    next_close = float(
        latest_row["close"].iloc[0] * (1 + next_return)
    )

    p1, p2, p3 = st.columns(3)

    p1.metric(
        "Estimated Next-Day Return",
        f"{next_return * 100:+.3f}%",
    )

    p2.metric(
        "Latest Closing Price",
        f"{latest_row['close'].iloc[0]:,.2f}",
    )

    p3.metric(
        "Illustrative Next-Day Price",
        f"{next_close:,.2f}",
    )

    st.caption(
        "This is a model estimate, not a guaranteed future price. "
        "It excludes transaction costs and does not constitute investment advice."
    )


# -------------------- DOWNLOADS --------------------

st.subheader("4. Download Results")

left, right = st.columns(2)

with left:
    st.download_button(
        "Download Prediction Results CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="stock_return_predictions.csv",
        mime="text/csv",
    )

with right:
    st.download_button(
        "Download Historical Data CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected.lower()}_historical_data.csv",
        mime="text/csv",
    )

with st.expander("View historical data"):
    st.dataframe(df, use_container_width=True)
