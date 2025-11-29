import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#-----utilities-------
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

@st.cache_data(show_spinner=False)
def download_stock(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Download OHLCV data and cache it for repeated requests."""
    df = yf.download(symbol, start=start, end=end)
    return df

# main model
def build_lstm(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def get_cache_path(symbol: str, window: int, epochs: int, lr=None):
    """Create a unique filename for model+scaler cache"""
    fname = f"{symbol.replace('.', '_')}_w{window}_e{epochs}.pkl"
    return os.path.join(MODEL_DIR, fname)


def save_model_and_scaler(path: str, model, scaler):
    # Save model (Keras) and scaler via joblib for metadata; Keras saved separately
    model_path = path.replace('.pkl', '.h5')
    model.save(model_path)
    joblib.dump(scaler, path)


def load_model_and_scaler(path: str):
    model_path = path.replace('.pkl', '.h5')
    if os.path.exists(path) and os.path.exists(model_path):
        scaler = joblib.load(path)
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        return model, scaler
    return None, None


def create_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y


def rmse(actual, pred):
    return np.sqrt(mean_squared_error(actual, pred))


def mape(actual, pred):
    # sklearn's mean_absolute_percentage_error is OK, but keep a safe fallback
    try:
        return mean_absolute_percentage_error(actual, pred) * 100
    except Exception:
        return np.mean(np.abs((actual - pred) / actual)) * 100


#-----main Streamlit function--------

def stock_market_simulator():
    st.title("📈 Daily Portfolio Tracker")

    st.markdown(
        """
        This application uses analyze and predict stock price trends based on historical data.
        Its helps users visualize past performance, understand market movements, and make informed investment decisions.
        
        Simply select a stock, view its data trends, and explore future predictions powered by intelligent forecasting models.

        """
    )

    # Top companies (expand as needed)
    companies = {
        "Apple (AAPL)": "AAPL",
        "Microsoft (MSFT)": "MSFT",
        "Google (GOOGL)": "GOOGL",
        "Amazon (AMZN)": "AMZN",
        "Tesla (TSLA)": "TSLA",
        "NVIDIA (NVDA)": "NVDA",
        "Meta (META)": "META",
        "Netflix (NFLX)": "NFLX",
        "Infosys (INFY)": "INFY",
        "Reliance (RELIANCE.NS)": "RELIANCE.NS",
    }

    col1, col2 = st.columns([2, 1])

    with col1:
        company_name = st.selectbox("Select a company", list(companies.keys()))
        symbol = companies[company_name]

        start_date = st.date_input("Start date", pd.to_datetime("2018-01-01"))
        end_date = st.date_input("End date", pd.to_datetime("2025-01-01"))

    with col2:
        window = st.number_input("Sequence window (days)", min_value=10, max_value=200, value=60, step=5)
        epochs = st.number_input("Training epochs", min_value=1, max_value=100, value=15)
        forecast_days = st.selectbox("Forecast horizon (days)", [7, 14, 30, 60, 90], index=2)
        retrain = st.checkbox("Force retrain (ignore cache)", value=False)

    if st.button("Train & Predict"):
        with st.spinner("Running pipeline — fetch, train/load, predict..."):
            # 1) fetch
            df = download_stock(symbol, start_date, end_date)
            if df.empty:
                st.error("No data found for that symbol/date range — try different dates or symbol.")
                return

            close = df['Close'].values.reshape(-1, 1)

            # 2) scale
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(close)

            # 3) sequences
            X, y = create_sequences(scaled, window)
            if len(X) < 10:
                st.error("Not enough data for the chosen window; reduce window or widen date range.")
                return

            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            cache_path = get_cache_path(symbol, window, epochs)

            # 4) load or train
            model, saved_scaler = None, None
            if (not retrain) and os.path.exists(cache_path):
                model, saved_scaler = load_model_and_scaler(cache_path)
                if model is not None:
                    st.info("Loaded cached model and scaler — skipping retrain.")
                    scaler = saved_scaler
            if model is None:
                # build & train
                model = build_lstm((X_train.shape[1], 1))
                es = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
                model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, callbacks=[es])
                save_model_and_scaler(cache_path, model, scaler)
                st.success("Model trained and cached.")

            # 5) predictions on test
            preds_scaled = model.predict(X_test)
            preds = scaler.inverse_transform(preds_scaled)
            y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

            # metrics
            score_rmse = rmse(y_test_actual, preds)
            score_mape = mape(y_test_actual, preds)

            st.write(f"**Test RMSE:** {score_rmse:.4f}")
            st.write(f"**Test MAPE:** {score_mape:.2f}%")

            # 6) trained-fit series (model predictions on train set) — optional visualization
            train_preds_scaled = model.predict(X_train)
            train_preds = scaler.inverse_transform(train_preds_scaled)
            y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))

            # 7) Future multi-day forecast
            last_window = scaled[-window:]
            future_scaled = []
            cur_input = last_window.copy()
            for _ in range(forecast_days):
                p = model.predict(cur_input.reshape(1, window, 1), verbose=0)
                future_scaled.append(p[0, 0])
                cur_input = np.append(cur_input[1:], p)
                cur_input = cur_input.reshape(window, 1)

            future_preds = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()

            # 8) Build DataFrames for plotting
            test_index = df.index[window + split_idx: window + split_idx + len(y_test_actual)]
            train_index = df.index[window: window + len(y_train_actual)]
            future_index = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')

            df_train_plot = pd.DataFrame({
                'Actual': y_train_actual.flatten(),
                'Fitted': train_preds.flatten()
            }, index=train_index)

            df_test_plot = pd.DataFrame({
                'Actual': y_test_actual.flatten(),
                'Predicted': preds.flatten()
            }, index=test_index)

            df_future_plot = pd.DataFrame({'Predicted': future_preds}, index=future_index)


            # 8.1 Combined chart — Actual, Fitted, Predicted, and Future Forecast
            st.subheader(f"📊 {company_name} — Actual vs Predicted (Train, Test & Forecast)")

            # Combine all series into one DataFrame
            combined_df = pd.concat([
                df_train_plot.rename(columns={'Fitted': 'Predicted'}),
                df_test_plot,
                df_future_plot
            ])

            # Optional: Add section labels for clarity
            combined_df['Type'] = ['Train'] * len(df_train_plot) + ['Test'] * len(df_test_plot) + ['Future'] * len(df_future_plot)

            # Plot with Matplotlib for better control
            import matplotlib.pyplot as plt
            plt.style.use('dark_background') 
            plt.figure(figsize=(12, 6))
            plt.plot(df_train_plot.index, df_train_plot['Actual'], label='Actual (Train)', color='blue')
            plt.plot(df_test_plot.index, df_test_plot['Predicted'], label='Predicted (Test)', color='red')
            plt.plot(df_future_plot.index, df_future_plot['Predicted'], label='Forecast (Future)', color='green')

            plt.title(f"{company_name} — Actual vs Predicted & Forecast")
            plt.xlabel("Date")
            plt.ylabel("Stock Price (₹)")
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)

            st.subheader(f"Future Forecast — next {forecast_days} business days")
            st.line_chart(df_future_plot)

            
        # 9) Buy/Sell simulation (improved version)
            initial_cash = 10000.0
            cash = initial_cash
            shares = 0
            portfolio_vals = []
            position = "none"  # can be "none" or "long"

            today_price = float(close[-1, 0])
            last_price = today_price

            # Add threshold (to ignore small noise)
            buy_threshold = 1.002   # 0.2% rise
            sell_threshold = 0.998  # 0.2% drop

            st.subheader("📈 Buy/Sell Trade Log")

            for price in future_preds:
                # BUY if price up more than threshold and not already in position
                if price > last_price * buy_threshold and position == "none":
                    shares = cash / last_price
                    cash = 0
                    position = "long"
                    st.write(f"🟢 BUY at {last_price:.2f}")

                # SELL if price drops more than threshold and currently holding shares
                elif price < last_price * sell_threshold and position == "long":
                    cash = shares * last_price
                    shares = 0
                    position = "none"
                    st.write(f"🔴 SELL at {last_price:.2f}")

                total_value = cash + shares * price
                portfolio_vals.append(total_value)
                last_price = price

            # Final portfolio value
            final_value = portfolio_vals[-1] if portfolio_vals else initial_cash
            pct_return = (final_value - initial_cash) / initial_cash * 100

            # Baseline buy-and-hold
            bh_shares = initial_cash / today_price
            bh_final = bh_shares * future_preds[-1] if len(future_preds) > 0 else initial_cash
            bh_return = (bh_final - initial_cash) / initial_cash * 100

            st.subheader("💰 Portfolio Summary")
            st.write(f"Initial cash: ₹{initial_cash:,.2f}")
            st.write(f"Final portfolio (strategy): ₹{final_value:,.2f} ({pct_return:.2f}%)")
            st.write(f"Buy & Hold baseline: ₹{bh_final:,.2f} ({bh_return:.2f}%)")

           

            # 10)  future predictions
            st.subheader("Future Predictions (table)")
            out_df = pd.DataFrame({
                'Date': future_index,
                'Predicted Close': future_preds
            })
            out_df.set_index('Date', inplace=True)
            st.dataframe(out_df)

            st.success("Done — review charts, metrics and the simulation results.")
