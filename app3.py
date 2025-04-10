import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ----------------- Streamlit Page Configuration -----------------
st.set_page_config(page_title="Stock Predictor", layout="wide")

# Dark theme CSS
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #ffffff;
}
.stApp {
    background-color: #0e1117;
    color: white;
}
h1, h2, h3, h4, h5, h6 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ----------------- Header -----------------
st.title("📈 LSTM Stock Price Forecast with Technical Analysis")

# ----------------- User Input -----------------
stock = st.text_input("Enter Stock Symbol (e.g. AAPL, GOOG, ^NSEI)", value="AAPL")
future_days = st.slider("Days to Predict into Future", min_value=1, max_value=90, value=30)
start_date = st.date_input("Select Start Date", value=datetime.now() - timedelta(days=365*5))
end_date = st.date_input("Select End Date", value=datetime.now())

# ----------------- Load Model -----------------
model = load_model("Stock_Prediction_Future_Model2.keras")

# ----------------- Download Stock Data -----------------
@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    return df

if st.button("🔄 Refresh Stock Data"):
    st.experimental_rerun()

df = load_data(stock, start_date, end_date)

if df.empty:
    st.error("❌ Could not fetch data. Please check the symbol.")
    st.stop()

st.success(f"✅ Data Loaded: {df.shape[0]} rows")

# ----------------- Technical Indicator Calculation -----------------
def add_technical_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26

    # Bollinger Bands
    ma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['Upper_BB'] = ma20 + (2 * std20)
    df['Lower_BB'] = ma20 - (2 * std20)

    return df

df = add_technical_indicators(df)
df = df.dropna()

# ----------------- Prepare Data for Prediction -----------------
data_close = df[['Close']].values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_close)

past_days = 100
x_input = scaled_data[-past_days:]
x_input = x_input.reshape(1, past_days, 1)

# Predict Future
future_predictions = []
for _ in range(future_days):
    pred = model.predict(x_input)[0][0]
    future_predictions.append(pred)
    x_input = np.append(x_input[:, 1:, :], [[[pred]]], axis=1)

predicted_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
last_date = df.index[-1]
future_dates = [last_date + timedelta(days=i+1) for i in range(future_days)]
predicted_df = pd.DataFrame(predicted_prices, columns=["Predicted"], index=future_dates)

# ----------------- Calculate RMSE -----------------
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - past_days:]

x_test = []
y_test = []

for i in range(past_days, len(test_data)):
    x_test.append(test_data[i - past_days:i])
    y_test.append(test_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_pred = model.predict(x_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# ----------------- Plotting -----------------
st.subheader(f"📊 {stock.upper()} Stock Forecast & Technical Chart")
plot_type = st.radio("Choose Chart Type", ["Candlestick", "Line Chart"])

if plot_type == "Candlestick":
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'], name="Candlestick"))
    fig.add_trace(go.Scatter(x=predicted_df.index, y=predicted_df['Predicted'],
                             mode='lines', name="Predicted Price", line=dict(color='red', dash='dash')))
else:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Actual Price', line=dict(color='lime')))
    fig.add_trace(go.Scatter(x=predicted_df.index, y=predicted_df['Predicted'], name='Predicted Price',
                             line=dict(color='red', dash='dash')))

# Add selected indicators
selected_indicators = st.multiselect("📈 Select Indicators to Display", 
                                     ["SMA_50", "EMA_20", "RSI", "MACD", "Upper_BB", "Lower_BB"],
                                     default=["SMA_50", "EMA_20"])

for indicator in selected_indicators:
    fig.add_trace(go.Scatter(x=df.index, y=df[indicator], name=indicator))

fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False,
                  title=f"{stock.upper()} Stock Price with Forecast",
                  xaxis_title="Date", yaxis_title="Price")

st.plotly_chart(fig, use_container_width=True)

# ----------------- Show Forecast -----------------
st.subheader("🔮 Predicted Prices Table")
st.dataframe(predicted_df)

# ----------------- Model Confidence -----------------
st.info(f"📉 Model Confidence (RMSE): `{rmse:.2f}`")

# ----------------- Footer -----------------
st.caption("⚡ Powered by LSTM, Streamlit, and yFinance")
