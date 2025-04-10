# 📈 Stock Price Prediction

This project aims to predict stock prices using machine learning models trained on historical stock data. It leverages deep learning techniques and data visualization to analyze market trends.

## 🚀 Features
- Predicts stock prices based on historical data.
- Implements LSTM-based deep learning models.
- Visualizes trends with interactive plots.
- Uses moving averages and other indicators.

## 📸 Results:

![image](https://github.com/user-attachments/assets/3e35e20e-902f-4114-94d7-092f93bb4277)
![image](https://github.com/user-attachments/assets/c0ff3e3f-8a09-4133-a349-319b5f0fc7fb)
![image](https://github.com/user-attachments/assets/129774d8-d7f6-4936-8ab9-a2f89e48c07b)

https://github.com/user-attachments/assets/7880d9fc-b893-4aba-a7b7-f43f7ff44e60



## 🛠️ INSTALLATION & SETUP

First you have to install Anaconda Navigator  -> Launch Jupyter Notebook -> Create New file (shown)
![image](https://github.com/user-attachments/assets/44c97eb0-ace3-4845-afd2-0b23ab1768af)

COPY CODE (.ipynb file):
``` bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
```
``` bash
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)
stock = '^NSEI'  # Replace with your desired stock symbol

data = yf.download(stock, start, end)
data.reset_index(inplace=True)
data.tail()
```
``` bash
num_zero_rows = 600  # Change if needed
zero_rows = pd.DataFrame(np.nan, index=range(num_zero_rows), columns=data.columns)
data = pd.concat([data, zero_rows], ignore_index=True)
print(data.shape)
```
``` bash
ma_100_days = data['Close'].rolling(100).mean()
ma_200_days = data['Close'].rolling(200).mean()

plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r', label='MA 100')
plt.plot(ma_200_days, 'b', label='MA 200')
plt.plot(data['Close'], 'g', label='Close')
plt.legend()
plt.show()

```
``` bash
def calculate_rsi(df, period=14, price_col='Close'):
    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

df = data.copy()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df['RSI'] = calculate_rsi(df)
```
``` bash
plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
plt.title('Stock Closing Price')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(df['Date'], df['RSI'], label='RSI', color='purple')
plt.axhline(70, linestyle='--', alpha=0.5, color='red', label='Overbought (70)')
plt.axhline(30, linestyle='--', alpha=0.5, color='green', label='Oversold (30)')
plt.title('Relative Strength Index (RSI)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

```
``` bash
data.dropna(inplace=True)

data_train = pd.DataFrame(data['Close'][0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data) * 0.80):])

print(data_train.shape[0])
print(data_test.shape[0])
```
``` bash
scaler = MinMaxScaler(feature_range=(0,1))
data_train_scale = scaler.fit_transform(data_train)

x = []
y = []

for i in range(100, data_train_scale.shape[0]):
    x.append(data_train_scale[i-100:i])
    y.append(data_train_scale[i,0])

x, y = np.array(x), np.array(y)
```
``` bash
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(x.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x, y, epochs=25, batch_size=32, verbose=1)
```
``` bash
past_100_days = data_train.tail(100)
```
``` bash
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
```
``` bash
data_test_scale = scaler.transform(data_test)
```
``` bash
x_test = []
y_test = []

for i in range(100, data_test_scale.shape[0]):
    x_test.append(data_test_scale[i-100:i])
    y_test.append(data_test_scale[i, 0])
```
``` bash
x_test, y_test = np.array(x_test), np.array(y_test)
y_pred = model.predict(x_test)
```
``` bash
scale_factor = 1 / scaler.scale_
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor
```
``` bash
plt.figure(figsize=(10, 8))
plt.plot(y_pred, 'r', label='Predicted Price')
plt.plot(y_test, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title("Stock Price Prediction vs Original")
plt.legend()
plt.show()
```
``` bash
future_days = 30  # Days to predict into the future

last_100_scaled = data_train_scale[-100:]
pred_input = list(last_100_scaled.reshape(100))
future_predicted = []

for _ in range(future_days):
    x_input = np.array(pred_input[-100:]).reshape(1, 100, 1)
    pred = model.predict(x_input, verbose=0)
    future_predicted.append(pred[0, 0])
    pred_input.append(pred[0, 0])

future_predicted = np.array(future_predicted).reshape(-1, 1)
future_predicted = future_predicted * scale_factor
```
``` bash
future_dates = pd.date_range(start=end + timedelta(days=1), periods=future_days)

plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Close'], label='Historical Price')
plt.plot(future_dates, future_predicted, label='Future Prediction', color='orange')
plt.title('Stock Price Forecast for Next 30 Days')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
```
``` bash
model.save('Stock_Prediction_Future_Model2.keras')
```

1. Activate environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
2. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-price-prediction.git
   cd stock-price-prediction
   ```
3. install all packages
   ``` bash
   pip install numpy pandas yfinance tensorflow scikit-learn matplotlib streamlit
   ```
4. Run the app:
   ```bash
   streamlit run app1.py
   ```


🧠 Model Training
- The model is trained using LSTM (Long Short-Term Memory) networks.
- Uses datasets from Yahoo Finance (or any stock data provider).
- Hyperparameters are tuned for optimal prediction accuracy.

📊 Visualization
Stock price trends are displayed using Matplotlib.
Moving Averages and price predictions are plotted.

🤝 Contributing
Feel free to open issues or contribute with pull requests.

📜 License
This project is licensed under the MIT License.
