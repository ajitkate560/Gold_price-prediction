import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.title("💰 Gold Price Prediction App")

data = yf.download("GC=F", start="2010-01-01", end="2025-01-01")
data['MA10'] = data['Close'].rolling(10).mean()
data['MA50'] = data['Close'].rolling(50).mean()
data['MA100'] = data['Close'].rolling(100).mean()
data.dropna(inplace=True)

X = data[['Open', 'High', 'Low', 'Volume', 'MA10', 'MA50', 'MA100']]
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)

st.write(f"### R² Score: {r2:.3f}")
st.write(f"### Mean Absolute Error: {mae:.2f}")

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(y_test.values, label='Actual', color='gold')
ax.plot(pred, label='Predicted', color='blue')
ax.legend()
st.pyplot(fig)

last = data.tail(1)[['Open','High','Low','Volume','MA10','MA50','MA100']]
next_price = model.predict(last)
st.write(f"### 💰 Predicted Next Day Gold Price: ${next_price[0]:.2f}")
