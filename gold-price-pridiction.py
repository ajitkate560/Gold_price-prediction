import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Title
st.title("🏆 Gold Price Prediction App (₹ per gram)")

# Fetch data
data = yf.download("GC=F", start="2010-01-01", end="2025-01-01")

# Moving averages
data['MA10'] = data['Close'].rolling(10).mean()
data['MA50'] = data['Close'].rolling(50).mean()
data['MA100'] = data['Close'].rolling(100).mean()
data.dropna(inplace=True)

# Features and labels
X = data[['Open', 'High', 'Low', 'Volume', 'MA10', 'MA50', 'MA100']]
y = data['Close']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)

st.write(f"📊 **R² Score:** {r2:.3f}")
st.write(f"📉 **Mean Absolute Error:** {mae:.2f}")

# Plot actual vs predicted
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y_test.values, label='Actual (USD/oz)', color='gold')
ax.plot(pred, label='Predicted (USD/oz)', color='blue')
ax.legend()
st.pyplot(fig)

# Predict next day
last = data.tail(1)[['Open', 'High', 'Low', 'Volume', 'MA10', 'MA50', 'MA100']]
next_pred_usd = model.predict(np.array(last))[0]

# Convert USD/oz → ₹/gram
usd_to_inr = 83         # current USD-INR conversion rate
ounce_to_gram = 31.1035
next_pred_inr = (next_pred_usd * usd_to_inr) / ounce_to_gram

# Add approximate retail multiplier
retail_multiplier = 2.0
retail_pred = next_pred_inr * retail_multiplier

st.markdown(f"### 💰 Predicted Next Day Gold Price (Wholesale COMEX): ₹{next_pred_inr:,.2f} per gram")
st.markdown(f"### 🪙 Approx. Indian Retail Gold Price (24K): ₹{retail_pred:,.2f} per gram")
st.caption("Converted using 1 USD = ₹83, 1 oz = 31.1035 g, and includes ~2× retail adjustment (taxes, duties, margins).")
