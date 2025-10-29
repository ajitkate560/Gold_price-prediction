import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# -------------------------------
# Streamlit App Title
# -------------------------------
st.set_page_config(page_title="Gold Price Prediction 🇮🇳", layout="centered")
st.title("🏆 Gold Price Prediction App (in ₹ per gram)")

# -------------------------------
# Step 1: Download Gold Data
# -------------------------------
st.write("📊 Fetching latest gold data from Yahoo Finance...")
data = yf.download("GC=F", start="2020-01-01", end="2025-10-01")
data = data[['Close']]
data = data.dropna().reset_index()

st.write("### Sample Data")
st.dataframe(data.tail())

# -------------------------------
# Step 2: Feature Engineering
# -------------------------------
data['Target'] = data['Close'].shift(-1)
data = data.dropna()

X = np.array(data[['Close']])
y = np.array(data['Target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# -------------------------------
# Step 3: Train Model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

predicted = model.predict(X_test)

# -------------------------------
# Step 4: Evaluate Model
# -------------------------------
r2 = r2_score(y_test, predicted)
mae = mean_absolute_error(y_test, predicted)

st.subheader(f"📈 R² Score: {r2:.3f}")
st.subheader(f"📉 Mean Absolute Error: {mae:.2f}")

# -------------------------------
# Step 5: Predict Next Day Price
# -------------------------------
last_price = data['Close'].iloc[-1]
next_day_pred_usd = model.predict(np.array([[last_price]]))[0]


# Convert USD/ounce → ₹/gram
usd_to_inr = 83.0  # static conversion rate
predicted_inr_per_gram = (next_day_pred_usd * usd_to_inr) / 28.3495

st.markdown(
    f"### 💰 Predicted Next Day Gold Price: **₹{predicted_inr_per_gram:.2f} per gram**"
)

# -------------------------------
# Step 6: Plot Actual vs Predicted
# -------------------------------
plt.figure(figsize=(10, 5))
plt.plot(y_test, color='gold', label='Actual')
plt.plot(predicted, color='blue', label='Predicted')
plt.legend()
plt.title('Actual vs Predicted Gold Price')
plt.xlabel('Days')
plt.ylabel('Price (USD/ounce)')
st.pyplot(plt)

st.write("📅 Prediction is based on historical gold prices (USD/ounce) converted to ₹/gram.")
