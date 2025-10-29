# Save this as gold_price_app.py (replace your old file)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Gold Price Prediction (INR)", layout="centered")
st.title("🏆 Gold Price Prediction App — Train & Predict in INR")

# ---------- User inputs ----------
st.sidebar.header("Settings")
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End date", value=pd.to_datetime("today"))

# Let user enter current USD->INR (default 83). You can paste today's exchange rate here.
usd_to_inr = st.sidebar.number_input("USD → INR exchange rate", value=83.0, format="%.4f")
unit_choice = st.sidebar.selectbox("Output unit", ["per gram", "per 10 grams"])
st.sidebar.markdown(
    "Notes:\n\n"
    "- Data source: Yahoo Finance (COMEX futures `GC=F`, USD per ounce).\n"
    "- Retail prices (jewellers) may include premiums/taxes and are often quoted per 10g."
)

# ---------- Load data ----------
st.write("📥 Fetching gold price data (USD per ounce) ...")
data = yf.download("GC=F", start=start_date, end=end_date)

if data.empty:
    st.error("No data returned. Try different dates.")
    st.stop()

data = data[['Close']].dropna().reset_index()
data.rename(columns={"Close": "USD_per_oz"}, inplace=True)

# ---------- Convert historical series to INR per gram (so model learns desired unit) ----------
OUNCE_TO_GRAM = 28.3495
data['INR_per_gram'] = (data['USD_per_oz'] * usd_to_inr) / OUNCE_TO_GRAM
if unit_choice == "per 10 grams":
    data['INR_per_unit'] = data['INR_per_gram'] * 10
else:
    data['INR_per_unit'] = data['INR_per_gram']

st.write("### Sample (converted to INR based on USD→INR and unit)")
st.dataframe(data.tail(5)[['Date', 'USD_per_oz', 'INR_per_unit']])

# ---------- Feature engineering: use lag features and moving averages in INR space ----------
# We'll predict next-day INR price (INR_per_unit shifted by -1)
data['Target'] = data['INR_per_unit'].shift(-1)
# Create features: last day's INR price and moving averages in INR
data['MA3'] = data['INR_per_unit'].rolling(3).mean()
data['MA7'] = data['INR_per_unit'].rolling(7).mean()
data['MA14'] = data['INR_per_unit'].rolling(14).mean()
data = data.dropna().reset_index(drop=True)

features = ['INR_per_unit', 'MA3', 'MA7', 'MA14']
X = data[features]
y = data['Target']

# ---------- Split ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ---------- Train model ----------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

# ---------- Metrics ----------
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)
st.sidebar.write(f"Model R²: {r2:.3f}")
st.sidebar.write(f"Model MAE: {mae:.2f}")

st.markdown(f"### 📊 Model performance (trained in INR, unit = **{unit_choice}**)")
st.write(f"- R² score: **{r2:.3f}**")
st.write(f"- Mean Absolute Error (MAE): **{mae:.2f} {unit_choice.split()[-1]}**")

# ---------- Plot actual vs predicted ----------
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(y_test.values, label='Actual (INR)', color='gold')
ax.plot(preds, label='Predicted (INR)', color='blue')
ax.set_title(f"Actual vs Predicted ({unit_choice})")
ax.set_ylabel(f"Price ({unit_choice})")
ax.legend()
st.pyplot(fig)

# ---------- Predict next day ----------
# Prepare last available row features (INR_per_unit and moving averages)
last_row = data.iloc[[-1]]  # keep as dataframe
X_last = last_row[features].values  # shape (1, n_features)
next_pred_inr = model.predict(X_last)[0]

st.markdown("---")
st.markdown(f"## 🔮 Predicted Next Day Gold Price: **₹{next_pred_inr:,.2f} {unit_choice}**")
st.caption(
    "This prediction is based on COMEX futures (USD/oz) converted to INR using the exchange rate above. "
    "Retail jeweller prices may differ due to local premiums and taxes."
)

# ---------- Optional: show comparison to a manual price (if user knows it) ----------
manual_price = st.number_input("Enter today's retail price to compare (optional)", value=0.0)
if manual_price > 0:
    st.write(f"- Retail price you entered: ₹{manual_price:,.2f} {unit_choice}")
    diff = manual_price - next_pred_inr
    pct = (diff / manual_price) * 100 if manual_price != 0 else 0
    st.write(f"- Difference (Retail - Predicted): ₹{diff:,.2f} ({pct:.2f}%)")

# ---------- Advice on improving accuracy ----------
st.markdown("### 🔧 Notes on accuracy & improvements")
st.write(
    "- This model uses converted historical futures prices (wholesale). Retail prices include premiums & taxes.\n"
    "- To better match local retail rates, use local exchange (MCX) or local retailer price feeds, or train directly on INR/gram retail data.\n"
    "- For better forecasting performance, try time-series models (LSTM, Prophet) and include exogenous features (currency, inflation, interest rates)."
)
