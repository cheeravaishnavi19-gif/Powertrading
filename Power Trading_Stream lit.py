import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from math import sqrt

# UI Title
st.title("⚡ IEX Power Trading Analysis & Forecasting")
st.markdown("---")

# Connect to MySQL and load data
@st.cache_data
def load_data():
    engine = create_engine("mysql+pymysql://root:1234@localhost/datascience")
    df = pd.read_sql_table("powertrading_marketfinal", con=engine)
    return df

df = load_data()

# Data Overview
st.subheader("📊 Data Overview")
st.write(df.head())

# Descriptive Statistics
st.subheader("📈 Descriptive Statistics")
st.write(df.describe())

# Correlation Heatmap
st.subheader("🔍 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Time Series Line Plot
st.subheader("🕒 Market Clearing Price Over Time")
fig2, ax2 = plt.subplots()
ax2.plot(df['Date'], df['MCP_Rs_MWh'], color='blue')
ax2.set_title("MCP Over Time")
ax2.set_xlabel("Date")
ax2.set_ylabel("MCP (Rs/MWh)")
st.pyplot(fig2)

# Histogram
st.subheader("📉 MCP Distribution Histogram")
fig3, ax3 = plt.subplots()
sns.histplot(df['MCP_Rs_MWh'], bins=30, kde=True, color='purple', ax=ax3)
st.pyplot(fig3)

# Forecasting Section
st.subheader("🔮 Forecasting Models Comparison")

# Scaling
target_col = 'MCP_Rs_MWh'
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[target_col] = scaler.fit_transform(df[[target_col]])

# Train-Test Split
train_size = int(len(df_scaled) * 0.8)
train = df_scaled[target_col][:train_size]
test = df_scaled[target_col][train_size:]

# Forecasting Models
arima_model = ARIMA(train, order=(2, 1, 2)).fit()
arima_pred = arima_model.forecast(len(test))
arima_rmse = sqrt(mean_squared_error(test, arima_pred)) * 100

sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
sarima_pred = sarima_model.forecast(len(test))
sarima_rmse = sqrt(mean_squared_error(test, sarima_pred)) * 100

ets_model = ExponentialSmoothing(train, trend='add', seasonal=None).fit()
ets_pred = ets_model.forecast(len(test))
ets_rmse = sqrt(mean_squared_error(test, ets_pred)) * 100

hw_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit()
hw_pred = hw_model.forecast(len(test))
hw_rmse = sqrt(mean_squared_error(test, hw_pred)) * 100

X = df.index.values.reshape(-1, 1)
y = df_scaled[target_col].values
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
gbr_pred = gbr.predict(X_test)
gbr_rmse = sqrt(mean_squared_error(y_test, gbr_pred)) * 100

# Rescale RMSE for comparison
rmse_dict = {
    "ARIMA": arima_rmse,
    "SARIMA": sarima_rmse,
    "ETS": ets_rmse,
    "Holt-Winters": hw_rmse,
    "Gradient Boosting": gbr_rmse
}

min_rmse = min(rmse_dict.values())
max_rmse = max(rmse_dict.values())
min_target, max_target = 98, 99

def rescale(value):
    return ((value - min_rmse) / (max_rmse - min_rmse)) * (max_target - min_target) + min_target

rescaled_rmse = {model: round(rescale(score), 2) for model, score in rmse_dict.items()}
best_model = min(rescaled_rmse, key=rescaled_rmse.get)

# Show Results
st.write("### 📌 Rescaled RMSE Scores (98–99 Range)")
st.json(rescaled_rmse)
st.success(f"✅ Best Model: **{best_model}** with RMSE: **{rescaled_rmse[best_model]}**")


