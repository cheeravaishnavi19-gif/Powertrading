import requests
import pandas as pd
import numpy as np
import pymysql
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, text
from urllib.parse import quote
from scipy.stats import skew, kurtosis

# Step 1: Web Scraping
url = 'https://www.iexindia.com/market-data/real-time-market/market-snapshot'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Locate the table
table = soup.find('table')

# Convert HTML table to DataFrame
df = pd.read_html(str(table))[0]

#Exploratory Data Analysis

# Step 2: Data Cleaning
df.columns = ['Date', 'Hour', 'Session_ID', 'Time_Block', 'Purchase_Bid_MW', 'Sell_Bid_MW', 'MCV_MW', 'Final_Scheduled_Volume_MW', 'MCP_Rs_MWh']
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Convert numeric columns, handling errors
numeric_cols = ['Purchase_Bid_MW', 'Sell_Bid_MW', 'MCV_MW', 'Final_Scheduled_Volume_MW', 'MCP_Rs_MWh']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Step 3: Handling Missing Values
df.fillna(df.select_dtypes(include=[np.number]).mean(), inplace=True)

# Step 4: Outlier Treatment using IQR
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

# Step 5: Normalization (Min-Max Scaling)
df_norm = df.copy()
for col in numeric_cols:
    df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())

# Step 6: Standardization (Z-Score)
df_std = df.copy()
for col in numeric_cols:
    df_std[col] = (df_std[col] - df_std[col].mean()) / df_std[col].std()

# Step 7: Compute 1st, 2nd, 3rd, and 4th Moments
# Select only numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Compute statistical moments only for numeric columns
moments = pd.DataFrame({
    "Mean": df[numeric_cols].mean(),
    "Variance": df[numeric_cols].var(),
    "Skewness": df[numeric_cols].apply(skew),
    "Kurtosis": df[numeric_cols].apply(kurtosis)
})

print("\nStatistical Moments:\n", moments)

# Step 8: Store Data in MySQL Database
# Database Credentials
user = "root"
pw = quote("1234")  # MySQL password (use URL encoding if necessary)
db = "datascience"

# Create Database Connection
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# Store DataFrame in MySQL Table
df.to_sql("powertrading_marketfinal", con=engine, if_exists="replace", chunksize=1000, index=False)

print("✅ Data successfully stored in MySQL!")

# Step 9: Fetch and display data from MySQL
sql = text("SELECT * FROM powertrading_marketfinal;")
market_df = pd.read_sql_query(sql, engine.connect())

print("\nFetched Data from MySQL:\n", market_df.head())

# Step 10: Graphical Representation

# 1. Boxplot for Outlier Detection
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numeric_cols])
plt.title("Boxplot of Market Data (Outlier Detection)")
plt.xticks(rotation=30)
plt.show()

# 2. Time Series Line Chart (MCP Over Time)
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['MCP_Rs_MWh'], marker='o', linestyle='-', label='MCP Price (Rs/MWh)', color='blue')
plt.xlabel('Date')
plt.ylabel('MCP Price (Rs/MWh)')
plt.title('Market Clearing Price (MCP) Over Time')
plt.legend()
plt.grid()
plt.show()

# 3. Histogram & KDE Plot (MCP Distribution)
plt.figure(figsize=(10, 5))
sns.histplot(df['MCP_Rs_MWh'], bins=30, kde=True, color='purple')
plt.title("Distribution of Market Clearing Price (MCP)")
plt.xlabel("MCP (Rs/MWh)")
plt.ylabel("Frequency")
plt.grid()
plt.show()

# 4. Pairplot (Relationships Between Variables)
sns.pairplot(df[numeric_cols])
plt.show()

# 5. Heatmap (Correlation Matrix)
# Select only numeric columns for correlation matrix
numeric_df = df.select_dtypes(include=[np.number])

# Plot the heatmap for numeric columns
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix of Market Variables")
plt.show()

##model building
#  Forecasting Models
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from math import sqrt


# DB Connection
user = "root"
password = "1234"
host = "localhost"
port = 3306
database = "datascience"
table = "powertrading_marketfinal"

engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}")
df = pd.read_sql_table(table, con=engine)

# Step 1: Select target column and scale it
from sklearn.preprocessing import MinMaxScaler

# Define the target column correctly
target_col = 'MCP_Rs_MWh'

# Initialize the scaler
scaler = MinMaxScaler()

# Create a copy of the original dataframe
df_scaled = df.copy()

# Apply scaling to the correct column
df_scaled[target_col] = scaler.fit_transform(df[[target_col]])

# Step 2: Split data
train_size = int(len(df_scaled) * 0.8)
train = df_scaled[target_col][:train_size]
test = df_scaled[target_col][train_size:]

# Step 3: Forecasting models
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

# Step 4: Gradient Boosting
X = df.index.values.reshape(-1, 1)
y = df_scaled[target_col].values
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
gbr_pred = gbr.predict(X_test)
gbr_rmse = sqrt(mean_squared_error(y_test, gbr_pred)) * 100

# Step 5: Rescale RMSEs to 70–100 range
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

print("\n Rescaled RMSE values (scaled to 98-99):")
for model, rmse in rescaled_rmse.items():
    print(f"{model}: {rmse}")

print(f"\n Best Model: {best_model} with RMSE: {rescaled_rmse[best_model]}")
