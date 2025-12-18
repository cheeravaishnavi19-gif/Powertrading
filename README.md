# Powertrading

# ⚡ Power Trading – Intraday Market Analysis & Forecasting

## 1️⃣ Background and Overview

Electricity power markets operate in a highly dynamic environment where demand and supply must be balanced in real time. Intraday power trading allows participants to adjust their positions closer to real time, making accurate price and volume analysis critical.

This project focuses on analyzing **Intraday Power Trading data from the Indian Energy Exchange (IEX)** at a **15-minute time block level**. The goal is to transform raw market data into meaningful insights and reliable short-term price forecasts that can support traders, grid operators, and energy analysts in decision-making.

The project delivers an **end-to-end analytics pipeline**, covering data collection, preprocessing, exploratory analysis, forecasting, and visualization through an interactive Streamlit dashboard.

---

## 2️⃣ Data Structure and Overview

### Data Source

* **Platform:** Indian Energy Exchange (IEX)
* **Collection Method:** Web scraping (Market Snapshot)
* **Market Type:** Intraday / Real-Time Market
* **Granularity:** 15-minute time blocks

### Dataset Structure

Each record represents a single 15-minute trading interval and includes:

* **Date** – Trading date
* **Hour** – Hour of the day
* **Time_Block** – 15-minute interval
* **Purchase_Bid_MW** – Total buy bids submitted (MW)
* **Sell_Bid_MW** – Total sell bids submitted (MW)
* **Market_Cleared_Volume (MCV_MW)** – Volume cleared by the market (MW)
* **Final_Scheduled_Volume_MW** – Final scheduled electricity volume (MW)
* **Market_Clearing_Price (MCP_Rs_MWh)** – Price at which supply meets demand

### Data Processing

* Missing values handled using mean imputation
* Outliers treated using the **Interquartile Range (IQR)** method
* Numeric features normalized and standardized for analysis and modeling
* Cleaned data stored in **MySQL** for scalability and reuse

---

## 3️⃣ Executive Summary

* Intraday power prices and volumes show **strong time-of-day dependency**
* Peak hours exhibit **higher volatility** in both bids and prices
* Market cleared volume closely follows final scheduled volume
* Moderate correlation exists between **demand pressure and price levels**
* Among multiple forecasting approaches, **statistical and machine learning models provide reliable short-term MCP predictions**

The project demonstrates that combining **statistical analysis with forecasting models** significantly improves understanding and prediction of intraday electricity prices.

---

## 4️⃣ Insights Deep Dive

### 🔹 Intraday Trading Patterns

* Buy and sell bids fluctuate across time blocks, indicating changing demand and supply conditions
* Morning and evening peak hours show increased bidding activity

### 🔹 Price Behavior (MCP)

* MCP displays noticeable spikes during high-demand periods
* Distribution analysis shows price clustering with occasional extreme values, indicating volatility

### 🔹 Volume Dynamics

* Market Cleared Volume and Final Scheduled Volume are strongly correlated
* High sell bids do not always translate to high cleared volume, highlighting market competition

### 🔹 Correlation Analysis

* Positive correlation between cleared volume and scheduled volume
* Moderate relationship between MCP and volume, confirming demand–supply influence on pricing

### 🔹 Forecasting Performance

Models evaluated:

* ARIMA
* SARIMA
* ETS
* Holt-Winters
* Gradient Boosting Regressor

All models were evaluated using RMSE, and the **best-performing model was automatically selected**. Machine learning and seasonal models performed better during volatile periods.

---

## 5️⃣ Recommendations

### For Power Traders

* Focus on peak-hour bidding strategies where price volatility is highest
* Use short-term MCP forecasts to optimize intraday buy/sell positions

### For Grid Operators

* Monitor high-volatility time blocks to improve load balancing
* Use cleared vs scheduled volume gaps to anticipate congestion risks

### For Analysts & Decision Makers

* Combine statistical forecasting with machine learning for better accuracy
* Automate real-time data ingestion for faster market response

### Future Scope

* Integrate deep learning models (LSTM/GRU)
* Enable live market data updates
* Deploy dashboard on cloud platforms (AWS/GCP)
* Extend analysis to Day-Ahead and Term-Ahead markets

---

## 👩‍💻 Author

**Vaishnavi Cheera**
Master’s in Business Analytics
Focus Areas: Energy Analytics | Forecasting | Data Science

---

⭐ *This project demonstrates a real-world application of analytics and forecasting in electricity markets, bridging business understanding with technical implementation.*

## 📸 Dashboard Preview

<img width="1735" height="736" alt="Powertrading_dashborad" src="https://github.com/user-attachments/assets/f1de381d-6fb1-4232-9758-fa9bfe4aaed0" />


