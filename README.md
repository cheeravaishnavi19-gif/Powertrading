# Powertrading

# ⚡ Power Trading – Intraday Market Analysis & Forecasting

## 📌 Project Overview

This project focuses on **intraday power trading analysis** using data from the **Indian Energy Exchange (IEX)**. Electricity markets are highly dynamic, operating on short time blocks (15-minute intervals), where demand–supply balance directly impacts prices and volumes.

The objective of this project is to:

* Analyze intraday power trading behavior
* Understand bid dynamics, cleared volumes, and price movements
* Build and compare forecasting models for **Market Clearing Price (MCP)**
* Present insights through an **interactive Streamlit dashboard**

This project combines **data engineering, exploratory data analysis (EDA), statistical analysis, forecasting, and visualization** to support data-driven decision-making in energy markets.

---

## 🧠 Business Problem

Power traders, grid operators, and analysts need **accurate short-term price forecasts** and **clear visibility into market behavior** to:

* Optimize buy/sell bidding strategies
* Anticipate price spikes and volatility
* Improve intraday operational decisions

Manual analysis of high-frequency power market data is inefficient. This project automates data collection, analysis, and forecasting into a single analytical pipeline.

---

## 📂 Data Source

* **Source:** Indian Energy Exchange (IEX)
* **Method:** Web scraping from IEX market snapshot
* **Granularity:** 15-minute time blocks
* **Market Focus:** Intraday / Real-Time Market

### Key Features

* Purchase Bid (MW)
* Sell Bid (MW)
* Market Cleared Volume (MW)
* Final Scheduled Volume (MW)
* Market Clearing Price (Rs/MWh)
* Date, Hour, and Time Block

---

## 🔧 Tech Stack & Tools

* **Programming:** Python
* **Libraries:** Pandas, NumPy, BeautifulSoup, Matplotlib, Seaborn, Scikit-learn, Statsmodels
* **Database:** MySQL (SQLAlchemy integration)
* **Visualization:** Matplotlib, Seaborn
* **Dashboard:** Streamlit
* **Forecasting Models:** ARIMA, SARIMA, ETS, Holt-Winters, Gradient Boosting

---

## 🔄 Project Workflow

### 1️⃣ Data Collection

* Web scraping of real-time market data from IEX
* HTML tables parsed using BeautifulSoup
* Data converted into Pandas DataFrames

### 2️⃣ Data Cleaning & Preprocessing

* Standardized column names
* Datetime parsing for Date and Time Blocks
* Numeric conversion with error handling
* Missing value treatment using mean imputation
* Outlier treatment using **Interquartile Range (IQR)**

### 3️⃣ Feature Scaling

* **Min-Max Normalization** for modeling
* **Z-score Standardization** for statistical analysis

### 4️⃣ Statistical Analysis

Computed **statistical moments** for market variables:

* Mean
* Variance
* Skewness
* Kurtosis

These help understand volatility, asymmetry, and distribution behavior of power prices and volumes.

---

## 📊 Exploratory Data Analysis (EDA)

Key visual analyses include:

* **Boxplots:** Detect volatility and extreme bid/price values
* **Time Series Plots:** MCP movement across time blocks
* **Histograms & KDE:** Distribution of market clearing prices
* **Pairplots:** Relationships between bids, volumes, and prices
* **Correlation Heatmap:** Strength of relationships among variables

### Key Insights

* Strong relationship between cleared volume and scheduled volume
* Time-of-day effects visible in both price and volume
* Higher volatility during peak demand hours
* Moderate correlation between MCP and trading volume

---

## 🗄️ Database Integration

* Cleaned data stored in **MySQL** database
* Table: `powertrading_marketfinal`
* Enables reusable analysis, scalability, and dashboard connectivity

---

## 🔮 Forecasting Models

The target variable for forecasting is **Market Clearing Price (MCP)**.

### Models Implemented

* **ARIMA:** Captures short-term linear trends
* **SARIMA:** Incorporates seasonality
* **ETS (Error-Trend-Seasonality):** Component-based forecasting
* **Holt-Winters:** Trend + seasonality smoothing
* **Gradient Boosting Regressor:** Machine learning-based approach

### Model Evaluation

* Train-Test split: 80%-20%
* Metric: **RMSE (Root Mean Squared Error)**
* RMSE scores rescaled to a common range (98–99) for easier comparison

✅ **Best-performing model is automatically selected** based on lowest RMSE.

---

## 📈 Interactive Dashboard (Streamlit)

The Streamlit app provides:

* Data preview and descriptive statistics
* Correlation heatmap
* MCP time series visualization
* MCP distribution analysis
* Forecasting model comparison with best model highlight

This enables **non-technical users** to interact with the analysis easily.

---

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone <repository-url>
cd power-trading-analysis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup MySQL

* Create a database (e.g., `datascience`)
* Update MySQL credentials in the Python scripts

### 4. Run Data Pipeline

```bash
python Power Trading Project.py
```

### 5. Launch Streamlit App

```bash
streamlit run Power Trading_Stream lit.py
```

---

## 🚀 Future Enhancements

* Integration with live API instead of scraping
* Advanced deep learning models (LSTM, GRU)
* Intraday demand forecasting
* Deployment on cloud (AWS/GCP)
* Automated daily data refresh

---

## 👩‍💻 Author

**Vaishnavi Cheera**
Master’s in Business Analytics
Focus: Energy Analytics | Forecasting | Data Science

---

## ⭐ Key Takeaway

This project demonstrates an **end-to-end analytics pipeline**—from raw market data to actionable insights and forecasts—tailored for **real-world power trading and energy market analysis**.

## 📸 Dashboard Preview

<img width="1735" height="736" alt="Powertrading_dashborad" src="https://github.com/user-attachments/assets/f0074a32-f218-4821-82c3-5663bad8717c" />

