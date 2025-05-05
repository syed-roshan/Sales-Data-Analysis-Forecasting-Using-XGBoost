import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from xgboost import XGBRegressor

# Title
st.title("📈 Sales Forecasting using XGBoost")
st.markdown("This app analyzes actual sales data and forecasts the next 6 months using XGBoost.")

# Load dataset
df = pd.read_csv("C:/Users/rosha/Desktop/Qspiders Internship/project/Global-Superstore.csv", encoding='ISO-8859-1')

# Preprocess dates
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df = df.dropna(subset=['Order Date', 'Sales'])
df['Order Month'] = df['Order Date'].dt.to_period('M')
monthly_sales = df.groupby('Order Month')['Sales'].sum().reset_index()
monthly_sales['Order Month'] = pd.to_datetime(monthly_sales['Order Month'].astype(str))
monthly_sales['Month_Num'] = np.arange(len(monthly_sales))

# Display historical data
st.subheader("📊 Historical Monthly Sales")
st.dataframe(monthly_sales[['Order Month', 'Sales']])

# Train model
X = monthly_sales[['Month_Num']]
y = monthly_sales['Sales']
model = XGBRegressor()
model.fit(X, y)

# Forecast
future_months = np.arange(len(monthly_sales), len(monthly_sales) + 6).reshape(-1, 1)
predicted_sales = model.predict(future_months)

# Add artificial fluctuations for more visual realism
np.random.seed(42)
fluctuation = np.random.normal(loc=0, scale=0.05, size=len(predicted_sales))  # ±5% noise
predicted_sales = predicted_sales * (1 + fluctuation)

future_dates = pd.date_range(monthly_sales['Order Month'].max() + pd.offsets.MonthBegin(1), periods=6, freq='MS')
forecast_df = pd.DataFrame({
    'Month': future_dates,
    'Forecasted Sales': predicted_sales.astype(int)
})

# Combine for visual chart
full_df = pd.concat([
    monthly_sales[['Order Month', 'Sales']].rename(columns={'Order Month': 'Month', 'Sales': 'Value'}).assign(Type='Historical'),
    forecast_df.rename(columns={'Forecasted Sales': 'Value'}).assign(Type='Forecast')
])

# Plot
st.subheader("📉 Monthly Sales Forecast")
fig = px.line(full_df, x='Month', y='Value', color='Type', markers=True,
              title="Sales Forecast - Realistic 6 Month Projection")
st.plotly_chart(fig)

# Table
st.subheader("🔮 Forecasted Sales Table")
st.dataframe(forecast_df)

st.caption("Forecast enhanced with realistic fluctuation | Powered by XGBoost")
