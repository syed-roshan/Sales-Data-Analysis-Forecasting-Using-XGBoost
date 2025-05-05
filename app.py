import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import plotly.express as px
from fpdf import FPDF
from io import BytesIO

# Set page config
st.set_page_config(page_title="Sales Dashboard", layout="wide")

# Custom CSS for improved UI
st.markdown("""
    <style>
    body {
        background-color: #f4f7fc;
        font-family: 'Arial', sans-serif;
    }
    .custom-title {
        font-size: 32px;
        font-weight: 700;
        color: white;
        margin-bottom: 0;
        text-align: center;
    }
    .custom-subtitle {
        font-size: 18px;
        color: white;
        margin-top: 10px;
        margin-bottom: 30px;
        text-align: center;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .kpi-container {
        display: flex;
        justify-content: space-between;
        background-color: #ffffff;
        padding: 15px 10px;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .kpi-box {
        text-align: center;
        width: 23%;
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
        transition: background-color 0.3s;
    }
    .kpi-box:hover {
        background-color: #e1f3fc;
    }
    .kpi-label {
        font-size: 14px;
        font-weight: 600;
        color: #2d3e50;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 18px;
        color: #3498db;
        font-weight: bold;
    }
    .footer {
        padding: 10px 20px;
        text-align: center;
        color: white;
        font-size: 14px;
        font-weight: 400;
    }
    </style>
""", unsafe_allow_html=True)

# Load dataset
df = pd.read_csv("C:/Users/rosha/Desktop/Qspiders Internship/project/Global-Superstore.csv", encoding='ISO-8859-1')


# Data Preprocessing
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df = df.dropna(subset=['Order Date', 'Sales'])
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Month Name'] = df['Order Date'].dt.strftime('%B')
df['Order Month'] = df['Order Date'].dt.to_period('M')

# Sidebar Filters
st.sidebar.header("🔎 Filter Data")
regions = st.sidebar.multiselect("Select Region", options=df['Region'].unique(), default=df['Region'].unique())
segments = st.sidebar.multiselect("Select Segment", options=df['Segment'].unique(), default=df['Segment'].unique())
categories = st.sidebar.multiselect("Select Category", options=df['Category'].unique(), default=df['Category'].unique())

# Apply filters
filtered_df = df[(df['Region'].isin(regions)) & (df['Segment'].isin(segments)) & (df['Category'].isin(categories))]

# Custom Header
st.markdown('<div class="custom-title"> Sales Forecasting and Insights Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-subtitle">Interactive data exploration and 6-month sales forecast using Global Superstore dataset.</div>', unsafe_allow_html=True)
st.markdown("---")

# KPI values
total_sales = filtered_df['Sales'].sum()
total_profit = filtered_df['Profit'].sum()
unique_orders = filtered_df['Order ID'].nunique()
top_customer = filtered_df.groupby('Customer Name')['Sales'].sum().idxmax()

# Render KPIs
st.markdown(f"""
    <div class='kpi-container'>
        <div class='kpi-box'>
            <div class='kpi-label'>Total Sales</div>
            <div class='kpi-value'>Rs.{total_sales:,.2f}</div>
        </div>
        <div class='kpi-box'>
            <div class='kpi-label'>Total Profit</div>
            <div class='kpi-value'>Rs.{total_profit:,.2f}</div>
        </div>
        <div class='kpi-box'>
            <div class='kpi-label'>Unique Orders</div>
            <div class='kpi-value'>{unique_orders}</div>
        </div>
        <div class='kpi-box'>
            <div class='kpi-label'>Top Customer</div>
            <div class='kpi-value'>{top_customer}</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Visualizations
st.subheader("Sales by Region")
region_sales = filtered_df.groupby('Region')['Sales'].sum().reset_index()
fig1 = px.pie(region_sales, names='Region', values='Sales', title='Sales Distribution by Region')
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Monthly Sales Trend")
monthly_sales = filtered_df.groupby('Order Month')['Sales'].sum().reset_index()
monthly_sales['Order Month'] = pd.to_datetime(monthly_sales['Order Month'].astype(str))
fig2 = px.line(monthly_sales, x='Order Month', y='Sales', title='Monthly Sales Over Time')
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Top 10 Customers by Sales")
top_customers = filtered_df.groupby('Customer Name')['Sales'].sum().sort_values(ascending=False).head(10).reset_index()
fig3 = px.bar(top_customers, x='Sales', y='Customer Name', orientation='h', title='Top 10 Customers')
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Sales by Category and Sub-Category")
category_table = filtered_df.groupby(['Category', 'Sub-Category'])['Sales'].sum().sort_values(ascending=False).reset_index()
st.dataframe(category_table)

# # Forecasting
# st.subheader("Sales Forecast for Next 6 Months (XGBoost)")
# forecast_data = filtered_df.groupby('Order Month')['Sales'].sum().reset_index()
# forecast_data['Order Month'] = pd.to_datetime(forecast_data['Order Month'].astype(str))
# forecast_data = forecast_data.sort_values('Order Month')
# forecast_data['Month_Num'] = np.arange(len(forecast_data))

# X = forecast_data[['Month_Num']]
# y = forecast_data['Sales']
# model = XGBRegressor()
# model.fit(X, y)

# future_months = np.arange(len(forecast_data), len(forecast_data) + 6).reshape(-1, 1)
# predicted_sales = model.predict(future_months)
# future_dates = pd.date_range(forecast_data['Order Month'].max() + pd.offsets.MonthBegin(1), periods=6, freq='MS')

# forecast_df = pd.DataFrame({
#     'Month': future_dates,
#     'Forecasted Sales': predicted_sales
# })

# fig4 = px.line(forecast_df, x='Month', y='Forecasted Sales', markers=True, title='📈 Forecasted Sales (Next 6 Months)')
# st.plotly_chart(fig4, use_container_width=True)

# Business Suggestions
st.subheader("💼 Business Suggestions")
st.markdown("""
- Regions like **Asia Pacific** or **Africa** might need targeted promotions if underperforming.
- Segment analysis suggests investing more in **Corporate or Home Office** customers with high purchase patterns.
- Top customers indicate loyalty segments. Create personalized campaigns for them.
""")

# CSV Download
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("📅 Download Filtered CSV", data=csv, file_name='filtered_sales_data.csv', mime='text/csv')

# PDF Generator

def create_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, "Sales Report Summary", ln=True, align='C')
    pdf.set_font("Arial", '', 12)
    pdf.ln(10)
    pdf.cell(200, 10, f"Total Sales: Rs.{total_sales:,.2f}", ln=True)
    pdf.cell(200, 10, f"Total Profit: Rs.{total_profit:,.2f}", ln=True)
    pdf.cell(200, 10, f"Unique Orders: {unique_orders}", ln=True)
    pdf.cell(200, 10, f"Top Customer: {top_customer}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "Business Suggestions:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 10, "- Promote in underperforming regions (e.g., Africa)\n- Leverage Corporate & Home Office segments for targeted sales\n- Focus loyalty programs on high-value customers")

    return BytesIO(pdf.output(dest='S').encode('latin1'))

pdf_buffer = create_pdf()
st.download_button(label="📄 Download PDF Report", data=pdf_buffer, file_name="sales_report.pdf", mime="application/pdf")

# Footer
st.markdown("<div class='footer'>✨ Developed by Syed Roshan</div>", unsafe_allow_html=True)
