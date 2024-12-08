import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import date, timedelta
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# Title
st.title("Stock Insights: Prediction & Technical Analysis")

# Sidebar inputs
ticker = st.sidebar.text_input("Enter stock ticker", "AAPL")

# Select number of years to predict
prediction_years = st.sidebar.slider("Select years to predict:", 1, 4, 1)
forecast_period = prediction_years * 365

# Select historical data range
historical_range = st.sidebar.radio(
    "Select historical data range (years):",
    options=[10, 20, 30, 40],
    index=0
)

# Sidebar for additional tickers to compare
comparison_tickers = st.sidebar.text_area(
    "Enter other tickers to compare (comma-separated, max 5):", 
    "MSFT,GOOGL,AMZN"
).split(',')

comparison_tickers = [ticker.strip() for ticker in comparison_tickers if ticker.strip()]

if len(comparison_tickers) > 5:
    comparison_tickers = comparison_tickers[:5]  # Limit to 5 tickers

# Function to download data with fallback mechanism
@st.cache_data
def download_data(ticker, years):
    today = date.today()
    start_date = today - timedelta(days=years * 365)

    while years > 0:
        try:
            data = yf.download(ticker, start=start_date, end=today)
            if not data.empty:
                return data, years
        except Exception as e:
            st.warning(f"Error fetching data: {e}")

        # Reduce the range and retry
        years -= 10
        start_date = today - timedelta(days=years * 365)

    return pd.DataFrame(), 0

# Fetch data for the main ticker
data, actual_years = download_data(ticker, historical_range)

if data.empty:
    st.error("No data available for the selected stock.")
else:
    if actual_years < historical_range:
        st.warning(
            f"Data for the last {historical_range} years is not available. Using data from the last {actual_years} years instead."
        )

    # Prepare data for Prophet
    data = data[['Adj Close']].reset_index()
    data.columns = ['ds', 'y']
    data['ds'] = pd.to_datetime(data['ds'])
    data['y'] = pd.to_numeric(data['y'], errors='coerce')
    data = data.dropna()

    # Initialize and train the Prophet model
    m = Prophet()
    m.fit(data)

    # Create future dataframe
    future = m.make_future_dataframe(periods=forecast_period)

    # Forecast
    forecast = m.predict(future)

    # Price forecast section
    st.subheader(f"Price Forecast for {ticker}")
    
    # Current price and forecasted price after the selected forecast period
    current_price = data['y'].iloc[-1]
    
    st.write(f"**Current Price**: {current_price:.2f}")

    # Plot forecast
    fig1 = m.plot(forecast)
    st.pyplot(fig1)

    st.subheader("How to interpret the forecast plot:")
    st.markdown(
        "- **Blue line**: Predicted stock price.\n"
        "- **Shaded area**: Uncertainty intervals (confidence intervals)."
    )

    # Plot components
    st.subheader("Forecast Components")
    fig2 = m.plot_components(forecast)
    st.pyplot(fig2)

    st.subheader("How to interpret the components:")
    st.markdown(
        "- **Trend**: Shows the overall direction of the stock price over time.\n"
        "- **Yearly seasonality**: Highlights recurring annual patterns in the stock price.\n"
        "- **Weekly seasonality**: Displays weekly patterns or variations in the stock price."
    )

    # Compare with other stocks in the same period
    st.subheader(f"Price Comparison: {ticker} vs. Other Stocks")

    # Download data for comparison tickers
    comparison_data = {}
    for comp_ticker in comparison_tickers:
        comp_data, _ = download_data(comp_ticker, historical_range)
        if not comp_data.empty:
            comparison_data[comp_ticker] = comp_data[['Adj Close']].reset_index()
            comparison_data[comp_ticker].columns = ['ds', comp_ticker]
            comparison_data[comp_ticker]['ds'] = pd.to_datetime(comparison_data[comp_ticker]['ds'])
    
    # Merge all data for comparison
    merged_data = data[['ds', 'y']].copy()
    for comp_ticker, comp_data in comparison_data.items():
        merged_data = merged_data.merge(comp_data[['ds', comp_ticker]], on='ds', how='left')

    # Plot comparison data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(merged_data['ds'], merged_data['y'], label=f"{ticker} Price", color="black", linewidth=1)
    for comp_ticker in comparison_tickers:
        if comp_ticker in merged_data.columns:
            ax.plot(merged_data['ds'], merged_data[comp_ticker], label=f"{comp_ticker} Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # Technical Indicators
    st.subheader("Technical Indicators")
    st.markdown(
        "This section displays several technical indicators to analyze the historical data of the selected stock."
    )

    # Calculate Simple Moving Average (SMA), Exponential Moving Average (EMA), and Bollinger Bands
    data['SMA_50'] = data['y'].rolling(window=50).mean()
    data['EMA_50'] = data['y'].ewm(span=50, adjust=False).mean()

    # Calculate Bollinger Bands
    data['Bollinger_Mid'] = data['SMA_50']
    data['Bollinger_Upper'] = data['Bollinger_Mid'] + 2 * data['y'].rolling(window=50).std()
    data['Bollinger_Lower'] = data['Bollinger_Mid'] - 2 * data['y'].rolling(window=50).std()

    # Plot the indicators with thinner lines
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['ds'], data['y'], label="Stock Price", color="black", linewidth=1)
    ax.plot(data['ds'], data['SMA_50'], label="50-Day SMA", color="orange", linewidth=1)
    ax.plot(data['ds'], data['EMA_50'], label="50-Day EMA", color="red", linewidth=1)
    ax.fill_between(data['ds'], data['Bollinger_Lower'], data['Bollinger_Upper'], color="gray", alpha=0.2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    st.subheader("How to interpret the technical indicators:")
    st.markdown(
        "- **SMA (Simple Moving Average)**: A simple average of the stock price over a specified period, often used to identify trends.\n"
        "- **EMA (Exponential Moving Average)**: A weighted moving average that gives more importance to recent prices.\n"
        "- **Bollinger Bands**: Two bands (upper and lower) that represent the price volatility. The stock is considered overbought when it is near the upper band and oversold near the lower band."
    )

    # Ichimoku Cloud Analysis
    st.subheader("Ichimoku Cloud Analysis")
    st.markdown(
        "The Ichimoku Cloud is a technical analysis tool that defines support and resistance levels, identifies trend direction, and provides buy and sell signals. It consists of five lines: Tenkan-sen (Conversion Line), Kijun-sen (Base Line), Senkou Span A, Senkou Span B, and Chikou Span."
    )

    # Calculate Ichimoku Cloud components
    high_9 = data['y'].rolling(window=9).max()
    low_9 = data['y'].rolling(window=9).min()
    data['Tenkan_sen'] = (high_9 + low_9) / 2

    high_26 = data['y'].rolling(window=26).max()
    low_26 = data['y'].rolling(window=26).min()
    data['Kijun_sen'] = (high_26 + low_26) / 2

    data['Senkou_Span_A'] = ((data['Tenkan_sen'] + data['Kijun_sen']) / 2).shift(26)
    high_52 = data['y'].rolling(window=52).max()
    low_52 = data['y'].rolling(window=52).min()
    data['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)
    data['Chikou_Span'] = data['y'].shift(-26)

    # Plot Ichimoku Cloud with thinner lines
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['ds'], data['y'], label="Stock Price", color="black", linewidth=1)
    ax.plot(data['ds'], data['Tenkan_sen'], label="Tenkan-sen (Conversion Line)", color="blue", linewidth=1)
    ax.plot(data['ds'], data['Kijun_sen'], label="Kijun-sen (Base Line)", color="red", linewidth=1)
    ax.fill_between(data['ds'], data['Senkou_Span_A'], data['Senkou_Span_B'], color="green", alpha=0.2)
    ax.plot(data['ds'], data['Chikou_Span'], label="Chikou Span (Lagging Line)", color="purple", linewidth=1)
    ax.legend()  # Add the legend for Senkou Span A and B
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    st.pyplot(fig)

    st.subheader("How to interpret the Ichimoku Cloud:")
    st.markdown(
    """
    - **Tenkan-sen (Conversion Line)**: A fast-moving average.
    - **Kijun-sen (Base Line)**: A slower-moving average.
    - **Senkou Span A & B (Cloud)**: The space between these two lines forms the Ichimoku Cloud, which helps identify trends. A stock is in an uptrend if the price is above the cloud and in a downtrend if below.
    - **Chikou Span (Lagging Line)**: Shows the current price relative to historical prices.
    """
    )
