# App Stock Analysis
import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import date, timedelta
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import seaborn as sns
import plotly.graph_objects as go

# Configurazione della pagina
st.set_page_config(page_title="Stock Insights: Prediction & Technical Analysis", page_icon="📈", layout="wide")

st.markdown("""
    <p style="font-size: 12px; text-align: center;">
        Created by: <a href="https://www.linkedin.com/in/luca-girlando-775463302/" target="_blank">Luca Girlando</a>
    </p>
""", unsafe_allow_html=True)

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



# Verifica se 'Adj Close' è presente nel DataFrame
if 'Adj Close' in data.columns:
    data = data[['Adj Close']].reset_index()
else:
    raise KeyError("'Adj Close' column not found in the DataFrame. Please check the input data.")
    
st.write("Columns in DataFrame:", data.columns)

if 'Adj Close' in data.columns:
    data = data[['Adj Close']].reset_index()
elif 'Close' in data.columns:  # Usa una colonna alternativa
    st.warning("Using 'Close' column instead of 'Adj Close'.")
    data = data[['Close']].reset_index()
else:
    st.error("'Adj Close' or 'Close' column not found in the DataFrame.")
    st.stop()  # Ferma l'esecuzione dell'app se nessuna colonna è valida

if data.empty:
    st.error("The data file is empty or not loaded correctly. Please check the input.")
    st.stop()

try:
    data = data[['Adj Close']].reset_index()
except KeyError:
    st.error("The 'Adj Close' column is missing. Please check your data source.")
    st.stop()

    

    
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

    st.markdown(
        "- **Blue line**: Predicted stock price.\n"
        "- **Shaded area**: Uncertainty intervals (confidence intervals)."
    )

    # Plot components
    st.subheader("Forecast Components")
    fig2 = m.plot_components(forecast)
    st.pyplot(fig2)

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

    # Create Plotly figure
fig = go.Figure()

# Plot primary ticker price
fig.add_trace(go.Scatter(
    x=merged_data['ds'], 
    y=merged_data['y'], 
    mode='lines', 
    name=f"{ticker} Price", 
    line=dict(color="black", width=1)
))

# Plot comparison tickers
for comp_ticker in comparison_tickers:
    if comp_ticker in merged_data.columns:
        fig.add_trace(go.Scatter(
            x=merged_data['ds'], 
            y=merged_data[comp_ticker], 
            mode='lines', 
            name=f"{comp_ticker} Price"
        ))

# Layout customization
fig.update_layout(
    title=f"Comparison of {ticker} and Other Tickers",
    xaxis_title="Date",
    yaxis_title="Price",
    legend_title="Tickers",
    template="plotly_white",
    hovermode="x unified",  # Unified hover for comparison
    width=900,
    height=600
)

# Display in Streamlit
st.plotly_chart(fig)

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

    # Plot the indicators with Plotly
fig = go.Figure()

# Stock Price
fig.add_trace(go.Scatter(
    x=data['ds'], 
    y=data['y'], 
    mode='lines', 
    name="Stock Price", 
    line=dict(color="black", width=1)
))

# 50-Day SMA
fig.add_trace(go.Scatter(
    x=data['ds'], 
    y=data['SMA_50'], 
    mode='lines', 
    name="50-Day SMA", 
    line=dict(color="orange", width=1)
))

# 50-Day EMA
fig.add_trace(go.Scatter(
    x=data['ds'], 
    y=data['EMA_50'], 
    mode='lines', 
    name="50-Day EMA", 
    line=dict(color="red", width=1)
))

# Bollinger Bands (Shaded Area)
fig.add_trace(go.Scatter(
    x=data['ds'], 
    y=data['Bollinger_Upper'], 
    mode='lines', 
    name="Bollinger Upper", 
    line=dict(color="gray", width=0.5),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=data['ds'], 
    y=data['Bollinger_Lower'], 
    mode='lines', 
    name="Bollinger Lower", 
    line=dict(color="gray", width=0.5),
    fill='tonexty',  # Fill between the upper and lower bands
    fillcolor='rgba(128, 128, 128, 0.2)',  # Transparent gray
    showlegend=False
))

# Layout customization
fig.update_layout(
    title="Indicators: SMA, EMA, and Bollinger Bands",
    xaxis_title="Date",
    yaxis_title="Price",
    legend_title="Legend",
    template="plotly_white",  # Clean aesthetic theme
    hovermode="x unified",  # Unified hover for easier comparison
    width=900,
    height=600
)

# Display in Streamlit
st.plotly_chart(fig)

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

    # Plot Ichimoku Cloud with Plotly
fig = go.Figure()

# Stock Price
fig.add_trace(go.Scatter(
    x=data['ds'], 
    y=data['y'], 
    mode='lines', 
    name="Stock Price", 
    line=dict(color="black", width=1)
))

# Tenkan-sen (Conversion Line)
fig.add_trace(go.Scatter(
    x=data['ds'], 
    y=data['Tenkan_sen'], 
    mode='lines', 
    name="Tenkan-sen (Conversion Line)", 
    line=dict(color="blue", width=1)
))

# Kijun-sen (Base Line)
fig.add_trace(go.Scatter(
    x=data['ds'], 
    y=data['Kijun_sen'], 
    mode='lines', 
    name="Kijun-sen (Base Line)", 
    line=dict(color="red", width=1)
))

# Senkou Span A and B (Ichimoku Cloud)
fig.add_trace(go.Scatter(
    x=data['ds'], 
    y=data['Senkou_Span_A'], 
    mode='lines', 
    name="Senkou Span A", 
    line=dict(color="green", width=0.5),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=data['ds'], 
    y=data['Senkou_Span_B'], 
    mode='lines', 
    name="Senkou Span B", 
    line=dict(color="orange", width=0.5),
    showlegend=False,
    fill='tonexty',  # Fill the area between Senkou Span A and B
    fillcolor='rgba(0, 255, 0, 0.2)'  # Transparent green
))

# Chikou Span (Lagging Line)
fig.add_trace(go.Scatter(
    x=data['ds'], 
    y=data['Chikou_Span'], 
    mode='lines', 
    name="Chikou Span (Lagging Line)", 
    line=dict(color="purple", width=1)
))

# Layout customization
fig.update_layout(
    title="Ichimoku Cloud",
    xaxis_title="Date",
    yaxis_title="Price",
    legend_title="Legend",
    template="plotly_white",  # Clean aesthetic theme
    hovermode="x unified",  # Unified hover for easier comparison
    width=900,
    height=600
)

# Display in Streamlit
st.plotly_chart(fig)

# Add interpretation guide for Ichimoku Cloud
st.markdown(
    """
    - **Tenkan-sen (Conversion Line):** A short-term trend indicator. It shows the average of the highest high and lowest low over the past 9 periods. 
      - When the price is above the Tenkan-sen, it suggests a bullish trend.
      - When the price is below, it suggests a bearish trend.

    - **Kijun-sen (Base Line):** A medium-term trend indicator. It calculates the average of the highest high and lowest low over the past 26 periods.
      - This line acts as a dynamic support or resistance level.
      - A crossover between the Tenkan-sen and Kijun-sen can generate buy (bullish crossover) or sell (bearish crossover) signals.

    - **Senkou Span A & B (Cloud):** The "Cloud" is the area between Span A and Span B.
      - **Bullish Cloud:** Span A is above Span B, and the cloud is typically green.
      - **Bearish Cloud:** Span B is above Span A, and the cloud is typically red.
      - The thickness of the cloud represents the strength of the trend. A thin cloud may indicate weaker support/resistance, while a thicker cloud suggests stronger support/resistance.
      - If the price is above the cloud, the trend is bullish. If below, it is bearish.

    - **Chikou Span (Lagging Line):** The current price shifted 26 periods into the past.
      - If the Chikou Span is above the price, it confirms a bullish trend.
      - If it is below, it confirms a bearish trend.

    **Key Signals:**
    1. **Bullish Signal:** Tenkan-sen crosses above Kijun-sen, and the price is above the cloud.
    2. **Bearish Signal:** Tenkan-sen crosses below Kijun-sen, and the price is below the cloud.
    3. **Neutral Trend:** The price is within the cloud, suggesting consolidation or indecision.

    The Ichimoku Cloud helps visualize support, resistance, momentum, and trend direction all in one glance, making it a powerful tool for technical analysis.
    """
)
    
    # Descriptive Statistics and Correlation
st.subheader("Exploratory Data Analysis")
def descriptive_statistics(data):
        st.write("Descriptive Statistics:")
        st.write(data.describe())

def correlation_matrix(data):
        st.write("Correlation Matrix:")
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
        st.pyplot(plt)

descriptive_statistics(data)
correlation_matrix(data)

def calculate_metrics(data):
        returns = data.pct_change().dropna()
        annualized_return = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility
        max_drawdown = (data / data.cummax() - 1).min()

        st.write("Risk and Return Metrics:")
        st.write(f"Average Annualized Return: {annualized_return.mean():.2%}")
        st.write(f"Average Annualized Volatility: {volatility.mean():.2%}")
        st.write(f"Average Sharpe Ratio: {sharpe_ratio.mean():.2f}")
        st.write(f"Average Maximum Drawdown: {max_drawdown.mean():.2%}")

        st.subheader("Risk and Return Metrics")
        calculate_metrics(data)
    
    # Interpretation of Descriptive Statistics and Heatmap
"""
The descriptive statistics table provides key insights into the distribution and variability of each variable:

- **Mean**: The average value, indicating the central tendency of the data.
- **Standard Deviation (std)**: Measures the spread of the data. A higher value indicates greater variability.
- **Min/Max**: The minimum and maximum values, showing the range of the data.
- **25%, 50%, 75% (Quartiles)**: Divide the data into four equal parts, helping to understand the distribution.

The correlation heatmap visually represents the relationships between variables:

- **Correlation Coefficients**: Values range from -1 to 1.
    - A coefficient close to 1 implies a strong positive correlation (as one variable increases, the other also increases).
    - A coefficient close to -1 implies a strong negative correlation (as one variable increases, the other decreases).
    - A coefficient near 0 indicates little or no linear relationship.
- **Color Intensity**: Darker or brighter shades represent stronger correlations. For instance, darker red indicates strong negative correlation, while brighter blue indicates strong positive correlation.

By analyzing both outputs, you can identify patterns, outliers, and relationships in the data, guiding further analysis or decision-making."""


import random

# Function for Monte Carlo simulation of the stock price
def monte_carlo_simulation(data, forecast_period, num_simulations=1000):
    # Calculate daily log returns
    data['daily_returns'] = np.log(data['y'] / data['y'].shift(1))
    daily_return_mean = data['daily_returns'].mean()
    daily_volatility = data['daily_returns'].std()

    # Monte Carlo simulation
    simulations = np.zeros((num_simulations, forecast_period))
    last_price = data['y'].iloc[-1]

    for i in range(num_simulations):
        price_series = [last_price]
        for j in range(forecast_period):
            # Calculate the next price using mean return and volatility
            price_next = price_series[-1] * np.exp(daily_return_mean + daily_volatility * np.random.normal())
            price_series.append(price_next)
        simulations[i, :] = price_series[1:]  # Ignore the first price (initial)

    return simulations

# Monte Carlo simulation
st.subheader(f"Monte Carlo Simulation for {ticker}")
simulations = monte_carlo_simulation(data, forecast_period)

# Create a plot for the simulations
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(simulations.shape[0]):
    ax.plot(range(forecast_period), simulations[i, :], color='blue', alpha=0.1)  # Plot each simulation

# Show the average simulation
mean_simulation = simulations.mean(axis=0)
ax.plot(range(forecast_period), mean_simulation, color='red', label="Mean Simulation", lw=2)

# Layout the plot
ax.set_title(f"Monte Carlo Simulations of Future Price for {ticker}")
ax.set_xlabel("Days")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Description for interpreting the graph
st.markdown(
    "- Each blue line represents a simulation of the future price.\n"
    "- The red line represents the average of the simulations."
)
