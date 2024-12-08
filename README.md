# Stock-Insights-Prediction-Analysis
A powerful Streamlit-based application for stock price prediction, technical analysis, and comparisons, utilizing Prophet, ARIMA, and ARCH models alongside classic indicators.

## Overview
This project provides an interactive web application for stock price prediction and analysis. It uses advanced forecasting models such as **Prophet**, **ARIMA**, and **ARCH** along with classic technical indicators. The application allows users to forecast stock prices, view historical data, and compare the performance of different stocks over a customizable time horizon.

### Key Features:
- **Stock Price Prediction**: Forecast future stock prices using advanced time series models.
- **Technical Indicators**: View key stock indicators like Moving Averages, Bollinger Bands, and more.
- **Multiple Models**: Use **Prophet**, **ARIMA**, and **ARCH** models for different types of predictions.
- **Stock Comparison**: Compare the historical performance and predictions of multiple stocks.
- **Data Visualization**: Interactive charts and graphs to visualize stock price trends, predictions, and technical indicators.

### Usage
Once the application is running, follow these steps to use it:
- **Enter Stock Ticker**: In the sidebar, input the stock ticker symbol (e.g., AAPL for Apple Inc., GOOGL for Alphabet Inc.).
- **Select Prediction Horizon**: Choose how far in the future you want to predict (e.g., 1 month, 6 months, 1 year).
- **Choose Historical Data Range**: Select the range of historical data to use for prediction (e.g., 1 year, 5 years).
- **Add Additional Tickers for Comparison**: You can input multiple stock tickers to compare their performance and predictions. This allows you to analyze different stocks side by side.
- **Analyze Technical Indicators**: View key stock indicators such as Simple Moving Averages (SMA), Exponential Moving Averages (EMA), Bollinger Bands, and others to help assess the stock's performance and trends.
- **View Predictions and Trends**: The app will generate predictions for the selected stock(s) using the chosen forecasting models. You can view both the historical trends and future predictions on an interactive chart.

## Requirements

To run the application, you'll need the following libraries:

- **Python 3.8+**
- **Streamlit**: For creating the interactive web app.
- **pandas**: For data manipulation and analysis.
- **yfinance**: For retrieving stock data from Yahoo Finance.
- **prophet**: For time series forecasting using the Prophet model.
- **matplotlib**: For data visualization (e.g., plotting stock price trends).
- **statsmodels**: For statistical modeling (ARIMA).
- **arch**: For time series modeling with ARCH and GARCH models.
