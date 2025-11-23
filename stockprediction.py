import streamlit as st
from datetime import date
import pandas as pd

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "1970-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME', 'TSLA', 'NVDA')
selected_stock = st.selectbox('Select ticker', stocks)

years_of_prediction = st.slider('Years of prediction:', 1, 10)
period = years_of_prediction * 365


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Opening Price", mode='lines'))
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Closing Price", mode='lines'))
fig.update_xaxes(title_text="Date", rangeslider_visible=True, rangeselector=dict(
    buttons=list([
        dict(count=1, label="1d", step="day"),
        dict(count=7, label="1w", step="day"),
        dict(count=1, label="1m", step="month"),
        dict(count=3, label="3m", step="month"),
        dict(count=6, label="6m", step="month"),
        dict(count=1, label="1y", step="year"),
        dict(step="all")
    ])
))
fig.update_yaxes(title_text="Price")
fig.update_layout(title_text='Historical Prices', height=500)
st.plotly_chart(fig, use_container_width=True)

# Forecasting with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

model = Prophet()
model.fit(df_train)
future_prices = model.make_future_dataframe(periods=period)
forecast = model.predict(future_prices)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecasting for {years_of_prediction} years')
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = model.plot_components(forecast)
st.write(fig2)
