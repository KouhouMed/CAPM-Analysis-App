import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf 
import datetime as dt

# Configuring web application
st.set_page_config(page_title= "Market Insight Analysis", page_icon= ":chart:", layout = 'wide')

# Adding page title to web application
st.title("Capital Asset Pricing Model")

col1, col2 = st.columns([3, 1])
with col1:
    stock_list = st.multiselect("Choose your stocks", (
    'AI.PA', 'AIR.PA', 'ALO.PA', 'MT.AS', 'ATO.PA', 'CS.PA', 'BNP.PA', 'CAP.PA', 
    'CA.PA', 'ACA.PA', 'DSY.PA', 'ENGI.PA', 'EL.PA', 'RMS.PA', 'KER.PA', 'OR.PA', 
    'LR.PA', 'MC.PA', 'ML.PA', 'ORA.PA', 'RI.PA', 'PUB.PA', 'RNO.PA', 'SAF.PA', 
    'SAN.PA', 'SU.PA', 'GLE.PA', 'STLA.PA', 'STM.PA', 'TEP.PA', 'HO.PA', 'TTE.PA', 
    'URW.AS', 'VIE.PA', 'DG.PA', 'VIV.PA'), 
	['MC.PA', 'OR.PA', 'TTE.PA'])
    
with col2:
    years = st.number_input("Number of years", 1, 10)

start_date = dt.date(dt.date.today().year - years, dt.date.today().month, dt.date.today().day)
end_date = dt.date.today()

try:
    # Fetching CAC 40 data from Yahoo Finance and storing as DataFrame
    cac40_data = yf.download('^FCHI', start=start_date, end=end_date)

    # Reset index to make 'Date' an actual column instead of an index column
    cac40_data.reset_index(inplace=True)

    # Get 'Date' and 'Close' columns only
    data1 = cac40_data[['Date', 'Close']]
    cac40 = data1.copy()

    cac40.rename(columns={'Close': 'CAC40'}, inplace=True)

    # Retrieving close price data for selected stocks and storing as DataFrame
    stock_df = pd.DataFrame()
    for stock in stock_list:
        stocks_data = yf.download(stock, period=f'{years}y')
        stock_df[f'{stock}'] = stocks_data['Close']
    
    stock_df.reset_index(inplace=True)

    stocks_df = pd.merge(stock_df, cac40, on='Date', how='inner')

    # Displaying stocks data on web application 
    col3, col4 = st.columns([1,1])
    with col3:
        st.markdown("### Stocks Data Head")
        st.dataframe(stocks_df.head(), use_container_width=True)
    with col4:
        st.markdown("### Stocks Data Tail")
        st.dataframe(stocks_df.tail(), use_container_width=True)
    
    # Plotly line charts for selected stocks
    def plot(df):
        fig1 = px.line()

        for i in df.columns[1:]:
            fig1.add_scatter(x=df['Date'], y=df[i], name=i)
        fig1.update_layout(width=700, margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right',x=1))

        return fig1
    
    # Min-max normalization of stock values
    def normalize(df2):
        df = df2.copy()
        for i in df.columns[1:]:
            min_val = df[i].min()
            max_val = df[i].max()
            df[i] = (df[i] - min_val) / (max_val - min_val)

        return df
    

    # Line charts for selected stocks on web application
    col5, col6 = st.columns([1,1])
    with col5:
        st.markdown('### Stock Price Trends over Time')
        st.plotly_chart(plot(stocks_df), use_container_width=True)
    with col6:
        st.markdown('### Normalized Stock Price Trends over Time')
        st.plotly_chart(plot(normalize(stocks_df)), use_container_width=True)
            
    
except:
    st.write("Something went wrong")