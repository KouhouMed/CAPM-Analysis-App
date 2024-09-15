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
    
    # Calculate daily return of each asset
    def stocks_daily_return(df):
        df_daily_return = df.copy()
		    
        for i in df.columns[1:]:
            df_daily_return[i] = ((df[i] - df[i].shift(1)) / df[i].shift(1)) * 100
        
        return df_daily_return
    
    # Calculate Daily Return for stocks and market over time
    daily_return = stocks_daily_return(stocks_df).dropna()

    # Calculate Market Return
    market_return = daily_return['CAC40']

    # Calculate Covariance for each stock
    cov_matrix = daily_return.iloc[:,1:].cov() # Covariance matrix

    # Calculate Variance for market
    market_variance = np.var(market_return)

    # Calculating Beta values for each stock
    beta_value = {} 
    for column in daily_return.iloc[:,1:].columns:
        covariance = cov_matrix[column]['CAC40']
        beta = covariance / market_variance
        beta_value[column] = beta
    
    # Calculating market portfolio return (rm)
    market_portfolio_return = market_return.mean() * 252

    # Considering Risk-Free Rate of Return (rf) as 0
    risk_free_rate = 0.068

    # Calculate Expected Return (ri) using CAPM
    expected_return = {}
    for stock, beta in beta_value.items():
        expected_returns = risk_free_rate + beta * (market_portfolio_return - risk_free_rate)
        expected_return[stock] = expected_returns
    
    # Converting dict to dataframes
    beta_df = pd.DataFrame.from_dict(beta_value, orient='index', columns=['Beta Value'])
    beta_df.reset_index(inplace=True)
    beta_df.rename(columns={'index': 'Stocks'}, inplace=True)

    expected_return_df = pd.DataFrame.from_dict(expected_return, orient='index', columns=['Expected Return (in %)'])
    expected_return_df.reset_index(inplace=True)
    expected_return_df.rename(columns={'index': 'Stocks'}, inplace=True)

    col7, col8 = st.columns([1, 1])
    with col7:
        data2 = daily_return.iloc[:, :-1]
        st.markdown("### Daily Returns for Different Stocks")
        st.dataframe(data2.sort_values(by='Date', ascending=False).head(), use_container_width = True)
    with col8:
        st.markdown("### Market Returns for CAC40")
        st.dataframe(daily_return[['Date', 'CAC40']].sort_values(by='Date', ascending=False).head(), use_container_width=True)
    
    # Function to plot bar chart for daily returns for different stocks
    def plot2(df, col):
        color ='#0072BB'
        fig2 = px.bar(df, x=df['Date'], y=df[col], color_discrete_sequence=[color])
        fig2.update_layout(width=700, margin=dict(l=20, r=20, t=50, b=20), xaxis_title='', yaxis_title='')
        return fig2

    # Function to plot bar chart for market return for CAC40
    def plot3(df):
        color = '#004B8D'
        fig3 = px.bar(df, x=df['Date'], y=df['CAC40'], color_discrete_sequence=[color])
        fig3.update_layout(height=500, width=700, margin=dict(l=20, r=20, t=50, b=20), xaxis_title='', yaxis_title='')
        return fig3	
    
    col9, col10 = st.columns([1, 1])
    with col9:
        st.markdown("### Daily Return Trends")
        stock_lst = st.selectbox("Choose a stock", (stock_list))
        st.plotly_chart(plot2(daily_return.iloc[:, :-1], stock_lst), use_container_width=True)
    with col10:
        st.markdown("### Market Return Trends")
        st.plotly_chart(plot3(daily_return), use_container_width=True)
    
    col11, col12 = st.columns([1, 1])
    with col11:
        st.markdown("### Calculated Beta Value")
        st.dataframe(beta_df, use_container_width = True)
    with col12:
        st.markdown("### Calculated Expected Returns using CAPM")
        st.dataframe(expected_return_df, use_container_width=True)
    
    # Function to plot Bar chart for beta values of different stocks
    def plot4(df):
        fig4 = px.bar(x=df['Stocks'], y=df['Beta Value'])
        fig4.update_layout(width=700, xaxis_title='', yaxis_title='')

        return fig4

    # Function to plot Bar chart for expected returns of different stocks
    def plot5(df):
        color = '#73e6f5'  # A shade of blue
        fig5 = px.bar(x=df['Stocks'], y=df['Expected Return (in %)'], color_discrete_sequence=[color])
        fig5.update_layout(width=700, xaxis_title='', yaxis_title='')

        return fig5

    col13, col14 = st.columns([1, 1])
    with col13:
        st.markdown("### Beta Values for Different Stocks and Market")
        st.plotly_chart(plot4(beta_df), use_container_width=True)
    with col14:
        st.markdown("### Expected Returns for Different Stocks and Market")
        st.plotly_chart(plot5(expected_return_df), use_container_width=True)
    
    # Calculating Expected Return for the portfolio
    n = len(stocks_df.columns) - 1
    portfolio_weights = 1/n * np.ones(n)
    er_portfolio = sum(np.array(list(expected_return.values())) * portfolio_weights)

    # Calculate Sharpe Ratio
    def calculate_sharpe_ratio(returns, risk_free_rate):
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized

    # Calculate Sharpe Ratio for each stock
    sharpe_ratios = {}
    for stock in stock_list:
        stock_returns = daily_return[stock].dropna()
        sharpe_ratios[stock] = calculate_sharpe_ratio(stock_returns, risk_free_rate / 252)  # Daily risk-free rate

    # Convert Sharpe Ratios to DataFrame
    sharpe_df = pd.DataFrame.from_dict(sharpe_ratios, orient='index', columns=['Sharpe Ratio'])
    sharpe_df.reset_index(inplace=True)
    sharpe_df.rename(columns={'index': 'Stocks'}, inplace=True)

    # Display Sharpe Ratios
    st.subheader("Sharpe Ratios")
    st.dataframe(sharpe_df, use_container_width=True)

    # Visualize Sharpe Ratios
    def plot_sharpe_ratios(df):
        fig = px.bar(df, x='Stocks', y='Sharpe Ratio', title='Sharpe Ratios for Selected Stocks')
        fig.update_layout(xaxis_title='Stocks', yaxis_title='Sharpe Ratio')
        return fig

    st.plotly_chart(plot_sharpe_ratios(sharpe_df), use_container_width=True)

    # Calculate portfolio Sharpe Ratio
    portfolio_returns = daily_return[stock_list].mean(axis=1)
    portfolio_sharpe_ratio = calculate_sharpe_ratio(portfolio_returns, risk_free_rate / 252)

    st.subheader(f"Portfolio Sharpe Ratio: {portfolio_sharpe_ratio:.2f}")

    # Calculate correlation matrix
    correlation_matrix = daily_return[stock_list].corr()

    # Create a heatmap using seaborn
    def plot_correlation_heatmap(corr_matrix):
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Correlation Heatmap of Selected Stocks')
        return plt

    st.subheader(f'Conclusion: The Expected Return Based on CAPM for the portfolio is roughly {round(er_portfolio, 2)}%')

except:
    st.write("Something went wrong")