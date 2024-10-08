# Capital Asset Pricing Model (CAPM) Analysis App

The app is deployed on Streamlit Community Cloud. You can access it via the following link :

https://capm-analysis-app-j4ekb6bgyrc7vlfvwjret6.streamlit.app/

## Overview

This Streamlit web application implements the Capital Asset Pricing Model (CAPM) for stock market analysis, focusing on French stocks from the CAC 40 index. It provides a user-friendly interface for investors and financial analysts to perform CAPM analysis, offering both numerical results and visual representations to aid in understanding market dynamics and making informed investment decisions.

## Features

- Select multiple stocks from CAC 40 components for analysis
- Choose the time frame for historical data (1-10 years)
- View interactive visualizations of stock price trends and returns
- Calculate and display beta values and expected returns using CAPM
- Analyze an equally weighted portfolio of selected stocks

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- yfinance

## Installation

1. Clone this repository:

```
git clone https://github.com/KouhouMed/CAPM-Analysis-App.git
```

2. Navigate to the project directory:

```
cd CAPM-Analysis-App
```

3. Install the required packages:

```
pip install -r requirements.txt
```

## Usage

To run the app, use the following command in the project directory:

```
streamlit run app.py
```

Then, open a web browser and go to `http://localhost:8501`.

## How It Works

1. Select stocks from the CAC 40 list
2. Choose the number of years for historical data
3. The app fetches stock data using yfinance
4. It calculates daily returns, beta values, and expected returns
5. The results are displayed in interactive charts and tables

## Contributing

Contributions, issues, and feature requests are welcome.


## Acknowledgments

- Data provided by Yahoo Finance
- Inspired by modern portfolio theory and financial analysis techniques