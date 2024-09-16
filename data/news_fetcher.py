import requests
import os

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

def fetch_stock_news(symbol):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    if "feed" not in data:
        return []
    
    news_items = data["feed"][:5]  # Get the top 5 news items
    
    formatted_news = []
    for item in news_items:
        formatted_news.append({
            "title": item["title"],
            "url": item["url"],
            "time_published": item["time_published"],
            "summary": item["summary"]
        })
    
    return formatted_news