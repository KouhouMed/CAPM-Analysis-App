import streamlit as st

def display_stock_news(news_items):
    if not news_items:
        st.write("No recent news available for this stock.")
        return

    for item in news_items:
        st.markdown(f"### [{item['title']}]({item['url']})")
        st.write(f"Published: {item['time_published']}")
        st.write(item['summary'])
        st.markdown("---")
