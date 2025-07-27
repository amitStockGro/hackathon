import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import logging
import re
from datetime import datetime
from collections import Counter

# Configure logging
logger = logging.getLogger(__name__)

# Sentiment ranges for categorization
SENTIMENT_RANGES = {
    "Very Positive": (0.6, 1.0),
    "Positive": (0.2, 0.6),
    "Neutral": (-0.1, 0.2),
    "Negative": (-0.4, -0.1),
    "Very Negative": (-1.0, -0.4)
}

SENTIMENT_COLORS = {
    "Very Positive": "#00CC44",
    "Positive": "#66DD66",
    "Neutral": "#FFAA00",
    "Negative": "#FF6666",
    "Very Negative": "#CC0000"
}


def categorize_sentiment(score):
    """Categorize sentiment score into ranges"""
    for category, (min_val, max_val) in SENTIMENT_RANGES.items():
        if min_val <= score < max_val:
            return category
    return "Neutral"


@st.cache_data(ttl=300)
def fetch_news_data():
    """Fetch news data from API with error handling"""
    try:
        response = requests.get("https://stage.stockgro.com/hackathon/news", timeout=30)
        response.raise_for_status()

        api_data = response.json()
        if api_data.get("success", False):
            news_items = api_data.get("data", [])

            # Process news data
            processed_news = []
            for item in news_items:
                try:
                    sentiment_category = categorize_sentiment(item.get('sentiment_score', 0))
                    processed_item = {
                        **item,
                        'sentiment_category': sentiment_category,
                        'datetime': datetime.strptime(item['date'], '%Y-%m-%d %H:%M:%S.%f').strftime('%Y-%m-%d %H:%M')
                    }
                    processed_news.append(processed_item)
                except Exception as e:
                    logger.warning(f"Error processing news item: {e}")
                    continue

            return processed_news
        else:
            st.error(f"News API Error: {api_data.get('message', 'Unknown error')}")
            return None

    except requests.exceptions.Timeout:
        st.error("News API request timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the news API server.")
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        logger.error(f"News API error: {str(e)}")

    return None


def render_news_sentiment_analysis():
    """Render news sentiment analysis charts"""
    if not st.session_state.news_data:
        return

    st.subheader("📰 Market Sentiment Analysis")

    try:
        sentiment_counts = Counter([item['sentiment_category'] for item in st.session_state.news_data])
        sector_sentiment = {}

        for item in st.session_state.news_data:
            sector = item['sector']
            sentiment = item['sentiment_category']
            if sector not in sector_sentiment:
                sector_sentiment[sector] = Counter()
            sector_sentiment[sector][sentiment] += 1

        col1, col2 = st.columns(2)

        with col1:
            sentiment_df = pd.DataFrame([
                {'Sentiment': sentiment, 'Count': count}
                for sentiment, count in sentiment_counts.items()
            ])

            fig1 = px.pie(
                sentiment_df,
                values='Count',
                names='Sentiment',
                title="Overall Market Sentiment Distribution",
                color='Sentiment',
                color_discrete_map=SENTIMENT_COLORS,
                hole=0.4
            )
            fig1.update_traces(textposition='inside', textinfo='percent+label')
            fig1.update_layout(showlegend=True, height=400)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            sector_counts = Counter([item['sector'] for item in st.session_state.news_data])
            top_sectors = dict(sector_counts.most_common(10))

            fig2 = px.bar(
                x=list(top_sectors.keys()),
                y=list(top_sectors.values()),
                title="News Volume by Sector (Top 10)",
                labels={'x': 'Sector', 'y': 'Number of News Items'},
                color=list(top_sectors.values()),
                color_continuous_scale='viridis'
            )
            fig2.update_layout(xaxis_tickangle=-45, height=400, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            stocks_in_range = Counter([item['sentiment_category'] for item in st.session_state.news_data])

            fig3 = px.bar(
                x=list(stocks_in_range.keys()),
                y=list(stocks_in_range.values()),
                title="Number of Stocks by Sentiment Range",
                labels={'x': 'Sentiment Range', 'y': 'Number of Stocks'},
                color=list(stocks_in_range.keys()),
                color_discrete_map=SENTIMENT_COLORS
            )
            fig3.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            sectors_by_sentiment = {}
            for item in st.session_state.news_data:
                sentiment = item['sentiment_category']
                sector = item['sector']
                if sentiment not in sectors_by_sentiment:
                    sectors_by_sentiment[sentiment] = set()
                sectors_by_sentiment[sentiment].add(sector)

            sectors_count = {k: len(v) for k, v in sectors_by_sentiment.items()}

            fig4 = px.bar(
                x=list(sectors_count.keys()),
                y=list(sectors_count.values()),
                title="Number of Sectors by Sentiment Range",
                labels={'x': 'Sentiment Range', 'y': 'Number of Sectors'},
                color=list(sectors_count.keys()),
                color_discrete_map=SENTIMENT_COLORS
            )
            fig4.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)

    except Exception as e:
        st.error(f"Error rendering sentiment analysis: {str(e)}")


def render_news_feed():
    """Render categorized news feed"""
    if not st.session_state.news_data:
        return

    st.subheader("📊 Latest Market News")

    try:
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            sentiment_filter = st.selectbox(
                "Filter by Sentiment:",
                options=['All'] + list(SENTIMENT_RANGES.keys()),
                index=0
            )

        with col2:
            sectors_in_news = sorted(list(set([item['sector'] for item in st.session_state.news_data])))
            selected_sectors_filter = st.multiselect(
                "Filter by Sectors:",
                options=sectors_in_news,
                default=[]
            )

        with col3:
            sort_by = st.selectbox(
                "Sort by:",
                options=['Latest', 'Sentiment Score'],
                index=0
            )

        # Filter news data
        filtered_news = st.session_state.news_data.copy()

        if sentiment_filter != 'All':
            filtered_news = [item for item in filtered_news if item['sentiment_category'] == sentiment_filter]

        if selected_sectors_filter:
            filtered_news = [item for item in filtered_news if item['sector'] in selected_sectors_filter]

        # Sort news data
        if sort_by == 'Latest':
            filtered_news = sorted(filtered_news, key=lambda x: x['date'], reverse=True)
        else:
            filtered_news = sorted(filtered_news, key=lambda x: x['sentiment_score'], reverse=True)

        st.markdown(f"**Showing {len(filtered_news)} news items**")

        # Display news items
        for item in filtered_news[:20]:
            color = SENTIMENT_COLORS.get(item['sentiment_category'], "#666666")

            with st.container():
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"""
                    <div style="border-left: 4px solid {color}; padding-left: 10px; margin-bottom: 10px;">
                        <h4 style="margin: 0; color: #333;">{item['heading']}</h4>
                        <p style="margin: 5px 0; color: #666; font-size: 14px;">
                            <strong>{item['symbol']}</strong> | {item['sector']} | {item['datetime']}
                        </p>
                        <p style="margin: 5px 0; font-style: italic; color: #555;">
                            {item['caption']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.metric(
                        label="Sentiment",
                        value=item['sentiment_category'],
                        delta=f"Score: {item['sentiment_score']:.3f}"
                    )

                with st.expander(f"Read more about {item['symbol']}"):
                    clean_details = re.sub('<.*?>', '', item['details'])
                    st.write(clean_details)

            st.divider()

    except Exception as e:
        st.error(f"Error rendering news feed: {str(e)}")


def render_sidebar_news_sentiment():
    """Render news sentiment section for sidebar"""
    if st.session_state.news_data:
        with st.expander("📰 News Sentiment", expanded=True):
            sentiment_counts = Counter([item['sentiment_category'] for item in st.session_state.news_data])
            total_news = len(st.session_state.news_data)

            # Sentiment meter
            st.metric("Total News Items", total_news)

            for sentiment, color in SENTIMENT_COLORS.items():
                count = sentiment_counts.get(sentiment, 0)
                if count > 0:
                    percentage = (count / total_news) * 100
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                        <div style="width: 12px; height: 12px; background-color: {color}; 
                                    border-radius: 3px; margin-right: 8px;"></div>
                        <span style="flex-grow: 1;">{sentiment}</span>
                        <span>{count} ({percentage:.1f}%)</span>
                    </div>
                    """, unsafe_allow_html=True)


def render_news_sentiment_tab():
    """Render the complete news and sentiment tab"""
    # News tab action buttons
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔄 Refresh News", key="refresh_news",
                     help="Load latest market news and sentiment data", use_container_width=True):
            with st.spinner("Loading latest news..."):
                st.session_state.news_data = fetch_news_data()
                if st.session_state.news_data:
                    st.success(f"Loaded {len(st.session_state.news_data)} news items!")
                    st.rerun()

    if st.session_state.show_news:
        if st.session_state.news_data:
            render_news_sentiment_analysis()
            st.markdown("---")
            render_news_feed()
        else:
            st.info("📰 Click 'Refresh News' above to load market news and sentiment analysis.")
            if st.button("🔄 Load News Now", use_container_width=True):
                with st.spinner("Loading latest news..."):
                    st.session_state.news_data = fetch_news_data()
                    if st.session_state.news_data:
                        st.success(f"Loaded {len(st.session_state.news_data)} news items!")
                        st.rerun()
    else:
        st.info("📰 Enable 'Show News Analysis' in the sidebar to view market sentiment.")