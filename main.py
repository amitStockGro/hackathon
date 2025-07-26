import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import requests
import logging
from collections import Counter
from stock_explorer import fetch_stock_data
from trade_view import fetch_trade_views, render_trade_views_table, render_trade_views_summary, render_sector_analysis, prepare_trade_views_dataframe

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Stock Investment Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Initialize session state with defaults
def initialize_session_state():
    defaults = {
        'api_response': None,
        'news_data': None,
        'show_results': False,
        'edited_df': None,
        'selected_sectors': ["Power", "Infrastructure"],
        'capital': 100000,
        'risk_index': 0,
        'horizon_index': 1,
        'last_edit_time': None,
        'show_news': True,
        'selected_sentiment_filter': 'All'
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# Initialize session state at the start
initialize_session_state()

# Constants
SECTORS = ["Power", "Infrastructure", "FMCG", "Telecom", "Non - Ferrous Metals", "Healthcare", "Diamond  &  Jewellery",
           "Iron & Steel", "Finance", "IT", "Textile", "Hospitality", "Inds. Gases & Fuels", "Paper",
           "Business Services", "Automobile & Ancillaries", "Ship Building", "Aviation", "Diversified", "Chemicals",
           "Consumer Durables", "Plastic Products", "Alcohol", "Capital Goods", "Logistics", "Realty", "Abrasives",
           "Mining", "Electricals", "Media & Entertainment", "Bank", "Trading", "Gas Transmission", "Retailing",
           "Ratings", "Insurance", "Ferro Manganese", "Construction Materials", "Agri", "Crude Oil", "Miscellaneous"]
RISK_PROFILES = ["Conservative", "Moderate", "Aggressive"]
TIME_HORIZONS = ["Short-term: 1–4 weeks", "Medium-term: 1–3 months", "Long-term: 6–12 months"]
TIMELINE_MAPPING = {
    "Short-term: 1–4 weeks": "short_term",
    "Medium-term: 1–3 months": "medium_term",
    "Long-term: 6–12 months": "long_term"
}

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


@st.cache_data(ttl=300)
def convert_timeline(ui_timeline):
    """Convert UI timeline to API format"""
    return TIMELINE_MAPPING.get(ui_timeline, "medium_term")


def validate_inputs(amount, sectors):
    """Validate user inputs"""
    errors = []
    if amount < 1000:
        errors.append("Investment amount must be at least ₹1,000")
    if not sectors:
        errors.append("Please select at least one sector")
    return errors


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


def generate_api_response(amount, risk, sectors, timeline):
    """Make API call with proper error handling and timeout"""
    api_url = "https://stage.stockgro.com/hackathon/recommendations"
    headers = {"Content-Type": "application/json"}

    payload = {
        "amount": amount,
        "risk": risk,
        "sectors": sectors,
        "timeline": convert_timeline(timeline)
    }

    try:
        response = requests.get(
            api_url,
            headers=headers,
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        api_data = response.json()
        if api_data.get("success", False):
            return {
                "status": "success",
                "recommendations": api_data["data"]["stocks"],
                "total_amount": amount,
                "risk_profile": risk,
                "timeline": timeline,
                "timestamp": datetime.now().isoformat()
            }
        else:
            error_msg = api_data.get('message', 'Unknown API error')
            st.error(f"API Error: {error_msg}")
            logger.error(f"API returned error: {error_msg}")
            return None

    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API server. Please check if the server is running.")
    except Exception as e:
        st.error(f"API request failed: {str(e)}")

    return None


@st.cache_data
def prepare_dataframe(response_data):
    """Prepare dataframe from API response with error handling"""
    try:
        portfolio_data = []
        recommendations = response_data['recommendations']
        total_amount = response_data['total_amount']

        for sector, stocks in recommendations.items():
            for stock in stocks:
                required_fields = ['symbol', 'allocation', 'price', 'risk', 'target', 'expected_return',
                                   'expected_profit']
                if not all(field in stock for field in required_fields):
                    continue

                allocation_amount = total_amount * (stock['allocation'] / 100)

                portfolio_data.append({
                    "Sector": sector,
                    "Stock": stock['symbol'],
                    "Shares": stock['shares'],
                    "Allocation (%)": round(stock['allocation'], 2),
                    "Price (₹)": round(stock['price'], 2),
                    "Risk": stock['risk'],
                    "Target (₹)": round(stock['target'], 2),
                    "Expected Return (%)": round(stock['expected_return'], 2),
                    "Expected Profit (₹)": round(stock['expected_profit'], 2),
                    "Amount (₹)": round(allocation_amount, 2)
                })

        if not portfolio_data:
            st.error("No valid stock data received from API")
            return None

        return pd.DataFrame(portfolio_data)

    except Exception as e:
        st.error(f"Error preparing portfolio data: {str(e)}")
        return None


def update_portfolio_calculations(df):
    """Update portfolio calculations when shares change"""
    try:
        df["Amount (₹)"] = df["Shares"] * df["Price (₹)"]
        df["Expected Profit (₹)"] = (df["Target (₹)"] - df["Price (₹)"]) * df["Shares"]

        total_invested = df["Amount (₹)"].sum()
        if total_invested > 0:
            df["Allocation (%)"] = (df["Amount (₹)"] / total_invested) * 100

        return df.round(2)

    except Exception as e:
        st.error(f"Error updating calculations: {str(e)}")
        return df


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
                    import re
                    clean_details = re.sub('<.*?>', '', item['details'])
                    st.write(clean_details)

            st.divider()

    except Exception as e:
        st.error(f"Error rendering news feed: {str(e)}")


def render_input_form():
    """Render the investment preferences form"""
    with st.expander("💰 Investment Preferences", expanded=True):
        with st.form("investment_preferences", clear_on_submit=False):
            col1, col2 = st.columns(2)

            with col1:
                selected_sectors = st.multiselect(
                    "Select stock sectors (at least one required):",
                    options=SECTORS,
                    default=st.session_state.selected_sectors,
                    help="Choose sectors you want to invest in"
                )

                capital = st.number_input(
                    "Enter your investment capital (INR):",
                    min_value=1000,
                    max_value=10000000,
                    step=1000,
                    value=st.session_state.capital,
                    help="Minimum investment: ₹1,000"
                )

            with col2:
                risk_profile = st.radio(
                    "Select your risk preference:",
                    options=RISK_PROFILES,
                    index=st.session_state.risk_index,
                    help="Choose based on your risk tolerance"
                )

                time_horizon = st.radio(
                    "Select your investment time horizon:",
                    options=TIME_HORIZONS,
                    index=st.session_state.horizon_index,
                    help="Choose your investment duration"
                )

            submitted = st.form_submit_button("🚀 Get Recommendations", use_container_width=True)

            if submitted:
                errors = validate_inputs(capital, selected_sectors)
                if errors:
                    for error in errors:
                        st.error(error)
                    return

                st.session_state.selected_sectors = selected_sectors
                st.session_state.capital = capital
                st.session_state.risk_index = RISK_PROFILES.index(risk_profile)
                st.session_state.horizon_index = TIME_HORIZONS.index(time_horizon)

                with st.spinner('🔄 Fetching personalized recommendations...'):
                    response = generate_api_response(
                        amount=capital,
                        risk=risk_profile,
                        sectors=selected_sectors,
                        timeline=time_horizon
                    )

                    if response:
                        df = prepare_dataframe(response)
                        if df is not None:
                            st.session_state.api_response = response
                            st.session_state.edited_df = df
                            st.success("✅ Recommendations generated successfully!")
                            st.rerun()


def render_portfolio_results():
    """Render the portfolio recommendations and editor"""
    if st.session_state.api_response is None or st.session_state.edited_df is None:
        return

    st.markdown("---")
    st.subheader("🎯 Your Personalized Portfolio Recommendations")

    with st.expander("📋 Current Investment Summary", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        response = st.session_state.api_response
        with col1:
            st.metric("💰 Investment Capital", f"₹{response['total_amount']:,.0f}")
        with col2:
            st.metric("⚖️ Risk Profile", response['risk_profile'])
        with col3:
            st.metric("⏱️ Time Horizon", response['timeline'])
        with col4:
            total_stocks = len(st.session_state.edited_df)
            st.metric("📊 Total Stocks", f"{total_stocks}")

        sectors_list = ', '.join(response['recommendations'].keys())
        st.info(f"**Selected Sectors:** {sectors_list}")

    with st.expander("✏️ Customize Your Portfolio", expanded=True):
        st.markdown("**Instructions:** Adjust the number of shares for each stock to customize your allocation.")

        column_config = {
            "Sector": st.column_config.TextColumn("Sector", disabled=True, width="medium"),
            "Stock": st.column_config.TextColumn("Stock Symbol", disabled=True, width="medium"),
            "Shares": st.column_config.NumberColumn(
                "Shares",
                help="Adjust the number of shares to buy",
                min_value=0,
                step=1,
                width="small"
            ),
            "Allocation (%)": st.column_config.NumberColumn(
                "Allocation %",
                format="%.2f%%",
                disabled=True,
                width="small"
            ),
            "Price (₹)": st.column_config.NumberColumn(
                "Price",
                format="₹%.2f",
                disabled=True,
                width="small"
            ),
            "Risk": st.column_config.TextColumn("Risk Level", disabled=True, width="small"),
            "Target (₹)": st.column_config.NumberColumn(
                "Target",
                format="₹%.2f",
                disabled=True,
                width="small"
            ),
            "Expected Return (%)": st.column_config.NumberColumn(
                "Expected Return",
                format="%.2f%%",
                disabled=True,
                width="small"
            ),
            "Expected Profit (₹)": st.column_config.NumberColumn(
                "Expected Profit",
                format="₹%.2f",
                disabled=True,
                width="medium"
            ),
            "Amount (₹)": st.column_config.NumberColumn(
                "Investment Amount",
                format="₹%.2f",
                disabled=True,
                width="medium"
            )
        }

        edited_df = st.data_editor(
            st.session_state.edited_df,
            column_config=column_config,
            hide_index=True,
            use_container_width=True,
            key="portfolio_editor"
        )

        if not edited_df.equals(st.session_state.edited_df):
            st.session_state.edited_df = update_portfolio_calculations(edited_df)
            st.session_state.last_edit_time = datetime.now()
            st.rerun()

        col1, col2, col3 = st.columns(3)

        total_invested = st.session_state.edited_df["Amount (₹)"].sum()
        total_expected_profit = st.session_state.edited_df["Expected Profit (₹)"].sum()
        avg_expected_return = st.session_state.edited_df["Expected Return (%)"].mean()

        with col1:
            st.metric("💵 Total Invested", f"₹{total_invested:,.2f}")
        with col2:
            st.metric("💰 Expected Profit", f"₹{total_expected_profit:,.2f}")
        with col3:
            st.metric("📈 Avg Expected Return", f"{avg_expected_return:.2f}%")

        remaining_cash = st.session_state.api_response['total_amount'] - total_invested
        if remaining_cash > 0:
            st.info(f"💵 **Remaining Cash:** ₹{remaining_cash:,.2f}")
        elif remaining_cash < 0:
            st.warning(f"⚠️ **Over-invested by:** ₹{abs(remaining_cash):,.2f}")


def render_visualizations():
    """Render portfolio visualizations"""
    if st.session_state.edited_df is None:
        return

    st.subheader("📊 Portfolio Visualization")

    col1, col2 = st.columns(2)

    with col1:
        sector_data = st.session_state.edited_df.groupby('Sector')['Amount (₹)'].sum().reset_index()
        fig1 = px.pie(
            sector_data,
            values='Amount (₹)',
            names='Sector',
            title="Portfolio Allocation by Sector",
            hole=0.4
        )
        fig1.update_traces(textposition='inside', textinfo='percent+label')
        fig1.update_layout(showlegend=True, height=400)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        top_stocks = st.session_state.edited_df.nlargest(10, 'Amount (₹)')
        fig2 = px.bar(
            top_stocks,
            x='Stock',
            y='Amount (₹)',
            color='Sector',
            title="Top 10 Stock Allocations",
            text='Amount (₹)'
        )
        fig2.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
        fig2.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig2, use_container_width=True)


def render_sidebar():
    """Render sidebar with improved UI layout and organization"""
    with st.sidebar:
        try:
            st.image("image.png", width=70)
        except:
            st.markdown("📈")

        # Main Actions Section
        with st.expander("🔧 Actions", expanded=True):
            # Refresh buttons in columns
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 News", help="Refresh market news", use_container_width=True):
                    with st.spinner("Loading latest news..."):
                        st.session_state.news_data = fetch_news_data()
                        if st.session_state.news_data:
                            st.success(f"Loaded {len(st.session_state.news_data)} news items!")
                            st.rerun()

            with col2:
                if st.session_state.api_response is not None:
                    if st.button("🔄 Portfolio", help="Reset portfolio to original", use_container_width=True):
                        st.session_state.edited_df = prepare_dataframe(st.session_state.api_response)
                        st.success("Portfolio reset!")
                        st.rerun()

            # Download button
            if st.session_state.edited_df is not None:
                csv = st.session_state.edited_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Portfolio",
                    data=csv,
                    file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime='text/csv',
                    use_container_width=True,
                    help="Download your portfolio as CSV"
                )

        # Portfolio Summary Section
        if st.session_state.edited_df is not None:
            with st.expander("📊 Portfolio Summary", expanded=True):
                total_stocks = len(st.session_state.edited_df)
                total_invested = st.session_state.edited_df["Amount (₹)"].sum()

                st.metric("Total Stocks", total_stocks)
                st.metric("Total Investment", f"₹{total_invested:,.0f}")

                if st.session_state.last_edit_time:
                    st.caption(f"Last updated: {st.session_state.last_edit_time.strftime('%H:%M:%S')}")

        # News Sentiment Section
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

        # About Section
        with st.expander("ℹ️ About This App", expanded=False):
            st.markdown("""
            **Smart Stock Investment Advisor** helps you:
            - Get personalized recommendations
            - Analyze market sentiment
            - Track live stock data
            - Optimize your portfolio

            **Key Features:**
            - ✨ AI-powered analysis
            - 📈 Real-time data
            - 💰 Portfolio optimization
            - 📰 News sentiment tracking

            Made with ❤️ by Team 777
            """)

            st.markdown("---")
            st.caption("v1.0 | Last updated: July 2023")

def safe_convert_numeric(value, default=0.0):
    """Safely convert value to numeric with fallback"""
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def safe_convert_int(value, default=0):
    """Safely convert value to integer with fallback"""
    try:
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def safe_timestamp_convert(timestamp):
    """Safely convert timestamp to datetime"""
    try:
        return datetime.fromtimestamp(int(timestamp)) if timestamp else None
    except (ValueError, TypeError, OSError):
        return None


def prepare_stock_dataframe(stock_data):
    """Prepare optimized dataframe from stock API response"""
    if not stock_data:
        return None

    try:
        logger.info(f"Processing {len(stock_data)} stocks into DataFrame")
        processed_data = []

        for stock in stock_data:
            # Safely extract nested dictionaries
            week52 = stock.get("week52", {})
            today = stock.get("today", {})
            delta = stock.get("delta", {})

            processed_stock = {
                "Symbol": stock.get("symbol", "N/A"),
                "Sector": stock.get("secor", "Unknown").title(),  # Fixed typo and normalize case
                "Last Trade Price": safe_convert_numeric(stock.get("last_trade_price")),
                "Market Cap": stock.get("market_cap", "Unknown").title(),
                "Volume": safe_convert_int(stock.get("volume", 0)),
                "52 Week High": safe_convert_numeric(week52.get("high")),
                "52 Week Low": safe_convert_numeric(week52.get("low")),
                "Today's Open": safe_convert_numeric(today.get("open")),
                "Today's High": safe_convert_numeric(today.get("high")),
                "Today's Low": safe_convert_numeric(today.get("low")),
                "Delta Price": safe_convert_numeric(delta.get("price", 0)),
                "Delta Percentage": safe_convert_numeric(delta.get("percentage", 0)),
                "Delta Volume": safe_convert_int(delta.get("volume", 0)),
                "Analyst Recommendation": stock.get("analyst_recommendation", "N/A").title(),
                "Sentiment Score": stock.get("score", "N/A").title(),
                "Score Count": safe_convert_int(stock.get("score_count", 0)),
                "Last Trade Time": safe_timestamp_convert(stock.get("last_trade_time"))
            }
            processed_data.append(processed_stock)

        df = pd.DataFrame(processed_data)

        # Optimize DataFrame memory usage
        categorical_cols = ["Sector", "Market Cap", "Analyst Recommendation", "Sentiment Score"]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')

        # Ensure numeric columns are properly typed
        numeric_columns = [
            "Last Trade Price", "Volume", "52 Week High", "52 Week Low",
            "Today's High", "Today's Low", "Delta Percentage", "Score Count"
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        logger.info(f"Successfully processed DataFrame with {len(df)} rows and {len(df.columns)} columns")
        return df

    except Exception as e:
        st.error(f"❌ Error processing stock data: {str(e)}")
        logger.error(f"DataFrame processing error: {e}")
        return None


def render_stock_explorer(stock_df):
    """Enhanced stock explorer interface"""
    if stock_df is None or stock_df.empty:
        st.warning("📭 No stock data available. Please try refreshing or check your connection.")
        if st.button("🔄 Retry", type="primary", key="retry_button"):
            fetch_stock_data.clear()
            st.rerun()
        return

    # Header with key metrics
    st.subheader("🔍 Stock Market Explorer")

    # Display summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Stocks", len(stock_df))
    with col2:
        st.metric("🏭 Sectors", stock_df["Sector"].nunique())
    with col3:
        avg_price = stock_df["Last Trade Price"].mean()
        st.metric("💰 Avg Price", f"₹{avg_price:.2f}")
    with col4:
        total_volume = stock_df["Volume"].sum()
        st.metric("📈 Total Volume", f"{total_volume:,.0f}")

    # Enhanced filters
    st.markdown("### 🔧 Filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        sectors = sorted([s for s in stock_df["Sector"].unique() if pd.notna(s)])
        selected_sectors = st.multiselect(
            "📊 Filter by Sector:",
            options=sectors,
            default=[],
            help="Select one or more sectors to filter the data",
            key="sector_filter"
        )

    with col2:
        market_caps = sorted([mc for mc in stock_df["Market Cap"].unique() if pd.notna(mc)])
        selected_market_caps = st.multiselect(
            "💰 Filter by Market Cap:",
            options=market_caps,
            default=[],
            help="Filter stocks by market capitalization",
            key="market_cap_filter"
        )

    with col3:
        sentiments = sorted([s for s in stock_df["Sentiment Score"].unique() if pd.notna(s)])
        selected_sentiments = st.multiselect(
            "😊 Filter by Sentiment:",
            options=sentiments,
            default=[],
            help="Filter by market sentiment analysis",
            key="sentiment_filter"
        )

    # Apply filters
    filtered_df = stock_df.copy()
    if selected_sectors:
        filtered_df = filtered_df[filtered_df["Sector"].isin(selected_sectors)]
    if selected_market_caps:
        filtered_df = filtered_df[filtered_df["Market Cap"].isin(selected_market_caps)]
    if selected_sentiments:
        filtered_df = filtered_df[filtered_df["Sentiment Score"].isin(selected_sentiments)]

    # Show filter results
    if len(filtered_df) != len(stock_df):
        st.info(f"🔍 Showing {len(filtered_df)} stocks (filtered from {len(stock_df)} total)")

    if filtered_df.empty:
        st.warning("🚫 No stocks match your filter criteria. Try adjusting the filters above.")
        return

    # Visualization tabs
    tab1, tab2 = st.tabs(["📊 Data Explorer", "📈 Sector Analysis"])

    with tab1:
        st.markdown("#### Detailed Stock Data")

        # Table controls
        col1, col2, col3 = st.columns(3)
        with col1:
            sort_column = st.selectbox(
                "📊 Sort by:",
                options=filtered_df.columns.tolist(),
                index=2,  # Default to Last Trade Price
                help="Choose column to sort the data table",
                key="sort_column_select"
            )
        with col2:
            sort_ascending = st.checkbox("⬆️ Ascending order", value=False, key="sort_ascending_check")
        with col3:
            show_rows = st.selectbox(
                "📄 Rows to show:",
                options=[10, 25, 50, 100, "All"],
                index=1,
                help="Number of rows to display",
                key="show_rows_select"
            )

        # Sort and display data
        display_df = filtered_df.sort_values(by=sort_column, ascending=sort_ascending)

        if show_rows != "All":
            display_df = display_df.head(show_rows)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=400
        )

        # Download functionality
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"stock_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download the current filtered dataset",
            key="download_csv_button"
        )

    with tab2:
        st.markdown("### 📊 Sector Performance Overview")

        try:
            col1, col2 = st.columns(2)

            with col1:
                if "Last Trade Price" in filtered_df.columns and filtered_df["Last Trade Price"].sum() > 0:
                    sector_price = filtered_df.groupby("Sector")["Last Trade Price"].mean().sort_values(ascending=True)
                    fig = px.bar(
                        x=sector_price.values,
                        y=sector_price.index,
                        orientation="h",
                        title="💰 Average Stock Price by Sector",
                        labels={"x": "Average Price (₹)", "y": "Sector"},
                        template="plotly_white"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("📊 Price data not available for sector analysis")

            with col2:
                if "Volume" in filtered_df.columns and filtered_df["Volume"].sum() > 0:
                    sector_volume = filtered_df.groupby("Sector")["Volume"].sum().sort_values(ascending=False)
                    fig = px.pie(
                        values=sector_volume.values,
                        names=sector_volume.index,
                        title="📈 Trading Volume Distribution by Sector",
                        template="plotly_white"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("📊 Volume data not available for sector analysis")

        except Exception as e:
            st.error(f"❌ Error creating sector analysis: {str(e)}")
            logger.error(f"Sector analysis error: {e}")


def main():
    """Main application function"""
    st.title("📈 Smart Stock Investment Advisor")
    st.markdown("*Get personalized stock recommendations with real-time market sentiment analysis*")

    # Load news data on app start if not already loaded
    if st.session_state.news_data is None and st.session_state.show_news:
        with st.spinner("Loading market news..."):
            st.session_state.news_data = fetch_news_data()

    # Main navigation with all tabs including Trade Views
    tab1, tab2, tab3, tab4 = st.tabs(
        ["💰 Investment Analysis", "📰 News & Sentiment", "🔍 Stock Explorer", "🐂 Trade Views"])

    with tab1:
        render_input_form()
        render_portfolio_results()
        if st.session_state.edited_df is not None:
            render_visualizations()

    with tab2:
        if st.session_state.show_news:
            if st.session_state.news_data:
                render_news_sentiment_analysis()
                st.markdown("---")
                render_news_feed()
            else:
                st.info("📰 Click 'Refresh News' in the sidebar to load market news and sentiment analysis.")
                if st.button("🔄 Load News Now", use_container_width=True):
                    with st.spinner("Loading latest news..."):
                        st.session_state.news_data = fetch_news_data()
                        if st.session_state.news_data:
                            st.success(f"Loaded {len(st.session_state.news_data)} news items!")
                            st.rerun()
        else:
            st.info("📰 Enable 'Show News Analysis' in the sidebar to view market sentiment.")

    with tab3:
        # Add refresh button for stock data
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🔄 Refresh Stock Data", key="refresh_stocks",
                         help="Clear cache and reload fresh stock data"):
                # Clear the specific cache for stock data
                fetch_stock_data.clear()
                st.rerun()

        # Fetch and display stock data
        stock_data = fetch_stock_data()
        stock_df = prepare_stock_dataframe(stock_data) if stock_data else None
        render_stock_explorer(stock_df)

    with tab4:
        # Trade Views tab with refresh functionality
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🔄 Refresh Trade Views", key="refresh_trade_views",
                         help="Clear cache and reload fresh trade views data"):
                # Clear the trade views cache
                fetch_trade_views.clear()
                st.rerun()

        # Render the trade views dashboard
        try:
            # Fetch and process trade views data
            trade_views_data = fetch_trade_views()
            trade_views_df = prepare_trade_views_dataframe(trade_views_data) if trade_views_data else None

            if trade_views_df is None or trade_views_df.empty:
                st.warning("⚠️ No live bullish trades available at the moment.")
                st.info(
                    "This could be due to market conditions or data availability. Please try refreshing in a few minutes.")
            else:
                # Render trade views components
                render_trade_views_summary(trade_views_df)
                st.markdown("---")
                render_sector_analysis(trade_views_df)
                st.markdown("---")
                render_trade_views_table(trade_views_df)

                # Footer with last updated time
                st.markdown("---")
                st.caption(f"Trade views last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        except Exception as e:
            st.error(f"❌ Error loading trade views: {str(e)}")
            logger.error(f"Trade views error in main: {e}")

    # Render sidebar
    render_sidebar()

if __name__ == "__main__":
    main()