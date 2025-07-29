import streamlit as st
import pandas as pd
import logging
from datetime import datetime
from collections import Counter

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

# Import custom modules with error handling
try:
    from stock_explorer import fetch_stock_data

    STOCK_EXPLORER_AVAILABLE = True
except ImportError as e:
    logger.error(f"Stock explorer module not available: {e}")
    STOCK_EXPLORER_AVAILABLE = False

try:
    from trade_view import fetch_trade_views, render_trade_views_table, render_trade_views_summary, \
        render_sector_analysis, prepare_trade_views_dataframe

    TRADE_VIEW_AVAILABLE = True
except ImportError as e:
    logger.error(f"Trade view module not available: {e}")
    TRADE_VIEW_AVAILABLE = False

try:
    from investment_analysis import render_investment_analysis_tab

    INVESTMENT_ANALYSIS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Investment analysis module not available: {e}")
    INVESTMENT_ANALYSIS_AVAILABLE = False

try:
    from news_sentiment import fetch_news_data, render_news_sentiment_tab, render_sidebar_news_sentiment

    NEWS_SENTIMENT_AVAILABLE = True
except ImportError as e:
    logger.error(f"News sentiment module not available: {e}")
    NEWS_SENTIMENT_AVAILABLE = False

try:
    from strategy import render_strategy_builder

    STRATEGY_AVAILABLE = True
except ImportError as e:
    logger.error(f"Strategy module not available: {e}")
    STRATEGY_AVAILABLE = False

try:
    from learn import render_learn_tab

    LEARN_AVAILABLE = True
except ImportError as e:
    logger.error(f"Learn module not available: {e}")
    LEARN_AVAILABLE = False


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
        'selected_sentiment_filter': 'All',
        # Strategy builder session state
        'technical_indicators': [],
        'sentiment_rules': [],
        'trade_view_rules': [],
        'stock_score_rule': None,
        'market_cap_rule': None,
        'sector_rule': None,
        'volume_rule': None,
        'price_rule': None,
        'strategy_results': None,
        # Learning progress session state
        'learning_progress': {
            "completed_topics": set(),
            "bookmarks": set(),
            "current_phase": "Beginner",
            "study_time": 0,
            "quiz_scores": {}
        },
        # Quiz state
        'quiz_active': False,
        'current_question': 0,
        'quiz_score': 0
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# Initialize session state at the start
initialize_session_state()


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
            if STOCK_EXPLORER_AVAILABLE:
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
            import plotly.express as px
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


def render_sidebar():
    """Render simplified sidebar with proper error handling"""
    with st.sidebar:
        try:
            st.image("image.png", width=70)
        except:
            st.markdown("📈")

        # Portfolio Summary Section
        if st.session_state.get('edited_df') is not None:
            with st.expander("📊 Portfolio Summary", expanded=True):
                total_stocks = len(st.session_state.edited_df)
                total_invested = st.session_state.edited_df["Amount (₹)"].sum()

                st.metric("Total Stocks", total_stocks)
                st.metric("Total Investment", f"₹{total_invested:,.0f}")

                if st.session_state.get('last_edit_time'):
                    st.caption(f"Last updated: {st.session_state.last_edit_time.strftime('%H:%M:%S')}")

        # Strategy Summary Section
        if (st.session_state.get('strategy_results') and
                isinstance(st.session_state.strategy_results, list) and
                len(st.session_state.strategy_results) > 0):
            with st.expander("🎯 Strategy Results", expanded=True):
                results_count = len(st.session_state.strategy_results)
                st.metric("Matching Stocks", results_count)

                if results_count > 0:
                    # Calculate average strategy score
                    avg_score = sum(
                        result.get('overall_score', 0) for result in st.session_state.strategy_results) / results_count
                    st.metric("Avg Strategy Score", f"{avg_score:.1f}")

                    # Show top recommendation
                    top_stock = max(st.session_state.strategy_results, key=lambda x: x.get('overall_score', 0))
                    top_symbol = top_stock.get('stock', {}).get('symbol', 'N/A')
                    st.write(f"🏆 Top Pick: **{top_symbol}**")

        # Learning Progress Section (with proper error handling)
        try:
            learning_progress = st.session_state.get('learning_progress', {})
            completed_topics = learning_progress.get('completed_topics', set())

            if completed_topics and len(completed_topics) > 0:
                with st.expander("🎓 Learning Progress", expanded=True):
                    completed_count = len(completed_topics)
                    current_phase = learning_progress.get('current_phase', 'Beginner')

                    st.metric("Topics Completed", completed_count)
                    st.metric("Current Level", current_phase)

                    # Progress bar based on phase
                    phase_progress = {
                        "Beginner": min(completed_count / 5, 1.0),
                        "Intermediate": min(max(completed_count - 5, 0) / 8, 1.0),
                        "Advanced": min(max(completed_count - 13, 0) / 6, 1.0),
                        "Expert": min(max(completed_count - 19, 0) / 4, 1.0)
                    }

                    current_progress = phase_progress.get(current_phase, 0)
                    st.progress(current_progress)

                    if completed_count > 0:
                        st.caption("Keep learning! 📚")
        except Exception as e:
            logger.error(f"Error displaying learning progress: {e}")
            # Don't show error to user, just log it

        # News Sentiment Section
        if NEWS_SENTIMENT_AVAILABLE:
            try:
                render_sidebar_news_sentiment()
            except Exception as e:
                logger.error(f"Error rendering news sentiment: {e}")
                st.error("⚠️ News sentiment temporarily unavailable")

        # About Section
        with st.expander("ℹ️ About This App", expanded=False):
            st.markdown("""
            **Smart Stock Investment Advisor** helps you:
            - Get personalized recommendations
            - Analyze market sentiment
            - Track live stock data
            - Optimize your portfolio
            - Build custom strategies
            - Learn stock market fundamentals

            **Key Features:**
            - ✨ AI-powered analysis
            - 📈 Real-time data
            - 💰 Portfolio optimization
            - 📰 News sentiment tracking
            - 🎯 Custom strategy builder
            - 🎓 Interactive learning center

            Made with ❤️ by Team 777
            """)

            st.markdown("---")
            st.caption("v1.2 | Last updated: July 2025")


def render_fallback_tab(tab_name, description):
    """Render fallback content for unavailable tabs"""
    st.error(f"❌ {tab_name} module is not available")
    st.markdown(f"""
    ### {tab_name} (Temporarily Unavailable)

    The {tab_name.lower()} module is currently experiencing issues. This could be due to:
    1. Missing module dependencies
    2. Import errors
    3. Configuration issues

    **Expected Features:**
    {description}

    Please contact the administrator to resolve this issue.
    """)


def main():
    """Main application function"""
    st.title("📈 Smart Stock Investment Advisor")
    st.markdown(
        "*Get personalized stock recommendations with real-time market sentiment analysis and comprehensive learning resources*")

    # Load news data on app start if not already loaded and module is available
    if (NEWS_SENTIMENT_AVAILABLE and
            st.session_state.get('news_data') is None and
            st.session_state.get('show_news', True)):
        with st.spinner("Loading market news..."):
            try:
                st.session_state.news_data = fetch_news_data()
            except Exception as e:
                logger.error(f"Error loading news data: {e}")
                st.session_state.news_data = None

    # Main navigation with all tabs including Learning Center
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💰 Investment Analysis",
        "📰 News & Sentiment",
        "🔍 Stock Explorer",
        "🐂 Trade Views",
        "🎯 Strategy Builder",
        "🎓 Learn"
    ])

    with tab1:
        if INVESTMENT_ANALYSIS_AVAILABLE:
            try:
                render_investment_analysis_tab()
            except Exception as e:
                st.error(f"❌ Error loading investment analysis: {str(e)}")
                logger.error(f"Investment analysis error: {e}")
        else:
            render_fallback_tab("Investment Analysis", """
            - Personalized stock recommendations based on your risk profile
            - Portfolio optimization suggestions
            - Risk assessment and capital allocation
            - Investment strategy recommendations
            """)

    with tab2:
        if NEWS_SENTIMENT_AVAILABLE:
            try:
                render_news_sentiment_tab()
            except Exception as e:
                st.error(f"❌ Error loading news sentiment: {str(e)}")
                logger.error(f"News sentiment error: {e}")
        else:
            render_fallback_tab("News & Sentiment", """
            - Real-time market news aggregation
            - Sentiment analysis of news articles
            - Market mood indicators
            - News impact on stock prices
            """)

    with tab3:
        # Stock Explorer tab action buttons
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🔄 Refresh Stock Data", key="refresh_stocks",
                         help="Clear cache and reload fresh stock data", use_container_width=True):
                try:
                    if STOCK_EXPLORER_AVAILABLE:
                        fetch_stock_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error refreshing data: {str(e)}")

        if STOCK_EXPLORER_AVAILABLE:
            try:
                # Fetch and display stock data
                stock_data = fetch_stock_data()
                stock_df = prepare_stock_dataframe(stock_data) if stock_data else None
                render_stock_explorer(stock_df)
            except Exception as e:
                st.error(f"❌ Error loading stock explorer: {str(e)}")
                logger.error(f"Stock explorer error: {e}")
        else:
            render_fallback_tab("Stock Explorer", """
            - Browse and filter stock market data
            - Interactive stock charts and analysis
            - Sector performance comparison
            - Real-time price tracking
            - Advanced filtering and sorting options
            """)

    with tab4:
        if TRADE_VIEW_AVAILABLE:
            # Trade Views tab action buttons
            col1, col2 = st.columns([4, 1])
            with col2:
                if st.button("🔄 Refresh Trade Views", key="refresh_trade_views",
                             help="Clear cache and reload fresh trade views data", use_container_width=True):
                    try:
                        fetch_trade_views.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error refreshing trade views: {str(e)}")

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
        else:
            render_fallback_tab("Trade Views", """
            - Live bullish trade recommendations
            - Sector-wise trade analysis
            - Market sentiment from trading community
            - Real-time trade signals and alerts
            """)

    with tab5:
        if STRATEGY_AVAILABLE:
            # Strategy Builder tab
            try:
                # Add helpful description at the top
                st.info(
                    "🎯 **Strategy Builder**: Create custom stock screening strategies using technical indicators, "
                    "sentiment analysis, and fundamental filters. Build your strategy step by step and execute it "
                    "to find stocks that match your criteria.")

                # Render the strategy builder interface
                render_strategy_builder()

            except Exception as e:
                st.error(f"❌ Error loading strategy builder: {str(e)}")
                logger.error(f"Strategy builder error in main: {e}")

                # Provide fallback message
                st.markdown("""
                ### 🎯 Strategy Builder (Temporarily Unavailable)

                The strategy builder is currently experiencing issues. Please try:
                1. Refreshing the page
                2. Checking your internet connection
                3. Contacting support if the issue persists

                **What you can do with Strategy Builder:**
                - Create custom technical indicator rules
                - Set sentiment analysis filters  
                - Define fundamental criteria (price, volume, market cap)
                - Execute strategies to find matching stocks
                - Download results for further analysis
                """)
        else:
            render_fallback_tab("Strategy Builder", """
            - Custom stock screening strategies
            - Technical indicator combinations
            - Fundamental analysis filters
            - Backtesting capabilities
            - Strategy performance tracking
            """)

    with tab6:
        if LEARN_AVAILABLE:
            # Learning tab
            try:
                st.info(
                    "🎓 **Learning Center**: Master the stock market from beginner to expert level. "
                    "Interactive lessons, progress tracking, quizzes, and curated resources to enhance your investment knowledge.")

                render_learn_tab()

            except Exception as e:
                st.error(f"❌ Error loading learning center: {str(e)}")
                logger.error(f"Learning center error in main: {e}")

                # Provide fallback message with more detail
                st.markdown("""
                ### 🎓 Learning Center (Error Occurred)

                There was an error loading the learning center. This could be due to:
                1. Module import issues
                2. Session state conflicts
                3. Interactive component errors

                **Expected Learning Modules:**
                - **🌱 Beginner Level**: Stock market basics, terminology, how to start investing
                - **📈 Intermediate Level**: Fundamental analysis, technical analysis, portfolio management
                - **🎯 Advanced Level**: Options & derivatives, sector analysis, quantitative methods
                - **🏆 Expert Level**: Advanced strategies, behavioral finance, institutional approaches
                - **📝 Interactive Features**: Quizzes, progress tracking, live demonstrations
                - **📚 Resources**: Curated books, courses, and additional learning materials

                Please refresh the page or contact support if the issue persists.
                """)
        else:
            render_fallback_tab("Learning Center", """
            - Comprehensive stock market education from beginner to expert
            - Interactive lessons and demonstrations
            - Progress tracking and achievement system
            - Quizzes and knowledge assessments
            - Curated learning resources and recommendations
            """)

    # Render sidebar with proper error handling
    try:
        render_sidebar()
    except Exception as e:
        logger.error(f"Sidebar error: {e}")
        # Show minimal sidebar if there's an error
        with st.sidebar:
            st.markdown("📈 **Stock Investment Advisor**")
            st.error("⚠️ Sidebar temporarily unavailable")


if __name__ == "__main__":
    main()