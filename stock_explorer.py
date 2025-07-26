import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
from typing import Optional, Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants - Fixed mapping to match DataFrame columns
METRIC_OPTIONS = {
    "Last Trade Price": "Last Trade Price",
    "Market Cap": "Market Cap",
    "Volume": "Volume",
    "52 Week High": "52 Week High",
    "52 Week Low": "52 Week Low",
    "Today's High": "Today's High",
    "Today's Low": "Today's Low",
    "Delta Percentage": "Delta Percentage",
    "Analyst Recommendation": "Analyst Recommendation",
    "Sentiment Score": "Sentiment Score"
}

# Numeric columns for proper aggregation
NUMERIC_COLUMNS = [
    "Last Trade Price", "Volume", "52 Week High", "52 Week Low",
    "Today's High", "Today's Low", "Delta Percentage", "Score Count"
]


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data():
    """Fetch stock data from API with enhanced error handling"""
    try:
        with st.spinner("🔄 Fetching latest stock data..."):
            logger.info("Fetching stock data from API")
            response = requests.get("https://stage.stockgro.com/hackathon/stocks", timeout=30)
            response.raise_for_status()

            api_data = response.json()
            if api_data.get("success", False):
                data = api_data.get("data", [])
                logger.info(f"Successfully fetched {len(data)} stocks")
                return data
            else:
                error_msg = api_data.get('message', 'Unknown API error')
                st.error(f"🚫 Stock API Error: {error_msg}")
                logger.error(f"API returned error: {error_msg}")
                return None

    except requests.exceptions.Timeout:
        st.error("⏰ Request timed out. The server might be slow - please try again.")
        logger.error("API request timeout")
    except requests.exceptions.ConnectionError:
        st.error(f"🔌 Connection Error Details: {str(requests.exceptions.ConnectionError)}")
        logger.error(f"Detailed connection error: {str(requests.exceptions.ConnectionError)}")
    except requests.exceptions.HTTPError as e:
        st.error(f"🚫 HTTP Error {e.response.status_code}: Server returned an error")
        logger.error(f"HTTP error: {e}")
    except Exception as e:
        st.error(f"❌ Unexpected error occurred: {str(e)}")
        logger.error(f"Unexpected error: {e}")

    return None


def safe_convert_numeric(value: Any, default: float = 0.0) -> float:
    """Safely convert value to numeric with fallback"""
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def safe_convert_int(value: Any, default: int = 0) -> int:
    """Safely convert value to integer with fallback"""
    try:
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def safe_timestamp_convert(timestamp: Any) -> Optional[datetime]:
    """Safely convert timestamp to datetime"""
    try:
        return datetime.fromtimestamp(int(timestamp)) if timestamp else None
    except (ValueError, TypeError, OSError):
        return None


def prepare_stock_dataframe(stock_data: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
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
        # Convert categorical columns
        categorical_cols = ["Sector", "Market Cap", "Analyst Recommendation", "Sentiment Score"]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')

        # Ensure numeric columns are properly typed
        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        logger.info(f"Successfully processed DataFrame with {len(df)} rows and {len(df.columns)} columns")
        return df

    except Exception as e:
        st.error(f"❌ Error processing stock data: {str(e)}")
        logger.error(f"DataFrame processing error: {e}")
        return None


def create_enhanced_scatter_plot(df: pd.DataFrame, x_metric: str, y_metric: str) -> go.Figure:
    """Create enhanced scatter plot with better error handling"""
    # Validate that metrics exist in DataFrame
    if x_metric not in df.columns or y_metric not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Selected metrics not available in current data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig

    # Filter out invalid data
    plot_df = df.dropna(subset=[x_metric, y_metric])

    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid data available for selected metrics",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig

    # Create scatter plot
    fig = px.scatter(
        plot_df,
        x=x_metric,
        y=y_metric,
        color="Sector",
        hover_name="Symbol",
        size="Volume" if plot_df["Volume"].max() > 0 else None,
        title=f"{x_metric} vs {y_metric}",
        template="plotly_white"
    )

    # Apply log scale only if appropriate (positive values)
    if plot_df[x_metric].min() > 0 and plot_df[x_metric].max() / plot_df[x_metric].min() > 10:
        fig.update_xaxes(type="log")
    if plot_df[y_metric].min() > 0 and plot_df[y_metric].max() / plot_df[y_metric].min() > 10:
        fig.update_yaxes(type="log")

    fig.update_layout(height=500)
    return fig


def create_enhanced_bar_chart(df: pd.DataFrame, x_metric: str, y_metric: str) -> go.Figure:
    """Create enhanced bar chart with validation and improved layout"""
    # Check if metrics exist in DataFrame
    if x_metric not in df.columns or y_metric not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Selected metrics not available in current data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        return fig

    # Check if metrics are numeric for proper aggregation
    x_is_numeric = x_metric in NUMERIC_COLUMNS
    y_is_numeric = y_metric in NUMERIC_COLUMNS

    if not (x_is_numeric and y_is_numeric):
        fig = go.Figure()
        fig.add_annotation(
            text="Please select numeric metrics for meaningful aggregation",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        return fig

    try:
        agg_df = df.groupby("Sector").agg({
            x_metric: "mean",
            y_metric: "mean",
            "Symbol": "count"
        }).reset_index().round(2)

        if agg_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for aggregation",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        fig = px.bar(
            agg_df,
            x="Sector",
            y=[x_metric, y_metric],
            title=f"Average {x_metric} and {y_metric} by Sector",
            barmode="group",
            labels={"value": "Average Value", "variable": "Metric"},
            hover_data=["Symbol"],
            template="plotly_white"
        )

        # Improved layout to prevent overlapping
        fig.update_layout(
            height=600,  # Increased height
            xaxis_tickangle=-45,  # Angled labels
            xaxis=dict(
                tickmode='linear',
                automargin=True  # Auto margin for labels
            ),
            margin=dict(b=120, l=60, r=60, t=80),  # Increased bottom margin
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            font=dict(size=12)
        )

        # Update x-axis to handle long sector names
        fig.update_xaxes(
            tickangle=-45,
            tickfont=dict(size=10)
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating bar chart: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig


def render_stock_explorer():
    """Main function to render the stock explorer - designed to be called from main.py"""
    # Add refresh button at the top
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔄 Refresh Data", type="primary", help="Clear cache and reload fresh data"):
            fetch_stock_data.clear()
            st.rerun()

    # Fetch and process data
    stock_data = fetch_stock_data()
    stock_df = prepare_stock_dataframe(stock_data) if stock_data else None

    # Render the explorer interface
    render_stock_explorer_interface(stock_df)


def render_stock_explorer_interface(stock_df: Optional[pd.DataFrame]):
    """Enhanced stock explorer interface with better UX"""
    if stock_df is None or stock_df.empty:
        st.warning("📭 No stock data available. Please try refreshing or check your connection.")
        if st.button("🔄 Retry", type="primary", key="retry_button"):
            fetch_stock_data.clear()
            st.rerun()
        return

    # Header with key metrics
    st.subheader("🔍 Stock Market Explorer")

    # Display summary statistics - removed avg price and total volume
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📊 Total Stocks", len(stock_df))
    with col2:
        st.metric("🏭 Sectors", stock_df["Sector"].nunique())

    # Enhanced filters with better organization
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

    # Apply filters with feedback
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

    # Enhanced metric selection
    st.markdown("### 📊 Analysis Configuration")
    col1, col2 = st.columns(2)

    with col1:
        x_metric = st.selectbox(
            "📈 X-axis Metric:",
            options=list(METRIC_OPTIONS.keys()),
            index=0,
            help="Choose the metric to display on the X-axis",
            key="x_metric_select"
        )

    with col2:
        y_metric = st.selectbox(
            "📊 Y-axis Metric:",
            options=list(METRIC_OPTIONS.keys()),
            index=1,
            help="Choose the metric to display on the Y-axis",
            key="y_metric_select"
        )

    # Enhanced visualization tabs
    tab1, tab2, tab3 = st.tabs(["📈 Scatter Analysis", "📊 Sector Comparison", "📋 Data Explorer"])

    with tab1:
        st.markdown(f"#### {x_metric} vs {y_metric}")
        fig = create_enhanced_scatter_plot(filtered_df, x_metric, y_metric)
        st.plotly_chart(fig, use_container_width=True)

        # Add correlation info for numeric metrics
        if x_metric in NUMERIC_COLUMNS and y_metric in NUMERIC_COLUMNS:
            try:
                corr = filtered_df[[x_metric, y_metric]].corr().iloc[0, 1]
                if not pd.isna(corr):
                    st.info(f"📊 Correlation coefficient: {corr:.3f}")
            except Exception:
                pass

    with tab2:
        st.markdown("#### Average Metrics by Sector")
        fig = create_enhanced_bar_chart(filtered_df, x_metric, y_metric)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("#### Detailed Stock Data")

        # Enhanced table controls - removed ascending order button
        col1, col2 = st.columns(2)
        with col1:
            sort_column = st.selectbox(
                "📊 Sort by:",
                options=filtered_df.columns.tolist(),
                index=2,  # Default to Last Trade Price
                help="Choose column to sort the data table",
                key="sort_column_select"
            )
        with col2:
            show_rows = st.selectbox(
                "📄 Rows to show:",
                options=[10, 25, 50, 100, "All"],
                index=1,
                help="Number of rows to display",
                key="show_rows_select"
            )

        # Sort and display data - default to descending order
        display_df = filtered_df.sort_values(by=sort_column, ascending=False)

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