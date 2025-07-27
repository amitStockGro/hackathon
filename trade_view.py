import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional, Tuple
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BULLISH_COLOR = "#4CAF50"
COLOR_SCHEME = px.colors.sequential.Greens
API_TIMEOUT = 30
CACHE_TTL = 300

# Numeric columns for DataFrame processing
NUMERIC_COLUMNS = [
    "Entry Price", "Current LTP", "Target Price", "Stop Loss",
    "Percentage Achieved"
]


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_trade_views() -> Optional[Dict[str, List[Dict]]]:
    """Fetch trade views data from API with enhanced error handling"""
    try:
        with st.spinner("🔄 Fetching latest trade views..."):
            logger.info("Fetching trade views data from API")
            response = requests.get(
                "https://stage.stockgro.com/hackathon/trade-views",
                timeout=1
            )
            response.raise_for_status()

            api_data = response.json()
            if not api_data.get("success", False):
                error_msg = api_data.get('message', 'Unknown API error')
                st.error(f"🚫 Trade Views API Error: {error_msg}")
                logger.error(f"API returned error: {error_msg}")
                return None

            data = api_data.get("data", {})
            logger.info(f"Successfully fetched trade views for {len(data)} symbols")
            return data

    except requests.exceptions.Timeout:
        st.error("⏰ Request timed out. The server might be slow - please try again.")
        logger.error("API request timeout")
    except requests.exceptions.ConnectionError:
        st.error("🔌 Could not connect to the trade views API server.")
        logger.error("Connection error")
    except requests.exceptions.HTTPError as e:
        st.error(f"🚫 HTTP Error {e.response.status_code}: Server returned an error")
        logger.error(f"HTTP error: {e}")
    except Exception as e:
        st.error(f"❌ Unexpected error occurred: {str(e)}")
        logger.error(f"Unexpected error: {e}")

    return None


@lru_cache(maxsize=1)
def _calculate_trade_metrics(entry_price: float, current_ltp: float,
                             target_price: float, stop_loss: float) -> float:
    """Calculate trade metrics with caching for repeated calculations"""
    if current_ltp <= 0:
        return 0.0

    downside_risk = round(((current_ltp - stop_loss) / current_ltp) * 100, 2)

    return downside_risk


def _process_single_view(symbol: str, view: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a single trade view into structured data"""
    # Filter for live Bullish trades only
    if view.get("status") != "live_trade" or view.get("direction") != "Bullish":
        return None

    # Extract required values with defaults
    entry_price = view.get("entry_price", 0)
    current_ltp = view.get("current_ltp", 0)
    target_price = view.get("target_price", 0)
    stop_loss = view.get("stop_loss_price", 0)

    # Format entry date
    entry_date = "N/A"
    if view.get("entry_date"):
        try:
            entry_date = datetime.fromtimestamp(view["entry_date"]).strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            logger.warning(f"Invalid entry_date for {symbol}: {view.get('entry_date')}")

    return {
        "Symbol": symbol,
        "Company": view.get("company_name", "N/A"),
        "Sector": view.get("sector", "Unknown").title(),
        "Entry Price": entry_price,
        "Current LTP": current_ltp,
        "Target Price": target_price,
        "Stop Loss": stop_loss,
        "Percentage Achieved": view.get("percentage_achieved", 0),
        "Entry Date": entry_date
    }


def prepare_trade_views_dataframe(trade_views_data: Dict[str, List[Dict]]) -> Optional[pd.DataFrame]:
    """Prepare optimized dataframe from trade views API response"""
    if not trade_views_data:
        return None

    try:
        logger.info("Processing trade views data into DataFrame")

        # Process all views using list comprehension for better performance
        processed_data = [
            processed_view
            for symbol, views in trade_views_data.items()
            for view in views
            if (processed_view := _process_single_view(symbol, view)) is not None
        ]

        if not processed_data:
            logger.warning("No live Bullish trades found in the data")
            return None

        # Create DataFrame efficiently
        df = pd.DataFrame(processed_data)

        # Optimize data types
        df = _optimize_dataframe_dtypes(df)

        logger.info(f"Successfully processed DataFrame with {len(df)} live Bullish trades")
        return df

    except Exception as e:
        st.error(f"❌ Error processing trade views data: {str(e)}")
        logger.error(f"DataFrame processing error: {e}")
        return None


def _optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame data types for better memory usage and performance"""
    # Convert numeric columns efficiently
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert categorical columns
    if "Sector" in df.columns:
        df["Sector"] = df["Sector"].astype('category')

    # Convert string columns to category if they have limited unique values
    for col in ["Symbol", "Company"]:
        if col in df.columns and df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype('category')

    return df


def _calculate_summary_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate summary metrics for trade views"""
    if df.empty:
        return {}

    return {
        "total_trades": len(df),
        "positive_percentage": (df["Percentage Achieved"] > 0).mean() * 100,
        "avg_percentage_achieved": df["Percentage Achieved"].mean(),
        "sectors_count": df["Sector"].nunique()
    }


def render_trade_views_summary(df: pd.DataFrame):
    """Render summary statistics and KPIs for trade views"""
    st.subheader("📊 Bullish Trade Views Summary")

    if df.empty:
        st.warning("No live Bullish trades available")
        return

    metrics = _calculate_summary_metrics(df)

    # Display KPIs in columns
    cols = st.columns(4)

    with cols[0]:
        st.metric("Total Bullish Trades", metrics["total_trades"])
    with cols[1]:
        st.metric("Avg % Achieved", f"{metrics['avg_percentage_achieved']:.1f}%")
    with cols[2]:
        st.metric("Positive Trades", f"{metrics['positive_percentage']:.1f}%")
    with cols[3]:
        st.metric("Active Sectors", metrics["sectors_count"])

    # Additional insights
    st.markdown(f"""
    📈 **{metrics['positive_percentage']:.1f}%** of trades are currently in positive territory  
    📊 **Average Performance:** {metrics['avg_percentage_achieved']:.1f}% achieved
    """)


def render_sector_analysis(df: pd.DataFrame):
    """Render sector-wise analysis of trade views"""
    if df.empty:
        return

    st.subheader("🏭 Sector-wise Analysis")

    # Calculate sector metrics efficiently
    sector_stats = df.groupby("Sector", observed=True).agg({
        "Symbol": "count",
        "Percentage Achieved": ["mean", "max"]
    }).round(2)

    # Flatten column names
    sector_stats.columns = ["Count", "Avg_Performance", "Max_Performance"]
    sector_stats = sector_stats.reset_index().sort_values("Avg_Performance", ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        # Sector distribution pie chart
        fig_pie = px.pie(
            sector_stats,
            names="Sector",
            values="Count",
            title="Trade Distribution by Sector",
            color_discrete_sequence=COLOR_SCHEME
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(showlegend=True, legend=dict(orientation="v"))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Average performance by sector
        fig_bar = px.bar(
            sector_stats,
            x="Sector",
            y="Avg_Performance",
            color="Avg_Performance",
            title="Average Performance by Sector",
            labels={"Avg_Performance": "Avg Performance (%)"},
            color_continuous_scale=COLOR_SCHEME,
            text="Avg_Performance"
        )
        fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_bar.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)


def _apply_filters(df: pd.DataFrame, selected_sectors: List[str],
                   min_percentage: float = 0) -> pd.DataFrame:
    """Apply filters to the DataFrame efficiently"""
    return df[
        (df["Sector"].isin(selected_sectors)) &
        (df["Percentage Achieved"] >= min_percentage)
        ]


def _format_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Format DataFrame columns for better display"""
    display_df = df.copy()

    # Format percentage columns
    percentage_cols = ["Percentage Achieved"]
    for col in percentage_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")

    # Format price columns
    price_cols = ["Entry Price", "Current LTP", "Target Price", "Stop Loss"]
    for col in price_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"₹{x:.2f}" if pd.notna(x) else "N/A")

    return display_df


def render_trade_views_table(df: pd.DataFrame):
    """Render interactive table of trade views with filtering options"""
    if df.empty:
        return

    st.subheader("📋 Live Bullish Trades")

    # Filter controls in columns
    col1, col2 = st.columns(2)

    with col1:
        sectors = sorted(df["Sector"].cat.categories if df["Sector"].dtype.name == 'category'
                         else df["Sector"].unique())
        selected_sectors = st.multiselect(
            "Filter by Sector:",
            options=sectors,
            default=sectors,
            help="Select sectors to display"
        )

    with col2:
        min_percentage = st.slider(
            "Minimum Percentage Achieved (%):",
            min_value=float(df["Percentage Achieved"].min()),
            max_value=float(df["Percentage Achieved"].max()),
            value=float(df["Percentage Achieved"].min()),
            step=1.0,
            format="%.1f"
        )

    # Apply filters
    filtered_df = _apply_filters(df, selected_sectors, min_percentage)

    if filtered_df.empty:
        st.warning("No trades match your filter criteria")
        return

    # Sort options
    sort_col = st.selectbox(
        "Sort by:",
        options=["Percentage Achieved", "Current LTP", "Entry Date"],
        index=0
    )

    # Sort data
    filtered_df = filtered_df.sort_values(sort_col, ascending=False)

    # Display summary of filtered results
    st.info(f"Showing {len(filtered_df)} trades out of {len(df)} total")

    # Format and display table
    display_df = _format_display_dataframe(filtered_df)

    display_columns = [
        "Symbol", "Company", "Sector", "Entry Price", "Current LTP",
        "Target Price", "Stop Loss", "Percentage Achieved", "Entry Date"
    ]

    st.dataframe(
        display_df[display_columns],
        use_container_width=True,
        height=400,
        hide_index=True
    )

    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Trades as CSV",
        data=csv,
        file_name=f"bullish_trades_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        help="Download the filtered data for further analysis"
    )


def render_trade_views():
    """Main function to render the trade views interface"""
    st.set_page_config(
        page_title="Bullish Trade Views",
        page_icon="🐂",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.title("🐂 Live Bullish Trade Views")
    st.markdown("Real-time analysis of active bullish trade recommendations")

    # Sidebar for additional controls
    with st.sidebar:
        st.header("Controls")
        auto_refresh = st.checkbox("Auto-refresh data", value=False)
        if auto_refresh:
            st.rerun()

        if st.button("🔄 Manual Refresh", key="refresh_trade_views"):
            fetch_trade_views.clear()
            st.rerun()

    # Fetch and process data
    with st.spinner("Loading trade data..."):
        trade_views_data = fetch_trade_views()
        trade_views_df = prepare_trade_views_dataframe(trade_views_data) if trade_views_data else None

    if trade_views_df is None or trade_views_df.empty:
        st.warning("⚠️ No live bullish trades available at the moment.")
        st.info("This could be due to market conditions or data availability. Please try refreshing in a few minutes.")
        return

    # Render components
    render_trade_views_summary(trade_views_df)

    st.markdown("---")
    render_sector_analysis(trade_views_df)

    st.markdown("---")
    render_trade_views_table(trade_views_df)

    # Footer with last updated time
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
