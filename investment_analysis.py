import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import logging
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional, Any, Tuple
import time

# Configure logging
logger = logging.getLogger(__name__)

# Constants - Made immutable with tuples for better performance
SECTORS = (
    "Power", "Infrastructure", "FMCG", "Telecom", "Non - Ferrous Metals", "Healthcare",
    "Diamond & Jewellery", "Iron & Steel", "Finance", "IT", "Textile", "Hospitality",
    "Inds. Gases & Fuels", "Paper", "Business Services", "Automobile & Ancillaries",
    "Ship Building", "Aviation", "Diversified", "Chemicals", "Consumer Durables",
    "Plastic Products", "Alcohol", "Capital Goods", "Logistics", "Realty", "Abrasives",
    "Mining", "Electricals", "Media & Entertainment", "Bank", "Trading",
    "Gas Transmission", "Retailing", "Ratings", "Insurance", "Ferro Manganese",
    "Construction Materials", "Agri", "Crude Oil", "Miscellaneous"
)

RISK_PROFILES = ("Conservative", "Moderate", "Aggressive")
TIME_HORIZONS = ("Short-term: 1–4 weeks", "Medium-term: 1–3 months", "Long-term: 6–12 months")

# Frozen dict for better performance
TIMELINE_MAPPING = {
    "Short-term: 1–4 weeks": "short_term",
    "Medium-term: 1–3 months": "medium_term",
    "Long-term: 6–12 months": "long_term"
}

# Configuration constants
API_CONFIG = {
    "base_url": "https://stage.stockgro.com/hackathon",
    "timeout": 30,
    "retry_attempts": 3,
    "retry_delay": 1
}

VALIDATION_CONFIG = {
    "min_investment": 1000,
    "max_investment": 10000000,
    "min_sectors": 1
}

# DataFrame column configurations for better reusability
PORTFOLIO_COLUMNS = {
    "Sector": {"type": "text", "disabled": True, "width": "medium"},
    "Stock": {"type": "text", "disabled": True, "width": "medium"},
    "Shares": {"type": "number", "min_value": 0, "step": 1, "width": "small"},
    "Allocation (%)": {"type": "number", "format": "%.2f%%", "disabled": True, "width": "small"},
    "Price (₹)": {"type": "number", "format": "₹%.2f", "disabled": True, "width": "small"},
    "Risk": {"type": "text", "disabled": True, "width": "small"},
    "Target (₹)": {"type": "number", "format": "₹%.2f", "disabled": True, "width": "small"},
    "Expected Return (%)": {"type": "number", "format": "%.2f%%", "disabled": True, "width": "small"},
    "Expected Profit (₹)": {"type": "number", "format": "₹%.2f", "disabled": True, "width": "medium"},
    "Amount (₹)": {"type": "number", "format": "₹%.2f", "disabled": True, "width": "medium"}
}


@st.cache_data(ttl=300, show_spinner=False)
def convert_timeline(ui_timeline: str) -> str:
    """Convert UI timeline to API format with validation"""
    return TIMELINE_MAPPING.get(ui_timeline, "medium_term")


def validate_inputs(amount: float, sectors: List[str]) -> List[str]:
    """Comprehensive input validation with detailed error messages"""
    errors = []

    # Amount validation
    if not isinstance(amount, (int, float)) or amount <= 0:
        errors.append("Investment amount must be a positive number")
    elif amount < VALIDATION_CONFIG["min_investment"]:
        errors.append(f"Investment amount must be at least ₹{VALIDATION_CONFIG['min_investment']:,}")
    elif amount > VALIDATION_CONFIG["max_investment"]:
        errors.append(f"Investment amount cannot exceed ₹{VALIDATION_CONFIG['max_investment']:,}")

    # Sectors validation
    if not sectors or len(sectors) < VALIDATION_CONFIG["min_sectors"]:
        errors.append("Please select at least one sector")
    elif not all(sector in SECTORS for sector in sectors):
        invalid_sectors = [s for s in sectors if s not in SECTORS]
        errors.append(f"Invalid sectors selected: {', '.join(invalid_sectors)}")

    return errors


def make_api_request(url: str, payload: Dict[str, Any], method: str = "GET") -> Optional[Dict[str, Any]]:
    """Enhanced API request with retry logic and better error handling"""
    headers = {"Content-Type": "application/json"}

    for attempt in range(API_CONFIG["retry_attempts"]):
        try:
            if method.upper() == "GET":
                response = requests.get(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=API_CONFIG["timeout"]
                )
            else:  # POST
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=API_CONFIG["timeout"]
                )

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            if attempt < API_CONFIG["retry_attempts"] - 1:
                st.warning(f"Request timeout. Retrying... (Attempt {attempt + 2}/{API_CONFIG['retry_attempts']})")
                time.sleep(API_CONFIG["retry_delay"] * (attempt + 1))  # Exponential backoff
                continue
            else:
                st.error("Request timed out after multiple attempts. Please try again later.")

        except requests.exceptions.ConnectionError:
            if attempt < API_CONFIG["retry_attempts"] - 1:
                st.warning(f"Connection error. Retrying... (Attempt {attempt + 2}/{API_CONFIG['retry_attempts']})")
                time.sleep(API_CONFIG["retry_delay"] * (attempt + 1))
                continue
            else:
                st.error("Cannot connect to the API server. Please check if the server is running.")

        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP error: {e.response.status_code} - {e.response.reason}")
            break

        except Exception as e:
            logger.error(f"Unexpected API error: {str(e)}")
            st.error(f"API request failed: {str(e)}")
            break

    return None


def generate_api_response(amount: float, risk: str, sectors: List[str], timeline: str) -> Optional[Dict[str, Any]]:
    """Generate API response with enhanced error handling and validation"""
    api_url = f"{API_CONFIG['base_url']}/recommendations"

    payload = {
        "amount": float(amount),
        "risk": risk,
        "sectors": list(sectors),  # Ensure it's a list
        "timeline": convert_timeline(timeline)
    }

    # Log the request for debugging
    logger.info(f"Making API request: {payload}")

    try:
        api_data = make_api_request(api_url, payload)

        if not api_data:
            return None

        if api_data.get("success", False):
            # Validate response structure
            if "data" not in api_data or "stocks" not in api_data["data"]:
                st.error("Invalid API response format")
                return None

            return {
                "status": "success",
                "recommendations": api_data["data"]["stocks"],
                "total_amount": amount,
                "risk_profile": risk,
                "timeline": timeline,
                "timestamp": datetime.now().isoformat(),
                "request_payload": payload  # Store for debugging
            }
        else:
            error_msg = api_data.get('message', 'Unknown API error')
            st.error(f"API Error: {error_msg}")
            logger.error(f"API returned error: {error_msg}")
            return None

    except Exception as e:
        logger.error(f"Error in generate_api_response: {str(e)}")
        st.error(f"Failed to process API response: {str(e)}")
        return None


@st.cache_data(ttl=600, show_spinner=False)  # Increased TTL for better performance
def prepare_dataframe(response_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Optimized dataframe preparation with better error handling and validation"""
    try:
        if not response_data or 'recommendations' not in response_data:
            st.error("Invalid response data structure")
            return None

        portfolio_data = []
        recommendations = response_data['recommendations']
        total_amount = response_data['total_amount']

        # Required fields validation
        required_fields = {'symbol', 'allocation', 'price', 'risk', 'target', 'expected_return', 'expected_profit'}

        for sector, stocks in recommendations.items():
            if not isinstance(stocks, list):
                logger.warning(f"Invalid stocks format for sector {sector}")
                continue

            for stock in stocks:
                # Validate stock data structure
                if not isinstance(stock, dict):
                    logger.warning(f"Invalid stock data format in sector {sector}")
                    continue

                missing_fields = required_fields - set(stock.keys())
                if missing_fields:
                    logger.warning(f"Missing fields {missing_fields} for stock {stock.get('symbol', 'Unknown')}")
                    continue

                try:
                    # Safe numeric conversions with validation
                    allocation = float(stock['allocation'])
                    price = float(stock['price'])
                    target = float(stock['target'])
                    expected_return = float(stock['expected_return'])
                    expected_profit = float(stock['expected_profit'])
                    shares = int(stock.get('shares', 0))

                    # Validate numeric ranges
                    if allocation < 0 or allocation > 100:
                        logger.warning(f"Invalid allocation {allocation}% for {stock['symbol']}")
                        continue
                    if price <= 0 or target <= 0:
                        logger.warning(f"Invalid price/target for {stock['symbol']}")
                        continue

                    allocation_amount = total_amount * (allocation / 100)

                    portfolio_data.append({
                        "Sector": str(sector),
                        "Stock": str(stock['symbol']),
                        "Shares": shares,
                        "Allocation (%)": round(allocation, 2),
                        "Price (₹)": round(price, 2),
                        "Risk": str(stock['risk']),
                        "Target (₹)": round(target, 2),
                        "Expected Return (%)": round(expected_return, 2),
                        "Expected Profit (₹)": round(expected_profit, 2),
                        "Amount (₹)": round(allocation_amount, 2)
                    })

                except (ValueError, TypeError) as e:
                    logger.warning(f"Error processing stock {stock.get('symbol', 'Unknown')}: {e}")
                    continue

        if not portfolio_data:
            st.error("No valid stock data could be processed from API response")
            return None

        df = pd.DataFrame(portfolio_data)

        # Optimize DataFrame memory usage
        categorical_columns = ["Sector", "Stock", "Risk"]
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')

        logger.info(f"Successfully created portfolio DataFrame with {len(df)} stocks")
        return df

    except Exception as e:
        logger.error(f"Error in prepare_dataframe: {str(e)}")
        st.error(f"Error preparing portfolio data: {str(e)}")
        return None


def update_portfolio_calculations(df: pd.DataFrame) -> pd.DataFrame:
    """Optimized portfolio calculations with comprehensive error handling"""
    try:
        if df is None or df.empty:
            return df

        # Create a copy to avoid modifying the original
        df_updated = df.copy()

        # Validate required columns
        required_cols = ["Shares", "Price (₹)", "Target (₹)"]
        missing_cols = [col for col in required_cols if col not in df_updated.columns]
        if missing_cols:
            st.error(f"Missing required columns for calculations: {missing_cols}")
            return df

        # Ensure numeric data types
        for col in ["Shares", "Price (₹)", "Target (₹)"]:
            df_updated[col] = pd.to_numeric(df_updated[col], errors='coerce').fillna(0)

        # Perform calculations with validation
        df_updated["Amount (₹)"] = df_updated["Shares"] * df_updated["Price (₹)"]
        df_updated["Expected Profit (₹)"] = (df_updated["Target (₹)"] - df_updated["Price (₹)"]) * df_updated["Shares"]

        # Calculate allocations
        total_invested = df_updated["Amount (₹)"].sum()
        if total_invested > 0:
            df_updated["Allocation (%)"] = (df_updated["Amount (₹)"] / total_invested) * 100
        else:
            df_updated["Allocation (%)"] = 0

        # Round all numeric columns
        numeric_columns = df_updated.select_dtypes(include=['float64', 'int64']).columns
        df_updated[numeric_columns] = df_updated[numeric_columns].round(2)

        return df_updated

    except Exception as e:
        logger.error(f"Error in update_portfolio_calculations: {str(e)}")
        st.error(f"Error updating calculations: {str(e)}")
        return df


def create_column_config() -> Dict[str, Any]:
    """Create optimized column configuration for data editor"""
    column_config = {}

    for col_name, config in PORTFOLIO_COLUMNS.items():
        if config["type"] == "text":
            column_config[col_name] = st.column_config.TextColumn(
                col_name.replace(" (₹)", "").replace(" (%)", ""),
                disabled=config.get("disabled", False),
                width=config.get("width", "medium")
            )
        elif config["type"] == "number":
            column_config[col_name] = st.column_config.NumberColumn(
                col_name.replace(" (₹)", "").replace(" (%)", ""),
                help=config.get("help", ""),
                min_value=config.get("min_value"),
                step=config.get("step"),
                format=config.get("format"),
                disabled=config.get("disabled", False),
                width=config.get("width", "medium")
            )

    # Add specific help text for Shares column
    column_config["Shares"] = st.column_config.NumberColumn(
        "Shares",
        help="Adjust the number of shares to buy",
        min_value=0,
        step=1,
        width="small"
    )

    return column_config


def render_input_form():
    """Enhanced investment preferences form with better UX"""
    with st.expander("💰 Investment Preferences", expanded=True):
        with st.form("investment_preferences", clear_on_submit=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**📊 Investment Details**")

                selected_sectors = st.multiselect(
                    "Select stock sectors (at least one required):",
                    options=list(SECTORS),  # Convert tuple to list for multiselect
                    default=st.session_state.get('selected_sectors', ["Power", "Infrastructure"]),
                    help="Choose sectors you want to invest in",
                    key="sector_multiselect"
                )

                capital = st.number_input(
                    "Enter your investment capital (INR):",
                    min_value=VALIDATION_CONFIG["min_investment"],
                    max_value=VALIDATION_CONFIG["max_investment"],
                    step=1000,
                    value=st.session_state.get('capital', 100000),
                    help=f"Minimum investment: ₹{VALIDATION_CONFIG['min_investment']:,}",
                    format="%d"
                )

            with col2:
                st.markdown("**⚖️ Risk & Timeline**")

                risk_profile = st.radio(
                    "Select your risk preference:",
                    options=list(RISK_PROFILES),  # Convert tuple to list
                    index=st.session_state.get('risk_index', 0),
                    help="Choose based on your risk tolerance"
                )

                time_horizon = st.radio(
                    "Select your investment time horizon:",
                    options=list(TIME_HORIZONS),  # Convert tuple to list
                    index=st.session_state.get('horizon_index', 1),
                    help="Choose your investment duration"
                )

            # Add summary before submit
            if selected_sectors and capital >= VALIDATION_CONFIG["min_investment"]:
                st.info(
                    f"💡 **Summary:** Investing ₹{capital:,} across {len(selected_sectors)} sectors with {risk_profile.lower()} risk profile")

            submitted = st.form_submit_button(
                "🚀 Get Recommendations",
                use_container_width=True,
                type="primary"
            )

            if submitted:
                # Comprehensive validation
                errors = validate_inputs(capital, selected_sectors)
                if errors:
                    for error in errors:
                        st.error(f"❌ {error}")
                    return

                # Update session state
                st.session_state.selected_sectors = selected_sectors
                st.session_state.capital = capital
                st.session_state.risk_index = list(RISK_PROFILES).index(risk_profile)
                st.session_state.horizon_index = list(TIME_HORIZONS).index(time_horizon)

                # Show loading state with progress
                with st.spinner('🔄 Fetching personalized recommendations...'):
                    progress_bar = st.progress(0)

                    # Simulate progress updates
                    progress_bar.progress(25)

                    response = generate_api_response(
                        amount=capital,
                        risk=risk_profile,
                        sectors=selected_sectors,
                        timeline=time_horizon
                    )

                    progress_bar.progress(75)

                    if response:
                        df = prepare_dataframe(response)
                        progress_bar.progress(100)

                        if df is not None:
                            st.session_state.api_response = response
                            st.session_state.edited_df = df
                            st.success("✅ Recommendations generated successfully!")
                            progress_bar.empty()
                            st.rerun()
                        else:
                            progress_bar.empty()
                    else:
                        progress_bar.empty()


def render_portfolio_results():
    """Enhanced portfolio results with better visualization and UX"""
    if st.session_state.get('api_response') is None or st.session_state.get('edited_df') is None:
        return

    # Portfolio action buttons at the top
    col1, col2, col3 = st.columns([3, 1, 1])

    with col2:
        if st.button("🔄 Reset Portfolio", help="Reset to original recommendations", use_container_width=True):
            with st.spinner("Resetting portfolio..."):
                st.session_state.edited_df = prepare_dataframe(st.session_state.api_response)
                st.success("✅ Portfolio reset!")
                time.sleep(0.5)  # Brief pause for UX
                st.rerun()

    with col3:
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

    st.markdown("---")
    st.subheader("🎯 Your Personalized Portfolio Recommendations")

    # Enhanced investment summary
    with st.expander("📋 Current Investment Summary", expanded=True):
        response = st.session_state.api_response

        # Main metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("💰 Investment Capital", f"₹{response['total_amount']:,.0f}")
        with col2:
            st.metric("⚖️ Risk Profile", response['risk_profile'])
        with col3:
            st.metric("⏱️ Time Horizon", response['timeline'])
        with col4:
            total_stocks = len(st.session_state.edited_df)
            st.metric("📊 Total Stocks", f"{total_stocks}")

        # Additional summary information
        sectors_list = ', '.join(response['recommendations'].keys())
        st.info(f"**Selected Sectors:** {sectors_list}")

        # Timestamp information
        try:
            timestamp = datetime.fromisoformat(response['timestamp'])
            st.caption(f"Generated: {timestamp.strftime('%Y-%m-%d at %H:%M:%S')}")
        except:
            pass

    # Enhanced portfolio editor
    with st.expander("✏️ Customize Your Portfolio", expanded=True):
        st.markdown("**Instructions:** Adjust the number of shares for each stock to customize your allocation.")

        # Use optimized column configuration
        column_config = create_column_config()

        edited_df = st.data_editor(
            st.session_state.edited_df,
            column_config=column_config,
            hide_index=True,
            use_container_width=True,
            key="portfolio_editor",
            height=400  # Fixed height for better UX
        )

        # Check for changes and update calculations
        if not edited_df.equals(st.session_state.edited_df):
            st.session_state.edited_df = update_portfolio_calculations(edited_df)
            st.session_state.last_edit_time = datetime.now()
            st.rerun()

        # Enhanced summary metrics
        st.markdown("### 📊 Portfolio Summary")
        col1, col2, col3, col4 = st.columns(4)

        total_invested = st.session_state.edited_df["Amount (₹)"].sum()
        total_expected_profit = st.session_state.edited_df["Expected Profit (₹)"].sum()
        avg_expected_return = st.session_state.edited_df["Expected Return (%)"].mean()
        total_stocks_with_shares = (st.session_state.edited_df["Shares"] > 0).sum()

        with col1:
            st.metric("💵 Total Invested", f"₹{total_invested:,.2f}")
        with col2:
            st.metric("💰 Expected Profit", f"₹{total_expected_profit:,.2f}")
        with col3:
            st.metric("📈 Avg Expected Return", f"{avg_expected_return:.2f}%")
        with col4:
            st.metric("🎯 Active Positions", f"{total_stocks_with_shares}")

        # Cash management
        remaining_cash = st.session_state.api_response['total_amount'] - total_invested
        if remaining_cash > 0:
            st.info(
                f"💵 **Remaining Cash:** ₹{remaining_cash:,.2f} ({(remaining_cash / st.session_state.api_response['total_amount'] * 100):.1f}% of total)")
        elif remaining_cash < 0:
            st.warning(f"⚠️ **Over-invested by:** ₹{abs(remaining_cash):,.2f}")
        else:
            st.success("✅ **Fully Invested** - No remaining cash")


def render_enhanced_visualizations():
    """Enhanced portfolio visualizations with better interactivity"""
    if st.session_state.get('edited_df') is None:
        return

    st.subheader("📊 Portfolio Visualization Dashboard")

    # Filter out zero-share positions for cleaner visualizations
    active_df = st.session_state.edited_df[st.session_state.edited_df["Shares"] > 0]

    if active_df.empty:
        st.info("📊 Adjust share quantities above to see portfolio visualizations")
        return

    # Main visualization row
    col1, col2 = st.columns(2)

    with col1:
        # Enhanced sector allocation pie chart
        sector_data = active_df.groupby('Sector')['Amount (₹)'].sum().reset_index()
        fig1 = px.pie(
            sector_data,
            values='Amount (₹)',
            names='Sector',
            title="💼 Portfolio Allocation by Sector",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig1.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Amount: ₹%{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
        )
        fig1.update_layout(showlegend=True, height=450)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Enhanced top stocks bar chart
        top_stocks = active_df.nlargest(10, 'Amount (₹)')
        fig2 = px.bar(
            top_stocks,
            x='Stock',
            y='Amount (₹)',
            color='Sector',
            title="🏆 Top 10 Stock Allocations",
            text='Amount (₹)',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig2.update_traces(
            texttemplate='₹%{text:,.0f}',
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Amount: ₹%{y:,.0f}<br>Sector: %{customdata}<extra></extra>'
        )
        fig2.update_layout(xaxis_tickangle=-45, height=450, showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

    # Additional analytics row
    col3, col4 = st.columns(2)

    with col3:
        # Risk distribution
        risk_data = active_df.groupby('Risk')['Amount (₹)'].sum().reset_index()
        fig3 = px.bar(
            risk_data,
            x='Risk',
            y='Amount (₹)',
            title="⚖️ Investment by Risk Level",
            color='Risk',
            color_discrete_map={
                'Conservative': '#2E8B57',
                'Moderate': '#FF8C00',
                'Aggressive': '#DC143C'
            }
        )
        fig3.update_traces(
            texttemplate='₹%{y:,.0f}',
            textposition='outside'
        )
        fig3.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Expected returns scatter plot
        fig4 = px.scatter(
            active_df,
            x='Amount (₹)',
            y='Expected Return (%)',
            size='Shares',
            color='Sector',
            hover_name='Stock',
            title="📈 Expected Returns vs Investment Amount",
            size_max=20
        )
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)


def render_investment_analysis_tab():
    """Render the complete optimized investment analysis tab"""
    render_input_form()
    render_portfolio_results()
    if st.session_state.get('edited_df') is not None:
        render_enhanced_visualizations()


# Additional utility functions for better maintainability
def get_portfolio_summary() -> Dict[str, Any]:
    """Get comprehensive portfolio summary for sidebar"""
    if st.session_state.get('edited_df') is None:
        return {}

    df = st.session_state.edited_df
    active_df = df[df["Shares"] > 0]

    return {
        "total_stocks": len(df),
        "active_positions": len(active_df),
        "total_invested": active_df["Amount (₹)"].sum(),
        "expected_profit": active_df["Expected Profit (₹)"].sum(),
        "avg_return": active_df["Expected Return (%)"].mean(),
        "sectors": active_df["Sector"].nunique(),
        "last_updated": st.session_state.get('last_edit_time')
    }


def export_portfolio_data(format_type: str = "csv") -> bytes:
    """Export portfolio data in various formats"""
    if st.session_state.get('edited_df') is None:
        return b""

    df = st.session_state.edited_df

    if format_type.lower() == "csv":
        return df.to_csv(index=False).encode('utf-8')
    elif format_type.lower() == "json":
        return df.to_json(orient='records', indent=2).encode('utf-8')
    else:
        return df.to_csv(index=False).encode('utf-8')