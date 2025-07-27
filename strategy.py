import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
API_TIMEOUT = 30
CACHE_TTL = 300

# Technical Indicators Options
TECHNICAL_INDICATORS = {
    "sma": "Simple Moving Average",
    "rsi": "Relative Strength Index",
    "macd": "MACD",
    "atr": "Average True Range",
    "momentum": "Momentum",
    "roc": "Rate of Change",
    "stddev": "Standard Deviation",
    "bollinger_width": "Bollinger Band Width",
    "obv": "On Balance Volume",
    "volume_roc": "Volume Rate of Change"
}

OPERATORS = {
    "gt": "Greater Than (>)",
    "gte": "Greater Than or Equal (>=)",
    "lt": "Less Than (<)",
    "lte": "Less Than or Equal (<=)",
    "eq": "Equal To (=)",
    "between": "Between"
}

SENTIMENT_LABELS = ["positive", "negative", "neutral"]
TRADE_STATUSES = ["live_trade"]
MARKET_CAP_CATEGORIES = ["large", "mid", "small"]
RECOMMENDATIONS = ["BUY"]

# Available sectors based on common Indian market sectors
SECTORS = [
    "Technology", "Healthcare", "Finance", "Energy", "Consumer",
    "Industrial", "Materials", "Utilities", "Real Estate", "Telecom",
    "Automotive", "Banking", "Pharmaceuticals", "IT Services", "FMCG"
]

def create_strategy_payload(strategy_config: Dict) -> Dict:
    """Create API payload from strategy configuration"""
    payload = {
        "technical_indicators": [],
        "sentiment_rules": [],
        "trade_view_rules": [],
        "stock_score_rule": None,
        "market_cap_rule": None,
        "sector_rule": None,
        "volume_rule": None,
        "price_rule": None,
        "limit": strategy_config.get("limit", 10)
    }

    # Technical Indicators
    for indicator in strategy_config.get("technical_indicators", []):
        tech_rule = {
            "type": indicator["type"],
            "period": indicator["period"],
            "operator": indicator["operator"],
            "value": round(float(indicator["value"]), 2)
        }
        if indicator["operator"] == "between":
            tech_rule["value2"] = round(float(indicator["value2"]), 2)
        if indicator["type"] == "macd":
            tech_rule["slow_period"] = indicator.get("slow_period", 26)
            tech_rule["signal_period"] = indicator.get("signal_period", 9)
        if indicator["type"] == "bollinger_width":
            tech_rule["std_dev_multiplier"] = round(float(indicator.get("std_dev_multiplier", 2.0)), 2)

        payload["technical_indicators"].append(tech_rule)

    # Sentiment Rules
    for sentiment in strategy_config.get("sentiment_rules", []):
        payload["sentiment_rules"].append({
            "score_operator": sentiment["score_operator"],
            "score_value": round(float(sentiment["score_value"]), 2),
            "label_filter": sentiment.get("label_filter", "")
        })

    # Trade View Rules
    for trade_view in strategy_config.get("trade_view_rules", []):
        payload["trade_view_rules"].append({
            "profit_percentage": round(float(trade_view["profit_percentage"]), 2),
            "status": trade_view.get("status", ""),
            "days_active": trade_view.get("days_active", 0)
        })

    # Stock Score Rule
    if strategy_config.get("stock_score_rule"):
        score_rule = strategy_config["stock_score_rule"]
        payload["stock_score_rule"] = {
            "min_score": round(float(score_rule.get("min_score", 0)), 2),
            "min_score_count": score_rule.get("min_score_count", 0),
            "recommendation": score_rule.get("recommendation", "")
        }

    # Market Cap Rule
    if strategy_config.get("market_cap_rule"):
        payload["market_cap_rule"] = {
            "category": strategy_config["market_cap_rule"]["category"]
        }

    # Sector Rule
    if strategy_config.get("sector_rule"):
        sector_rule = strategy_config["sector_rule"]
        payload["sector_rule"] = {
            "include": sector_rule.get("include", []),
            "exclude": sector_rule.get("exclude", [])
        }

    # Volume Rule
    if strategy_config.get("volume_rule"):
        volume_rule = strategy_config["volume_rule"]
        payload["volume_rule"] = {
            "min_volume": volume_rule.get("min_volume", 0),
            "max_volume": volume_rule.get("max_volume", 0)
        }

    # Price Rule
    if strategy_config.get("price_rule"):
        price_rule = strategy_config["price_rule"]
        payload["price_rule"] = {
            "min_price": round(float(price_rule.get("min_price", 0)), 2),
            "max_price": round(float(price_rule.get("max_price", 0)), 2)
        }

    return payload


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def execute_strategy(payload: Dict) -> Optional[Dict]:
    """Execute strategy via API with improved error handling"""
    try:
        logger.info("Executing strategy via API")
        logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

        response = requests.get(
            "http://localhost:80/hackathon/strategy/result",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=API_TIMEOUT
        )
        response.raise_for_status()

        result = response.json()

        # Handle API response format
        if isinstance(result, dict) and "data" in result:
            if result.get("success", False):
                data = result["data"]
                logger.info(f"Strategy executed successfully, found {len(data)} results")
                return data
            else:
                error_msg = result.get("message", "Unknown API error")
                st.error(f"❌ API Error: {error_msg}")
                logger.error(f"API returned error: {error_msg}")
                return None
        else:
            # Assume direct data array for backward compatibility
            logger.info(f"Strategy executed successfully, found {len(result)} results")
            return result

    except requests.exceptions.Timeout:
        st.error("⏰ Request timed out. The server might be processing - please try again.")
        logger.error("Strategy API request timeout")
    except requests.exceptions.ConnectionError:
        st.error("🔌 Could not connect to the strategy API server. Please check if the server is running.")
        logger.error("Strategy API connection error")
    except requests.exceptions.HTTPError as e:
        st.error(f"🚫 HTTP Error {e.response.status_code}: {e.response.text}")
        logger.error(f"Strategy API HTTP error: {e}")
    except json.JSONDecodeError:
        st.error("❌ Invalid response format from server")
        logger.error("Strategy API JSON decode error")
    except Exception as e:
        st.error(f"❌ Unexpected error occurred: {str(e)}")
        logger.error(f"Strategy API unexpected error: {e}")

    return None


def validate_strategy_config(strategy_config: Dict) -> tuple[bool, str]:
    """Validate strategy configuration before execution"""
    has_rules = any([
        strategy_config.get("technical_indicators"),
        strategy_config.get("sentiment_rules"),
        strategy_config.get("trade_view_rules"),
        strategy_config.get("stock_score_rule"),
        strategy_config.get("market_cap_rule"),
        strategy_config.get("sector_rule"),
        strategy_config.get("volume_rule"),
        strategy_config.get("price_rule")
    ])

    if not has_rules:
        return False, "Please define at least one strategy rule before executing."

    # Validate technical indicators
    for indicator in strategy_config.get("technical_indicators", []):
        if indicator["operator"] == "between" and indicator.get("value2", 0) <= indicator.get("value", 0):
            return False, f"For 'between' operator in {TECHNICAL_INDICATORS[indicator['type']]}, Value 2 must be greater than Value 1."

    # Validate price rule
    price_rule = strategy_config.get("price_rule")
    if price_rule and price_rule.get("max_price", 0) > 0 and price_rule.get("min_price", 0) >= price_rule["max_price"]:
        return False, "Maximum price must be greater than minimum price."

    # Validate volume rule
    volume_rule = strategy_config.get("volume_rule")
    if volume_rule and volume_rule.get("max_volume", 0) > 0 and volume_rule.get("min_volume", 0) >= volume_rule[
        "max_volume"]:
        return False, "Maximum volume must be greater than minimum volume."

    return True, ""


def render_technical_indicators_section():
    """Render technical indicators configuration section with improved UX"""
    st.subheader("📊 Technical Indicators")
    st.markdown("*Add technical analysis rules to filter stocks based on price patterns and momentum.*")

    if "technical_indicators" not in st.session_state:
        st.session_state.technical_indicators = []

    # Add new technical indicator
    with st.expander("➕ Add Technical Indicator", expanded=len(st.session_state.technical_indicators) == 0):
        col1, col2, col3 = st.columns(3)

        with col1:
            indicator_type = st.selectbox(
                "Indicator Type:",
                options=list(TECHNICAL_INDICATORS.keys()),
                format_func=lambda x: TECHNICAL_INDICATORS[x],
                key="new_tech_type",
                help="Choose the technical indicator to apply"
            )

        with col2:
            period = st.number_input(
                "Period:",
                min_value=1,
                max_value=200,
                value=14,
                key="new_tech_period",
                help="Number of periods for calculation"
            )

        with col3:
            operator = st.selectbox(
                "Operator:",
                options=list(OPERATORS.keys()),
                format_func=lambda x: OPERATORS[x],
                key="new_tech_operator",
                help="Comparison operator for the condition"
            )

        col1, col2 = st.columns(2)
        with col1:
            value = st.number_input(
                "Value:",
                value=0.0,
                step=0.01,
                format="%.2f",
                key="new_tech_value",
                help="Target value for comparison"
            )

        with col2:
            if operator == "between":
                value2 = st.number_input(
                    "Value 2 (upper bound):",
                    value=round(value + 1.0, 2),
                    step=0.01,
                    format="%.2f",
                    key="new_tech_value2",
                    help="Upper bound for 'between' operator"
                )
            else:
                value2 = 0.0

        # Additional parameters for specific indicators
        additional_params = {}
        if indicator_type == "macd":
            st.markdown("**MACD Parameters**")
            col1, col2 = st.columns(2)
            with col1:
                additional_params["slow_period"] = st.number_input(
                    "Slow Period:",
                    min_value=1,
                    value=26,
                    key="new_macd_slow",
                    help="Slow EMA period for MACD calculation"
                )
            with col2:
                additional_params["signal_period"] = st.number_input(
                    "Signal Period:",
                    min_value=1,
                    value=9,
                    key="new_macd_signal",
                    help="Signal line EMA period"
                )

        elif indicator_type == "bollinger_width":
            additional_params["std_dev_multiplier"] = st.number_input(
                "Std Dev Multiplier:",
                min_value=0.1,
                value=2.0,
                step=0.1,
                key="new_bb_std",
                help="Standard deviation multiplier for Bollinger Bands"
            )

        if st.button("Add Technical Indicator", key="add_tech_indicator", type="primary"):
            # Validation
            if operator == "between" and value2 <= value:
                st.error("Value 2 must be greater than Value 1 for 'between' operator.")
            else:
                new_indicator = {
                    "type": indicator_type,
                    "period": period,
                    "operator": operator,
                    "value": round(value, 2),
                    "value2": round(value2, 2),
                    **additional_params
                }
                st.session_state.technical_indicators.append(new_indicator)
                st.success(f"✅ Added {TECHNICAL_INDICATORS[indicator_type]} indicator")
                st.rerun()

    # Display existing indicators with improved formatting
    if st.session_state.technical_indicators:
        st.markdown("**Current Technical Indicators:**")
        for i, indicator in enumerate(st.session_state.technical_indicators):
            with st.container():
                col1, col2 = st.columns([5, 1])
                with col1:
                    desc = f"**{TECHNICAL_INDICATORS[indicator['type']]}** (Period: {indicator['period']}) {OPERATORS[indicator['operator']]} {indicator['value']:.2f}"
                    if indicator['operator'] == 'between':
                        desc += f" and {indicator['value2']:.2f}"

                    # Add additional parameter info
                    if indicator['type'] == 'macd':
                        desc += f" | Slow: {indicator.get('slow_period', 26)}, Signal: {indicator.get('signal_period', 9)}"
                    elif indicator['type'] == 'bollinger_width':
                        desc += f" | Std Dev: {indicator.get('std_dev_multiplier', 2.0):.1f}"

                    st.markdown(desc)

                with col2:
                    if st.button("🗑️", key=f"remove_tech_{i}", help="Remove this indicator"):
                        st.session_state.technical_indicators.pop(i)
                        st.success("Technical indicator removed")
                        st.rerun()

                st.divider()


def render_sentiment_rules_section():
    """Render sentiment rules configuration section"""
    st.subheader("😊 Sentiment Analysis Rules")
    st.markdown("*Filter stocks based on market sentiment scores and labels.*")

    if "sentiment_rules" not in st.session_state:
        st.session_state.sentiment_rules = []

    with st.expander("➕ Add Sentiment Rule", expanded=len(st.session_state.sentiment_rules) == 0):
        col1, col2, col3 = st.columns(3)

        with col1:
            score_operator = st.selectbox(
                "Score Operator:",
                options=["gt", "gte", "lt", "lte"],
                format_func=lambda x: OPERATORS[x],
                key="new_sentiment_operator",
                help="Comparison operator for sentiment score"
            )

        with col2:
            score_value = st.number_input(
                "Score Value:",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                format="%.2f",
                key="new_sentiment_value",
                help="Sentiment score threshold (0.0 to 1.0)"
            )

        with col3:
            label_filter = st.selectbox(
                "Label Filter:",
                options=[""] + SENTIMENT_LABELS,
                format_func=lambda x: "Any" if x == "" else x.title(),
                key="new_sentiment_label",
                help="Filter by specific sentiment label"
            )

        if st.button("Add Sentiment Rule", key="add_sentiment_rule", type="primary"):
            new_rule = {
                "score_operator": score_operator,
                "score_value": round(score_value, 2),
                "label_filter": label_filter
            }
            st.session_state.sentiment_rules.append(new_rule)
            st.success("✅ Added sentiment rule")
            st.rerun()

    # Display existing rules
    if st.session_state.sentiment_rules:
        st.markdown("**Current Sentiment Rules:**")
        for i, rule in enumerate(st.session_state.sentiment_rules):
            with st.container():
                col1, col2 = st.columns([5, 1])
                with col1:
                    desc = f"**Sentiment Score** {OPERATORS[rule['score_operator']]} {rule['score_value']:.2f}"
                    if rule['label_filter']:
                        desc += f" **AND** Label = {rule['label_filter'].title()}"
                    st.markdown(desc)

                with col2:
                    if st.button("🗑️", key=f"remove_sentiment_{i}", help="Remove this rule"):
                        st.session_state.sentiment_rules.pop(i)
                        st.success("Sentiment rule removed")
                        st.rerun()

                st.divider()


def render_trade_view_rules_section():
    """Render trade view rules configuration section"""
    st.subheader("🐂 Trade View Rules")
    st.markdown("*Filter stocks based on trading view recommendations and performance.*")

    if "trade_view_rules" not in st.session_state:
        st.session_state.trade_view_rules = []

    with st.expander("➕ Add Trade View Rule", expanded=len(st.session_state.trade_view_rules) == 0):
        col1, col2, col3 = st.columns(3)

        with col1:
            profit_percentage = st.number_input(
                "Min Profit %:",
                value=0.0,
                step=0.01,
                format="%.2f",
                key="new_tv_profit",
                help="Minimum profit percentage required"
            )

        with col2:
            status = st.selectbox(
                "Status Filter:",
                options=[""] + TRADE_STATUSES,
                format_func=lambda x: "Any" if x == "" else x.title(),
                key="new_tv_status",
                help="Filter by trade status"
            )

        with col3:
            days_active = st.number_input(
                "Min Days Active:",
                min_value=0,
                value=0,
                key="new_tv_days",
                help="Minimum number of days trade has been active"
            )

        if st.button("Add Trade View Rule", key="add_tv_rule", type="primary"):
            new_rule = {
                "profit_percentage": round(profit_percentage, 2),
                "status": status,
                "days_active": days_active
            }
            st.session_state.trade_view_rules.append(new_rule)
            st.success("✅ Added trade view rule")
            st.rerun()

    # Display existing rules
    if st.session_state.trade_view_rules:
        st.markdown("**Current Trade View Rules:**")
        for i, rule in enumerate(st.session_state.trade_view_rules):
            with st.container():
                col1, col2 = st.columns([5, 1])
                with col1:
                    desc = f"**Min Profit:** {rule['profit_percentage']:.2f}%"
                    if rule['status']:
                        desc += f", **Status:** {rule['status'].title()}"
                    if rule['days_active'] > 0:
                        desc += f", **Min Days:** {rule['days_active']}"
                    st.markdown(desc)

                with col2:
                    if st.button("🗑️", key=f"remove_tv_{i}", help="Remove this rule"):
                        st.session_state.trade_view_rules.pop(i)
                        st.success("Trade view rule removed")
                        st.rerun()

                st.divider()


def render_basic_filters_section():
    """Render basic filters section with improved layout"""
    st.subheader("🔍 Basic Filters")
    st.markdown("*Apply fundamental filters based on stock metrics and characteristics.*")

    # Stock Score and Market Cap filters
    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            st.markdown("**📊 Stock Score Filter**")
            enable_score = st.checkbox("Enable Stock Score Filter", key="enable_score_filter")
            if enable_score:
                min_score = st.slider(
                    "Minimum Score:",
                    0.0,
                    10.0,
                    5.0,
                    0.1,
                    format="%.1f",
                    key="min_score",
                    help="Minimum stock score required"
                )
                min_score_count = st.number_input(
                    "Minimum Score Count:",
                    min_value=0,
                    value=0,
                    key="min_score_count",
                    help="Minimum number of score evaluations"
                )
                recommendation = st.selectbox(
                    "Recommendation:",
                    options=[""] + RECOMMENDATIONS,
                    format_func=lambda x: "Any" if x == "" else x,
                    key="recommendation_filter",
                    help="Filter by analyst recommendation"
                )

                st.session_state.stock_score_rule = {
                    "min_score": round(min_score, 1),
                    "min_score_count": min_score_count,
                    "recommendation": recommendation
                }
            else:
                st.session_state.stock_score_rule = None

    with col2:
        with st.container():
            st.markdown("**💰 Market Cap Filter**")
            enable_mcap = st.checkbox("Enable Market Cap Filter", key="enable_mcap_filter")
            if enable_mcap:
                mcap_category = st.selectbox(
                    "Market Cap Category:",
                    options=MARKET_CAP_CATEGORIES,
                    format_func=lambda x: f"{x.title()} Cap",
                    key="mcap_category",
                    help="Filter by market capitalization category"
                )
                st.session_state.market_cap_rule = {"category": mcap_category}
            else:
                st.session_state.market_cap_rule = None

    st.markdown("---")

    # Price and Volume filters
    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            st.markdown("**💵 Price Filter**")
            enable_price = st.checkbox("Enable Price Filter", key="enable_price_filter")
            if enable_price:
                price_range = st.slider(
                    "Price Range (₹):",
                    min_value=0.0,
                    max_value=10000.0,
                    value=(0.0, 1000.0),
                    step=1.0,
                    format="%.2f",
                    key="price_range",
                    help="Set minimum and maximum price range"
                )
                st.session_state.price_rule = {
                    "min_price": round(price_range[0], 2),
                    "max_price": round(price_range[1], 2) if price_range[1] > price_range[0] else 0
                }
            else:
                st.session_state.price_rule = None

    with col2:
        with st.container():
            st.markdown("**📊 Volume Filter**")
            enable_volume = st.checkbox("Enable Volume Filter", key="enable_volume_filter")
            if enable_volume:
                min_volume = st.number_input(
                    "Minimum Volume:",
                    min_value=0,
                    value=10000,
                    step=1000,
                    key="min_volume",
                    help="Minimum trading volume required"
                )
                max_volume = st.number_input(
                    "Maximum Volume (0 = No limit):",
                    min_value=0,
                    value=0,
                    step=1000,
                    key="max_volume",
                    help="Maximum trading volume (0 for no limit)"
                )
                st.session_state.volume_rule = {
                    "min_volume": min_volume,
                    "max_volume": max_volume if max_volume > min_volume else 0
                }
            else:
                st.session_state.volume_rule = None

    st.markdown("---")

    # Sector Filter
    with st.container():
        st.markdown("**🏭 Sector Filter**")
        enable_sector = st.checkbox("Enable Sector Filter", key="enable_sector_filter")
        if enable_sector:
            col1, col2 = st.columns(2)
            with col1:
                include_sectors = st.multiselect(
                    "Include Sectors:",
                    options=SECTORS,
                    key="include_sectors",
                    help="Select sectors to include in results"
                )
            with col2:
                exclude_sectors = st.multiselect(
                    "Exclude Sectors:",
                    options=SECTORS,
                    key="exclude_sectors",
                    help="Select sectors to exclude from results"
                )

            # Validation
            if include_sectors and exclude_sectors:
                overlap = set(include_sectors) & set(exclude_sectors)
                if overlap:
                    st.warning(f"⚠️ Sectors in both include and exclude lists: {', '.join(overlap)}")

            st.session_state.sector_rule = {
                "include": include_sectors,
                "exclude": exclude_sectors
            }
        else:
            st.session_state.sector_rule = None


def prepare_results_dataframe(results: List[Dict]) -> pd.DataFrame:
    """Prepare results dataframe for display with optimized structure"""
    if not results:
        return pd.DataFrame()

    try:
        processed_data = []
        for result in results:
            stock = result.get("stock", {})

            # Handle week52 and today data safely
            week52 = stock.get("week52", {})
            today = stock.get("today", {})

            processed_stock = {
                "Symbol": stock.get("symbol", "N/A"),
                "Current Price (₹)": round(float(stock.get("last_trade_price", 0)), 2),
                "Volume": stock.get("volume", 0),
                "Market Cap": stock.get("market_cap", "N/A").title(),
                "52W High (₹)": round(float(week52.get("high", 0)), 2),
                "52W Low (₹)": round(float(week52.get("low", 0)), 2),
                "Today High (₹)": round(float(today.get("high", 0)), 2),
                "Today Low (₹)": round(float(today.get("low", 0)), 2)
            }
            processed_data.append(processed_stock)

        df = pd.DataFrame(processed_data)

        # Optimize data types
        numeric_columns = [
            "Current Price (₹)", "Volume", "52W High (₹)", "52W Low (₹)",
            "Today High (₹)", "Today Low (₹)"
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        return df

    except Exception as e:
        logger.error(f"Error preparing results dataframe: {e}")
        st.error(f"❌ Error processing results: {str(e)}")
        return pd.DataFrame()


def render_strategy_results(results: List[Dict]):
    """Render strategy execution results with enhanced visualizations"""
    if not results:
        st.warning("📭 No stocks matched your strategy criteria. Try adjusting your filters.")
        return

    # Success message
    st.success(f"🎯 Found {len(results)} stocks matching your strategy!")

    # Prepare dataframe
    df = prepare_results_dataframe(results)

    if df.empty:
        st.error("❌ Error processing results data")
        return

    # Enhanced summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🎯 Total Matches", len(results))

    with col2:
        avg_price = df["Current Price (₹)"].mean()
        st.metric("💰 Avg Price", f"₹{avg_price:,.2f}")

    with col3:
        total_volume = df["Volume"].sum()
        st.metric("📊 Total Volume", f"{total_volume:,}")

    # Visualization tabs
    tab1, tab2 = st.tabs(["📊 Results Table", "📈 Analytics"])

    with tab1:
        st.subheader("📋 Strategy Results")

        # Enhanced sorting and filtering
        col1, col2, col3 = st.columns(3)

        with col1:
            sort_by = st.selectbox(
                "Sort by:",
                options=["Current Price (₹)", "Volume", "52W High (₹)", "52W Low (₹)"],
                index=0,
                key="results_sort"
            )

        with col2:
            sort_order = st.selectbox(
                "Sort Order:",
                options=["Descending", "Ascending"],
                index=0,
                key="sort_order"
            )

        with col3:
            show_count = st.selectbox(
                "Show Results:",
                options=["All", "Top 10", "Top 25", "Top 50"],
                index=1,
                key="show_count"
            )

        # Apply sorting
        ascending = sort_order == "Ascending"
        sorted_df = df.sort_values(sort_by, ascending=ascending)

        # Apply count filter
        if show_count != "All":
            count = int(show_count.split()[-1])
            sorted_df = sorted_df.head(count)

        # Display results table
        st.dataframe(
            sorted_df,
            use_container_width=True,
            height=500,
            hide_index=True,
            column_config={
                "Current Price (₹)": st.column_config.NumberColumn(
                    "Current Price (₹)",
                    format="₹%.2f"
                ),
                "52W High (₹)": st.column_config.NumberColumn(
                    "52W High (₹)",
                    format="₹%.2f"
                ),
                "52W Low (₹)": st.column_config.NumberColumn(
                    "52W Low (₹)",
                    format="₹%.2f"
                ),
                "Today High (₹)": st.column_config.NumberColumn(
                    "Today High (₹)",
                    format="₹%.2f"
                ),
                "Today Low (₹)": st.column_config.NumberColumn(
                    "Today Low (₹)",
                    format="₹%.2f"
                )
            }
        )

        # Download functionality
        csv = sorted_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"strategy_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with tab2:
        st.subheader("📈 Strategy Analytics")

        # Create visualizations
        try:
            col1, col2 = st.columns(2)

            with col1:
                # Market Cap Distribution
                mcap_counts = df["Market Cap"].value_counts()
                fig_mcap = px.pie(
                    values=mcap_counts.values,
                    names=mcap_counts.index,
                    title="💰 Market Cap Distribution",
                    template="plotly_white"
                )
                fig_mcap.update_layout(height=400)
                st.plotly_chart(fig_mcap, use_container_width=True)

            with col2:
                # Price Distribution
                fig_price = px.histogram(
                    df,
                    x="Current Price (₹)",
                    nbins=20,
                    title="💵 Price Distribution",
                    labels={"count": "Number of Stocks"},
                    template="plotly_white"
                )
                fig_price.update_layout(height=400)
                st.plotly_chart(fig_price, use_container_width=True)

            # Price vs Volume scatter
            fig_scatter = px.scatter(
                df,
                x="Current Price (₹)",
                y="Volume",
                size="Volume",
                color="Market Cap",
                hover_data=["Symbol"],
                title="💎 Price vs Volume",
                template="plotly_white"
            )
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error creating visualizations: {str(e)}")
            logger.error(f"Visualization error: {e}")


def render_strategy_summary():
    """Render current strategy summary"""
    strategy_parts = []

    # Technical indicators
    if st.session_state.get("technical_indicators"):
        count = len(st.session_state.technical_indicators)
        strategy_parts.append(f"📊 {count} Technical Indicator{'s' if count != 1 else ''}")

    # Sentiment rules
    if st.session_state.get("sentiment_rules"):
        count = len(st.session_state.sentiment_rules)
        strategy_parts.append(f"😊 {count} Sentiment Rule{'s' if count != 1 else ''}")

    # Trade view rules
    if st.session_state.get("trade_view_rules"):
        count = len(st.session_state.trade_view_rules)
        strategy_parts.append(f"🐂 {count} Trade View Rule{'s' if count != 1 else ''}")

    # Basic filters
    basic_filters = []
    if st.session_state.get("stock_score_rule"):
        basic_filters.append("Stock Score")
    if st.session_state.get("market_cap_rule"):
        basic_filters.append("Market Cap")
    if st.session_state.get("price_rule"):
        basic_filters.append("Price")
    if st.session_state.get("volume_rule"):
        basic_filters.append("Volume")
    if st.session_state.get("sector_rule"):
        basic_filters.append("Sector")

    if basic_filters:
        strategy_parts.append(
            f"🔍 {len(basic_filters)} Basic Filter{'s' if len(basic_filters) != 1 else ''} ({', '.join(basic_filters)})")

    if strategy_parts:
        st.info("🎯 **Current Strategy**: " + " | ".join(strategy_parts))
    else:
        st.warning("⚠️ No strategy rules defined. Add some rules above to create your strategy.")


def render_strategy_builder():
    """Main strategy builder interface with improved UX"""
    st.title("🎯 Custom Stock Strategy Builder")
    st.markdown("""
    Create and execute powerful stock screening strategies using technical indicators, sentiment analysis, and fundamental filters. 
    Build your strategy step by step and discover stocks that match your criteria.
    """)

    # Strategy summary
    render_strategy_summary()
    st.markdown("---")

    # Strategy configuration sections
    render_technical_indicators_section()
    st.markdown("---")

    render_sentiment_rules_section()
    st.markdown("---")

    render_trade_view_rules_section()
    st.markdown("---")

    render_basic_filters_section()
    st.markdown("---")

    # Strategy execution section
    st.subheader("🚀 Execute Strategy")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        limit = st.number_input(
            "Max Results:",
            min_value=1,
            max_value=100,
            value=20,
            key="strategy_limit",
            help="Maximum number of results to return"
        )

    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        execute_button = st.button(
            "🚀 Run Strategy",
            key="execute_strategy",
            type="primary",
            use_container_width=True,
            help="Execute your strategy to find matching stocks"
        )

    with col3:
        st.write("")  # Spacer
        st.write("")  # Spacer
        if st.button(
                "🗑️ Clear All",
                key="clear_strategy",
                use_container_width=True,
                help="Clear all strategy rules and start over"
        ):
            # Clear all session state
            keys_to_clear = [
                "technical_indicators", "sentiment_rules", "trade_view_rules",
                "stock_score_rule", "market_cap_rule", "sector_rule",
                "volume_rule", "price_rule", "strategy_results"
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("🧹 Strategy cleared! You can now build a new strategy.")
            st.rerun()

    with col4:
        st.write("")  # Spacer
        st.write("")  # Spacer
        if st.button(
                "📋 View Payload",
                key="view_payload",
                use_container_width=True,
                help="View the API payload that will be sent"
        ):
            strategy_config = {
                "technical_indicators": st.session_state.get("technical_indicators", []),
                "sentiment_rules": st.session_state.get("sentiment_rules", []),
                "trade_view_rules": st.session_state.get("trade_view_rules", []),
                "stock_score_rule": st.session_state.get("stock_score_rule"),
                "market_cap_rule": st.session_state.get("market_cap_rule"),
                "sector_rule": st.session_state.get("sector_rule"),
                "volume_rule": st.session_state.get("volume_rule"),
                "price_rule": st.session_state.get("price_rule"),
                "limit": limit
            }

            payload = create_strategy_payload(strategy_config)
            st.json(payload)

    # Execute strategy
    if execute_button:
        # Build strategy configuration
        strategy_config = {
            "technical_indicators": st.session_state.get("technical_indicators", []),
            "sentiment_rules": st.session_state.get("sentiment_rules", []),
            "trade_view_rules": st.session_state.get("trade_view_rules", []),
            "stock_score_rule": st.session_state.get("stock_score_rule"),
            "market_cap_rule": st.session_state.get("market_cap_rule"),
            "sector_rule": st.session_state.get("sector_rule"),
            "volume_rule": st.session_state.get("volume_rule"),
            "price_rule": st.session_state.get("price_rule"),
            "limit": limit
        }

        # Validate strategy configuration
        is_valid, error_message = validate_strategy_config(strategy_config)

        if not is_valid:
            st.error(f"❌ {error_message}")
        else:
            # Create API payload
            payload = create_strategy_payload(strategy_config)

            # Execute strategy with progress indication
            with st.spinner("🔄 Executing your strategy... Please wait."):
                results = execute_strategy(payload)

            if results is not None:
                st.session_state.strategy_results = results
                st.markdown("---")
                render_strategy_results(results)

    # Show previous results if available
    elif st.session_state.get("strategy_results"):
        st.markdown("---")
        st.subheader("📊 Previous Results")
        st.info("These are the results from your last strategy execution. Run a new strategy to update.")
        render_strategy_results(st.session_state.strategy_results)

        # Clear previous results button
        if st.button("🗑️ Clear Previous Results", key="clear_previous_results"):
            del st.session_state.strategy_results
            st.success("Previous results cleared!")
            st.rerun()