import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Learning module constants
LEARNING_PHASES = {
    "Beginner": {"color": "#4CAF50", "icon": "🌱"},
    "Intermediate": {"color": "#FF9800", "icon": "📈"},
    "Advanced": {"color": "#2196F3", "icon": "🎯"},
    "Expert": {"color": "#9C27B0", "icon": "🏆"}
}


def initialize_learning_progress():
    """Initialize learning progress tracking in session state"""
    if 'learning_progress' not in st.session_state:
        st.session_state.learning_progress = {
            "completed_topics": set(),
            "bookmarks": set(),
            "current_phase": "Beginner",
            "study_time": 0,
            "quiz_scores": {}
        }


def update_progress(topic_id: str):
    """Update learning progress for a topic"""
    if 'learning_progress' not in st.session_state:
        initialize_learning_progress()
    st.session_state.learning_progress["completed_topics"].add(topic_id)


def render_progress_dashboard():
    """Render learning progress dashboard"""
    if 'learning_progress' not in st.session_state:
        initialize_learning_progress()

    progress = st.session_state.learning_progress

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Current Phase", progress["current_phase"])
    with col2:
        completed_count = len(progress["completed_topics"])
        st.metric("Topics Completed", completed_count)
    with col3:
        bookmarks_count = len(progress["bookmarks"])
        st.metric("Bookmarked", bookmarks_count)
    with col4:
        study_hours = progress["study_time"] / 60  # Convert minutes to hours
        st.metric("Study Time", f"{study_hours:.1f}h")


def render_topic_content(topic_id: str, title: str, content: dict, phase: str):
    """Render individual topic content with interactive elements"""
    try:
        phase_info = LEARNING_PHASES.get(phase, {"color": "#666", "icon": "📖"})

        # Topic header
        st.markdown(f"""
        <div style="border-left: 4px solid {phase_info['color']}; padding-left: 20px; margin-bottom: 20px;">
            <h3>{phase_info['icon']} {title}</h3>
            <span style="color: {phase_info['color']}; font-weight: bold;">{phase} Level</span>
        </div>
        """, unsafe_allow_html=True)

        # Topic controls
        col1, col2, col3 = st.columns([6, 1, 1])

        # Initialize learning progress if not exists
        if 'learning_progress' not in st.session_state:
            initialize_learning_progress()

        with col2:
            if st.button("📖", key=f"bookmark_{topic_id}", help="Bookmark this topic"):
                if topic_id in st.session_state.learning_progress["bookmarks"]:
                    st.session_state.learning_progress["bookmarks"].remove(topic_id)
                    st.success("Bookmark removed!")
                else:
                    st.session_state.learning_progress["bookmarks"].add(topic_id)
                    st.success("Topic bookmarked!")

        with col3:
            if st.button("✅", key=f"complete_{topic_id}", help="Mark as completed"):
                update_progress(topic_id)
                st.success("Topic completed!")

        # Content sections
        for section_title, section_content in content.items():
            if section_title == "interactive_demo":
                continue  # Handle separately

            st.markdown(f"### {section_title}")

            if isinstance(section_content, list):
                for item in section_content:
                    st.markdown(f"• {item}")
            else:
                st.markdown(section_content)

            st.markdown("---")

        # Interactive demo if available
        if "interactive_demo" in content:
            st.markdown("### 🎮 Interactive Demo")
            try:
                content["interactive_demo"]()
            except Exception as e:
                st.error(f"Error loading interactive demo: {str(e)}")
                logger.error(f"Interactive demo error for {topic_id}: {e}")

    except Exception as e:
        st.error(f"Error rendering topic content: {str(e)}")
        logger.error(f"Topic content error for {topic_id}: {e}")


def create_stock_price_demo():
    """Create interactive stock price demonstration"""
    try:
        st.markdown("**Understanding Stock Price Movements**")

        # Generate sample data
        days = 30
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')

        # Simulate stock price with user controls
        col1, col2 = st.columns(2)
        with col1:
            initial_price = st.slider("Starting Price (₹)", 100, 1000, 500, key="demo_price")
            volatility = st.slider("Volatility", 0.01, 0.1, 0.03, key="demo_volatility")

        with col2:
            trend = st.selectbox("Market Trend", ["Bullish", "Bearish", "Sideways"], key="demo_trend")
            news_impact = st.slider("News Impact", -10, 10, 0, key="demo_news")

        # Generate price data based on parameters
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(0, volatility, days)

        # Apply trend
        if trend == "Bullish":
            returns += 0.01
        elif trend == "Bearish":
            returns -= 0.01

        # Apply news impact on random day
        news_day = np.random.randint(10, 20)
        returns[news_day] += news_impact / 100

        prices = [initial_price]
        for return_rate in returns[1:]:
            prices.append(prices[-1] * (1 + return_rate))

        # Create interactive chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines+markers',
            name='Stock Price',
            line=dict(color='#2196F3', width=2),
            marker=dict(size=4)
        ))

        # Highlight news impact day
        fig.add_vline(
            x=dates[news_day],
            line_dash="dash",
            line_color="red",
            annotation_text="News Impact"
        )

        fig.update_layout(
            title="Stock Price Movement Simulation",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            template="plotly_white",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Analysis
        total_return = (prices[-1] - prices[0]) / prices[0] * 100
        max_price = max(prices)
        min_price = min(prices)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Return", f"{total_return:.2f}%")
        with col2:
            st.metric("Highest Price", f"₹{max_price:.2f}")
        with col3:
            st.metric("Lowest Price", f"₹{min_price:.2f}")

    except Exception as e:
        st.error(f"Error in stock price demo: {str(e)}")
        logger.error(f"Stock price demo error: {e}")


def create_portfolio_demo():
    """Create portfolio diversification demonstration"""
    try:
        st.markdown("**Portfolio Diversification Impact**")

        # Portfolio configuration
        col1, col2 = st.columns(2)
        with col1:
            tech_weight = st.slider("Technology Sector (%)", 0, 100, 40)
            remaining_after_tech = 100 - tech_weight
            finance_weight = st.slider("Finance Sector (%)", 0, remaining_after_tech, min(30, remaining_after_tech))

        with col2:
            remaining_after_finance = 100 - tech_weight - finance_weight
            healthcare_weight = st.slider("Healthcare Sector (%)", 0, remaining_after_finance,
                                          min(20, remaining_after_finance))
            other_weight = 100 - tech_weight - finance_weight - healthcare_weight
            st.metric("Other Sectors (%)", other_weight)

        # Simulate sector returns
        sectors = ['Technology', 'Finance', 'Healthcare', 'Others']
        weights = [tech_weight / 100, finance_weight / 100, healthcare_weight / 100, other_weight / 100]

        # Random sector performance
        np.random.seed(42)
        sector_returns = np.random.normal([0.12, 0.08, 0.10, 0.06], [0.15, 0.12, 0.10, 0.08])

        # Calculate portfolio return
        portfolio_return = sum(w * r for w, r in zip(weights, sector_returns))

        # Visualization
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "bar"}]],
            subplot_titles=["Portfolio Allocation", "Sector Returns"]
        )

        # Pie chart for allocation
        fig.add_trace(
            go.Pie(labels=sectors, values=weights, name="Allocation"),
            row=1, col=1
        )

        # Bar chart for returns
        fig.add_trace(
            go.Bar(x=sectors, y=sector_returns, name="Returns",
                   marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']),
            row=1, col=2
        )

        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"Your diversified portfolio return: **{portfolio_return:.2%}**")

    except Exception as e:
        st.error(f"Error in portfolio demo: {str(e)}")
        logger.error(f"Portfolio demo error: {e}")


def create_risk_reward_demo():
    """Create risk-reward analysis demonstration"""
    try:
        st.markdown("**Risk vs Reward Analysis**")

        # Investment options
        investments = {
            "Government Bonds": {"return": 0.06, "risk": 0.02},
            "Large Cap Stocks": {"return": 0.12, "risk": 0.15},
            "Small Cap Stocks": {"return": 0.18, "risk": 0.25},
            "Crypto": {"return": 0.25, "risk": 0.40}
        }

        # User risk tolerance
        risk_tolerance = st.slider("Your Risk Tolerance (1-10)", 1, 10, 5)
        investment_amount = st.number_input("Investment Amount (₹)", 10000, 1000000, 100000)

        # Calculate recommended allocation
        risk_factor = risk_tolerance / 10

        # Create risk-return scatter plot
        fig = go.Figure()

        colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']

        for i, (name, data) in enumerate(investments.items()):
            fig.add_trace(go.Scatter(
                x=[data["risk"]],
                y=[data["return"]],
                mode='markers+text',
                name=name,
                text=[name],
                textposition="top center",
                marker=dict(size=15, color=colors[i])
            ))

        # Add user risk tolerance line
        fig.add_hline(
            y=risk_factor * 0.3,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Your Risk Level: {risk_tolerance}/10"
        )

        fig.update_layout(
            title="Risk vs Return Analysis",
            xaxis_title="Risk (Volatility)",
            yaxis_title="Expected Return",
            template="plotly_white",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Recommendations
        st.markdown("### 💡 Personalized Recommendations")

        for name, data in investments.items():
            risk_score = data["risk"] * 10
            if abs(risk_score - risk_tolerance) <= 2:
                potential_return = investment_amount * data["return"]
                st.success(
                    f"✅ **{name}**: Matches your risk profile. Potential annual return: ₹{potential_return:,.0f}")

    except Exception as e:
        st.error(f"Error in risk reward demo: {str(e)}")
        logger.error(f"Risk reward demo error: {e}")


# Define learning content
LEARNING_CONTENT = {
    "beginner": {
        "what_is_stock_market": {
            "title": "What is the Stock Market?",
            "content": {
                "Basic Definition": """
                The stock market is like a giant marketplace where people buy and sell shares of companies. 
                When you buy a share, you own a tiny piece of that company and become a shareholder.

                Think of it like owning a slice of pizza - if the pizza (company) becomes more valuable, 
                your slice becomes more valuable too!
                """,

                "Key Players": [
                    "**Investors**: People who buy stocks to hold for long periods",
                    "**Traders**: People who buy and sell stocks quickly for profit",
                    "**Companies**: Businesses that sell their shares to raise money",
                    "**Stock Exchanges**: Platforms where buying and selling happens (BSE, NSE)",
                    "**Brokers**: People/firms who help you buy and sell stocks"
                ],

                "Why Companies Go Public": [
                    "Raise money for business expansion",
                    "Pay off existing debts",
                    "Fund research and development",
                    "Allow early investors to cash out",
                    "Increase brand visibility and credibility"
                ],

                "How Stock Prices Work": """
                Stock prices move based on supply and demand:
                - **More buyers than sellers** → Price goes UP 📈
                - **More sellers than buyers** → Price goes DOWN 📉
                - **Equal buyers and sellers** → Price stays STABLE ➡️

                Factors affecting demand include company performance, news, market sentiment, and economic conditions.
                """
            }
        },

        "basic_terminology": {
            "title": "Essential Stock Market Terms",
            "content": {
                "Core Terms": [
                    "**Share/Stock**: A unit of ownership in a company",
                    "**Portfolio**: Collection of all your investments",
                    "**Dividend**: Profit sharing payment from company to shareholders",
                    "**Market Cap**: Total value of all company shares (Price × Total Shares)",
                    "**Ticker Symbol**: Short code identifying a stock (e.g., TCS, RELIANCE)"
                ],

                "Price-Related Terms": [
                    "**Current Price/LTP**: Last traded price of the stock",
                    "**Bid Price**: Highest price a buyer is willing to pay",
                    "**Ask Price**: Lowest price a seller is willing to accept",
                    "**Spread**: Difference between bid and ask price",
                    "**Volume**: Number of shares traded in a period"
                ],

                "Market Conditions": [
                    "**Bull Market**: Generally rising prices (optimistic)",
                    "**Bear Market**: Generally falling prices (pessimistic)",
                    "**Volatility**: How much prices fluctuate",
                    "**Liquidity**: How easily you can buy/sell without affecting price",
                    "**Market Hours**: When stock exchanges are open for trading"
                ],

                "Order Types": [
                    "**Market Order**: Buy/sell immediately at current price",
                    "**Limit Order**: Buy/sell only at specific price or better",
                    "**Stop Loss**: Automatic sell order to limit losses",
                    "**Good Till Cancelled (GTC)**: Order remains active until executed or cancelled"
                ]
            }
        },

        "how_to_start_investing": {
            "title": "How to Start Investing",
            "content": {
                "Step 1: Financial Preparation": [
                    "Build an emergency fund (3-6 months expenses)",
                    "Pay off high-interest debt (credit cards, personal loans)",
                    "Determine your investment budget (money you won't need for 5+ years)",
                    "Understand your risk tolerance and investment goals"
                ],

                "Step 2: Choose a Broker": [
                    "**Full-Service Brokers**: Provide advice but charge higher fees (ICICI Direct, HDFC Securities)",
                    "**Discount Brokers**: Lower fees, minimal advice (Zerodha, Upstox, Groww)",
                    "Compare brokerage charges, account maintenance fees, and trading platforms",
                    "Check if they offer research reports and educational resources"
                ],

                "Step 3: Open Demat & Trading Account": [
                    "Demat Account: Holds your shares in electronic format",
                    "Trading Account: Used to buy and sell shares",
                    "Required documents: PAN card, Aadhar card, bank account details, passport-size photos",
                    "Complete KYC (Know Your Customer) process online or offline"
                ],

                "Step 4: Start Small": [
                    "Begin with blue-chip stocks or index funds",
                    "Invest only what you can afford to lose initially",
                    "Start with paper trading or virtual trading to practice",
                    "Gradually increase investment as you gain knowledge and confidence"
                ],

                "Common Beginner Mistakes to Avoid": [
                    "Don't put all money in one stock (lack of diversification)",
                    "Don't try to time the market perfectly",
                    "Don't invest based on tips from friends or social media",
                    "Don't panic sell during market downturns",
                    "Don't invest borrowed money or emergency funds"
                ]
            },
            "interactive_demo": create_stock_price_demo
        }
    },

    "intermediate": {
        "fundamental_analysis": {
            "title": "Fundamental Analysis",
            "content": {
                "What is Fundamental Analysis?": """
                Fundamental analysis involves evaluating a company's intrinsic value by examining:
                - Financial statements (Income Statement, Balance Sheet, Cash Flow)
                - Management quality and corporate governance
                - Industry position and competitive advantages
                - Economic and market conditions

                The goal is to determine if a stock is overvalued or undervalued.
                """,

                "Key Financial Ratios": [
                    "**P/E Ratio (Price-to-Earnings)**: Stock price ÷ Earnings per share. Lower P/E may indicate undervaluation",
                    "**P/B Ratio (Price-to-Book)**: Stock price ÷ Book value per share. Shows market premium over book value",
                    "**ROE (Return on Equity)**: Net income ÷ Shareholders' equity. Measures profitability efficiency",
                    "**ROA (Return on Assets)**: Net income ÷ Total assets. Shows how efficiently assets generate profit",
                    "**Current Ratio**: Current assets ÷ Current liabilities. Measures short-term financial health",
                    "**Debt-to-Equity**: Total debt ÷ Total equity. Shows financial leverage and risk"
                ],

                "Analyzing Financial Statements": [
                    "**Income Statement**: Shows revenue, expenses, and profit over a period",
                    "**Balance Sheet**: Shows assets, liabilities, and equity at a specific point",
                    "**Cash Flow Statement**: Shows actual cash generated and used in operations",
                    "Look for consistent revenue growth, improving profit margins, and positive cash flow"
                ],

                "Qualitative Factors": [
                    "**Management Quality**: Track record, vision, transparency in communication",
                    "**Competitive Moat**: Unique advantages that protect against competition",
                    "**Industry Trends**: Growing or declining industry, regulatory changes",
                    "**Business Model**: How the company makes money and its scalability"
                ]
            }
        },

        "technical_analysis": {
            "title": "Technical Analysis Basics",
            "content": {
                "What is Technical Analysis?": """
                Technical analysis studies past price movements and trading volumes to predict future price directions.
                It's based on the belief that all information is already reflected in the stock price.

                Key principles:
                - History tends to repeat itself
                - Price movements follow trends
                - Market sentiment drives prices
                """,

                "Chart Types": [
                    "**Line Chart**: Connects closing prices over time - simple trend view",
                    "**Bar Chart**: Shows open, high, low, close (OHLC) for each period",
                    "**Candlestick Chart**: Similar to bar chart but more visual with colored bodies",
                    "**Volume Chart**: Shows number of shares traded in each period"
                ],

                "Support and Resistance": [
                    "**Support**: Price level where stock tends to stop falling and bounce back",
                    "**Resistance**: Price level where stock tends to stop rising and fall back",
                    "These levels are created by psychological factors and previous trading activity",
                    "Break above resistance or below support can signal significant price moves"
                ],

                "Common Technical Indicators": [
                    "**Moving Averages**: Smooth out price data to identify trends (50-day, 200-day)",
                    "**RSI (Relative Strength Index)**: Measures if stock is overbought (>70) or oversold (<30)",
                    "**MACD**: Shows relationship between two moving averages, signals trend changes",
                    "**Bollinger Bands**: Price channels that expand and contract with volatility"
                ],

                "Chart Patterns": [
                    "**Head and Shoulders**: Reversal pattern signaling trend change",
                    "**Double Top/Bottom**: Price makes two similar highs/lows before reversing",
                    "**Triangles**: Continuation patterns showing price compression before breakout",
                    "**Flags and Pennants**: Short-term continuation patterns after strong moves"
                ]
            }
        },

        "portfolio_management": {
            "title": "Portfolio Management",
            "content": {
                "Asset Allocation Principles": """
                Asset allocation is dividing your investment portfolio among different asset categories:
                - **Stocks (Equity)**: Higher risk, higher potential returns
                - **Bonds (Debt)**: Lower risk, steady income
                - **Cash/Cash Equivalents**: Very low risk, high liquidity
                - **Alternative Investments**: REITs, commodities, international stocks
                """,

                "Diversification Strategies": [
                    "**Sector Diversification**: Spread investments across different industries",
                    "**Geographic Diversification**: Include domestic and international stocks",
                    "**Market Cap Diversification**: Mix of large-cap, mid-cap, and small-cap stocks",
                    "**Style Diversification**: Combine growth and value investing approaches",
                    "**Time Diversification**: Invest regularly over time (Dollar/Rupee Cost Averaging)"
                ],

                "Risk Management": [
                    "**Position Sizing**: Don't put more than 5-10% in any single stock",
                    "**Stop-Loss Orders**: Automatically sell if price falls to predetermined level",
                    "**Regular Portfolio Review**: Rebalance quarterly or when allocation drifts significantly",
                    "**Emergency Fund**: Keep 3-6 months expenses in liquid instruments"
                ],

                "Portfolio Rebalancing": [
                    "**Why Rebalance**: Maintain desired risk level as markets move",
                    "**When to Rebalance**: Set schedule (quarterly/annually) or threshold (5-10% drift)",
                    "**How to Rebalance**: Sell overweight assets, buy underweight assets",
                    "**Tax Considerations**: Use tax-advantaged accounts when possible"
                ]
            },
            "interactive_demo": create_portfolio_demo
        }
    },

    "advanced": {
        "options_derivatives": {
            "title": "Options and Derivatives",
            "content": {
                "Introduction to Options": """
                Options are financial contracts that give you the right (but not obligation) to buy or sell 
                a stock at a specific price within a certain time period.

                **Key Benefits:**
                - Leverage: Control large positions with less capital
                - Hedging: Protect existing positions from losses
                - Income Generation: Collect premium by selling options
                """,

                "Types of Options": [
                    "**Call Option**: Right to BUY a stock at strike price. Profit if price goes UP",
                    "**Put Option**: Right to SELL a stock at strike price. Profit if price goes DOWN",
                    "**American Options**: Can be exercised anytime before expiration",
                    "**European Options**: Can only be exercised on expiration date"
                ],

                "Option Pricing Factors": [
                    "**Underlying Price**: Current stock price vs strike price",
                    "**Time to Expiration**: More time = higher premium (time decay)",
                    "**Volatility**: Higher volatility = higher premium",
                    "**Interest Rates**: Affects present value calculations",
                    "**Dividends**: Expected dividends affect option pricing"
                ],

                "Basic Option Strategies": [
                    "**Covered Call**: Own stock + sell call option (income generation)",
                    "**Protective Put**: Own stock + buy put option (insurance)",
                    "**Bull Call Spread**: Buy call + sell higher strike call (limited profit/loss)",
                    "**Bear Put Spread**: Buy put + sell lower strike put (profit from decline)"
                ],

                "Futures Contracts": [
                    "**Definition**: Obligation to buy/sell asset at predetermined price and date",
                    "**Margin Requirements**: Only need to deposit small percentage of contract value",
                    "**Mark-to-Market**: Profits/losses settled daily",
                    "**Uses**: Hedging, speculation, arbitrage opportunities"
                ]
            }
        },

        "sector_analysis": {
            "title": "Sector and Industry Analysis",
            "content": {
                "Sector Classification": """
                The stock market is divided into sectors based on business activities:
                **Defensive Sectors** (stable during downturns):
                - Utilities, Healthcare, Consumer Staples

                **Cyclical Sectors** (sensitive to economic cycles):
                - Technology, Finance, Real Estate, Automobiles
                """,

                "Sector Rotation Strategy": [
                    "**Economic Expansion**: Technology, Consumer Discretionary perform well",
                    "**Peak Growth**: Materials, Energy sectors outperform",
                    "**Economic Contraction**: Utilities, Healthcare provide stability",
                    "**Recovery Phase**: Financials, Industrials lead the market"
                ],

                "Industry Analysis Framework": [
                    "**Porter's Five Forces**: Competitive rivalry, supplier power, buyer power, threat of substitutes, barriers to entry",
                    "**Industry Life Cycle**: Introduction, growth, maturity, decline stages",
                    "**Regulatory Environment**: Government policies affecting the industry",
                    "**Technology Disruption**: How innovation impacts traditional players"
                ],

                "Key Metrics by Sector": [
                    "**Banking**: NPA ratios, CASA ratio, Net Interest Margin, ROA",
                    "**Technology**: Revenue growth, margin expansion, client concentration",
                    "**Pharmaceuticals**: R&D spending, patent expiry, regulatory approvals",
                    "**Real Estate**: Inventory levels, pre-sales, debt levels, regulatory changes"
                ]
            }
        },

        "quantitative_analysis": {
            "title": "Quantitative Analysis",
            "content": {
                "Statistical Measures": [
                    "**Mean/Average**: Central tendency of returns",
                    "**Standard Deviation**: Measures volatility/risk",
                    "**Correlation**: How two stocks move relative to each other (-1 to +1)",
                    "**Beta**: Stock's sensitivity to market movements (>1 more volatile, <1 less volatile)"
                ],

                "Risk-Adjusted Returns": [
                    "**Sharpe Ratio**: (Return - Risk-free rate) / Standard deviation",
                    "**Alpha**: Excess return over expected return based on beta",
                    "**Maximum Drawdown**: Largest peak-to-trough decline",
                    "**Value at Risk (VaR)**: Maximum expected loss over specific time period"
                ],

                "Backtesting Strategies": [
                    "**Historical Data**: Use past data to test strategy performance",
                    "**Walk-Forward Analysis**: Test strategy on rolling time periods",
                    "**Out-of-Sample Testing**: Reserve recent data for final validation",
                    "**Transaction Costs**: Include brokerage and slippage in calculations"
                ],

                "Algorithmic Trading Concepts": [
                    "**Mean Reversion**: Prices tend to return to average over time",
                    "**Momentum**: Prices that are rising/falling tend to continue",
                    "**Pairs Trading**: Long one stock, short correlated stock",
                    "**Market Making**: Profit from bid-ask spread by providing liquidity"
                ]
            },
            "interactive_demo": create_risk_reward_demo
        }
    },

    "expert": {
        "advanced_strategies": {
            "title": "Advanced Investment Strategies",
            "content": {
                "Value Investing Mastery": [
                    "**Deep Value**: Stocks trading below book value or liquidation value",
                    "**Quality Value**: Profitable companies with sustainable competitive advantages",
                    "**Special Situations**: Spin-offs, bankruptcies, activist investors",
                    "**International Value**: Opportunities in emerging and developed markets"
                ],

                "Growth Investing Excellence": [
                    "**GARP (Growth at Reasonable Price)**: Balance growth potential with valuation",
                    "**Disruptive Innovation**: Identify companies creating new markets",
                    "**Scalable Business Models**: Software, platforms, network effects",
                    "**Management Quality**: Visionary leaders executing long-term strategies"
                ],

                "Alternative Strategies": [
                    "**Long/Short Equity**: Combine long positions with short selling",
                    "**Event-Driven**: Mergers, acquisitions, restructurings",
                    "**Distressed Investing**: Companies in financial trouble or bankruptcy",
                    "**Activist Investing**: Taking positions to influence management decisions"
                ],

                "Global Macro Approach": [
                    "**Currency Trends**: How exchange rates affect multinational companies",
                    "**Interest Rate Cycles**: Impact on different sectors and asset classes",
                    "**Geopolitical Events**: Wars, elections, trade policies",
                    "**Economic Indicators**: GDP, inflation, employment data interpretation"
                ]
            }
        },

        "behavioral_finance": {
            "title": "Behavioral Finance & Psychology",
            "content": {
                "Common Cognitive Biases": [
                    "**Confirmation Bias**: Seeking information that confirms existing beliefs",
                    "**Anchoring**: Over-relying on first piece of information encountered",
                    "**Overconfidence**: Overestimating one's ability to predict outcomes",
                    "**Loss Aversion**: Feeling losses more acutely than equivalent gains",
                    "**Herd Mentality**: Following crowd behavior rather than independent analysis"
                ],

                "Market Psychology Cycles": [
                    "**Euphoria Phase**: Excessive optimism, valuations reach extremes",
                    "**Denial Phase**: Refusing to acknowledge changing conditions",
                    "**Fear Phase**: Panic selling, flight to safety",
                    "**Capitulation**: Final surrender, maximum pessimism",
                    "**Recovery Phase**: Smart money starts accumulating"
                ],

                "Emotional Discipline Techniques": [
                    "**Pre-committed Rules**: Set entry/exit criteria before emotions kick in",
                    "**Position Sizing**: Risk only what you can afford to lose",
                    "**Regular Reviews**: Analyze decisions objectively with hindsight",
                    "**Meditation/Mindfulness**: Develop emotional awareness and control"
                ],

                "Contrarian Indicators": [
                    "**VIX (Fear Index)**: High VIX indicates fear, often good buying opportunity",
                    "**Put/Call Ratio**: High ratio suggests excessive pessimism",
                    "**Insider Trading**: When insiders buy heavily, often bullish signal",
                    "**Margin Debt**: Excessive margin indicates speculation, potential top",
                    "**Media Coverage**: When everyone talks about stocks, often near peak"
                ]
            }
        },

        "risk_management_mastery": {
            "title": "Advanced Risk Management",
            "content": {
                "Portfolio Risk Measures": [
                    "**Value at Risk (VaR)**: Maximum expected loss over specific time period",
                    "**Conditional VaR**: Average loss beyond VaR threshold",
                    "**Maximum Drawdown**: Largest peak-to-trough decline",
                    "**Tail Risk**: Probability of extreme losses beyond normal distribution",
                    "**Correlation Risk**: How portfolio components move together in stress"
                ],

                "Hedging Strategies": [
                    "**Portfolio Insurance**: Using puts to protect against major declines",
                    "**Collar Strategy**: Own stock + buy put + sell call (limited upside/downside)",
                    "**Currency Hedging**: Protect international investments from FX risk",
                    "**Sector Hedging**: Short sector ETFs to hedge sector concentration"
                ],

                "Position Sizing Models": [
                    "**Kelly Criterion**: Optimal bet size based on win probability and payoff",
                    "**Fixed Fractional**: Risk fixed percentage of capital per trade",
                    "**Volatility-Based**: Size positions inversely to volatility",
                    "**Risk Parity**: Equal risk contribution from each position"
                ],

                "Stress Testing": [
                    "**Historical Scenarios**: How portfolio would perform in past crises",
                    "**Monte Carlo Simulation**: Generate thousands of possible outcomes",
                    "**Factor Stress Testing**: Shock specific risk factors",
                    "**Liquidity Stress**: Estimate losses from forced selling in crisis"
                ]
            }
        },

        "institutional_strategies": {
            "title": "Institutional-Level Strategies",
            "content": {
                "Fund Management Principles": [
                    "**Asset Allocation Models**: Strategic vs tactical allocation decisions",
                    "**Benchmark Relative Performance**: Outperforming market indices consistently",
                    "**Factor Investing**: Systematic exposure to value, growth, momentum, quality factors",
                    "**Alternative Beta**: Accessing alternative risk premiums cost-effectively"
                ],

                "Quantitative Strategies": [
                    "**Statistical Arbitrage**: Exploiting temporary price discrepancies",
                    "**High-Frequency Trading**: Profiting from micro-second price movements",
                    "**Machine Learning Models**: AI-driven pattern recognition and prediction",
                    "**Multi-Factor Models**: Combining multiple signals for better predictions"
                ],

                "ESG and Sustainable Investing": [
                    "**ESG Integration**: Environmental, Social, Governance factors in analysis",
                    "**Impact Investing**: Generating positive social/environmental impact with returns",
                    "**Carbon Footprint Analysis**: Measuring and reducing portfolio carbon exposure",
                    "**Sustainable Development Goals**: Aligning investments with UN SDGs"
                ],

                "Regulatory and Compliance": [
                    "**SEBI Regulations**: Understanding regulatory framework for investments",
                    "**Tax Optimization**: Maximizing after-tax returns through tax-efficient strategies",
                    "**Reporting Requirements**: Institutional disclosure and transparency norms",
                    "**Fiduciary Responsibility**: Acting in best interest of clients/beneficiaries"
                ]
            }
        }
    }
}


def render_learning_phase(phase: str, topics: dict):
    """Render all topics for a learning phase"""
    try:
        phase_info = LEARNING_PHASES.get(phase, {"color": "#666", "icon": "📖"})

        st.markdown(f"""
        <div style="background: linear-gradient(90deg, {phase_info['color']}20, transparent); 
                    padding: 20px; border-radius: 10px; margin-bottom: 30px;">
            <h2>{phase_info['icon']} {phase.title()} Level</h2>
            <p>Master these concepts to advance to the next level</p>
        </div>
        """, unsafe_allow_html=True)

        for topic_id, topic_data in topics.items():
            with st.expander(f"{topic_data['title']}", expanded=False):
                render_topic_content(topic_id, topic_data['title'], topic_data['content'], phase.title())

    except Exception as e:
        st.error(f"Error rendering learning phase {phase}: {str(e)}")
        logger.error(f"Learning phase error for {phase}: {e}")


def render_quiz_section():
    """Render interactive quiz section"""
    try:
        st.subheader("📝 Test Your Knowledge")

        # Initialize quiz state
        if 'quiz_active' not in st.session_state:
            st.session_state.quiz_active = False
        if 'current_question' not in st.session_state:
            st.session_state.current_question = 0
        if 'quiz_score' not in st.session_state:
            st.session_state.quiz_score = 0

        # Sample quiz questions for different levels
        quiz_questions = {
            "Beginner": [
                {
                    "question": "What does P/E ratio stand for?",
                    "options": ["Price to Earnings", "Profit to Equity", "Price to Equity", "Profit to Earnings"],
                    "correct": 0,
                    "explanation": "P/E ratio means Price-to-Earnings ratio, calculated by dividing stock price by earnings per share."
                },
                {
                    "question": "What is a bull market?",
                    "options": ["Falling prices", "Rising prices", "Stable prices", "Volatile prices"],
                    "correct": 1,
                    "explanation": "A bull market is characterized by generally rising stock prices and investor optimism."
                }
            ],
            "Intermediate": [
                {
                    "question": "What is the primary purpose of diversification?",
                    "options": ["Maximize returns", "Minimize taxes", "Reduce risk", "Increase liquidity"],
                    "correct": 2,
                    "explanation": "Diversification primarily aims to reduce risk by spreading investments across different assets."
                }
            ]
        }

        selected_level = st.selectbox("Choose Quiz Level:", list(quiz_questions.keys()))

        if st.button("Start Quiz", key="start_quiz"):
            st.session_state.quiz_active = True
            st.session_state.current_question = 0
            st.session_state.quiz_score = 0

        if st.session_state.get('quiz_active', False):
            questions = quiz_questions[selected_level]
            current_q = st.session_state.get('current_question', 0)

            if current_q < len(questions):
                question_data = questions[current_q]

                st.markdown(f"**Question {current_q + 1}/{len(questions)}:**")
                st.markdown(question_data['question'])

                answer = st.radio("Choose your answer:", question_data['options'], key=f"q_{current_q}")

                if st.button("Submit Answer", key=f"submit_{current_q}"):
                    if answer == question_data['options'][question_data['correct']]:
                        st.success("✅ Correct!")
                        st.session_state.quiz_score += 1
                    else:
                        st.error("❌ Incorrect!")

                    st.info(f"💡 **Explanation:** {question_data['explanation']}")
                    st.session_state.current_question += 1

                    if st.session_state.current_question >= len(questions):
                        score_percentage = (st.session_state.quiz_score / len(questions)) * 100
                        st.balloons()
                        st.success(
                            f"🎉 Quiz Complete! Your score: {st.session_state.quiz_score}/{len(questions)} ({score_percentage:.0f}%)")
                        st.session_state.quiz_active = False

    except Exception as e:
        st.error(f"Error in quiz section: {str(e)}")
        logger.error(f"Quiz section error: {e}")


def render_learning_resources():
    """Render additional learning resources"""
    try:
        st.subheader("📚 Additional Learning Resources")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📖 Recommended Books")
            books = [
                {"title": "The Intelligent Investor", "author": "Benjamin Graham", "level": "Beginner"},
                {"title": "Common Stocks and Uncommon Profits", "author": "Philip Fisher", "level": "Intermediate"},
                {"title": "Security Analysis", "author": "Graham & Dodd", "level": "Advanced"},
                {"title": "Market Wizards", "author": "Jack Schwager", "level": "Advanced"}
            ]

            for book in books:
                level_color = LEARNING_PHASES.get(book['level'], {"color": "#666"})["color"]
                st.markdown(f"""
                <div style="border-left: 3px solid {level_color}; padding-left: 10px; margin-bottom: 10px;">
                    <strong>{book['title']}</strong><br>
                    <em>by {book['author']}</em><br>
                    <small style="color: {level_color};">{book['level']} Level</small>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("### 🎓 Online Courses")
            courses = [
                {"title": "Stock Market Basics", "platform": "NSE Academy", "level": "Beginner"},
                {"title": "Financial Modeling", "platform": "Coursera", "level": "Intermediate"},
                {"title": "CFA Level 1", "platform": "CFA Institute", "level": "Advanced"},
                {"title": "Algorithmic Trading", "platform": "QuantInsti", "level": "Expert"}
            ]

            for course in courses:
                level_color = LEARNING_PHASES.get(course['level'], {"color": "#666"})["color"]
                st.markdown(f"""
                <div style="border-left: 3px solid {level_color}; padding-left: 10px; margin-bottom: 10px;">
                    <strong>{course['title']}</strong><br>
                    <em>{course['platform']}</em><br>
                    <small style="color: {level_color};">{course['level']} Level</small>
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error rendering learning resources: {str(e)}")
        logger.error(f"Learning resources error: {e}")


def render_learning_dashboard():
    """Main learning dashboard interface"""
    try:
        st.title("🎓 Stock Market Learning Center")
        st.markdown("*Master the stock market from beginner to expert level*")

        # Initialize progress tracking
        initialize_learning_progress()

        # Progress dashboard
        render_progress_dashboard()
        st.markdown("---")

        # Navigation tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🌱 Beginner", "📈 Intermediate", "🎯 Advanced",
            "🏆 Expert", "📝 Quiz", "📚 Resources"
        ])

        with tab1:
            render_learning_phase("beginner", LEARNING_CONTENT["beginner"])

        with tab2:
            render_learning_phase("intermediate", LEARNING_CONTENT["intermediate"])

        with tab3:
            render_learning_phase("advanced", LEARNING_CONTENT["advanced"])

        with tab4:
            render_learning_phase("expert", LEARNING_CONTENT["expert"])

        with tab5:
            render_quiz_section()

        with tab6:
            render_learning_resources()

    except Exception as e:
        st.error(f"Error loading learning dashboard: {str(e)}")
        logger.error(f"Learning dashboard error: {e}")


# Main function for the learning module
def render_learn_tab():
    """Render the main learning tab"""
    try:
        render_learning_dashboard()
    except Exception as e:
        st.error(f"❌ Error loading learning module: {str(e)}")
        logger.error(f"Learning module error: {e}")

        # Fallback content
        st.markdown("""
        ### 🎓 Learning Center (Temporarily Unavailable)

        The learning center is currently experiencing issues. Please try:
        1. Refreshing the page
        2. Checking your internet connection
        3. Contacting support if the issue persists

        **What you can learn here:**
        - Stock market fundamentals from beginner to expert
        - Interactive demos and visualizations
        - Practical examples and case studies
        - Progress tracking and quizzes
        - Curated resources and reading materials
        """)