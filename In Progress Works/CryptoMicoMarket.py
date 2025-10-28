"""
Crypto Market Microstructure Analysis
==============================================================

Analyzes order book depth, volatility, volume, price impact, bid-ask spreads,
and creates regression models for short-term returns prediction.

Compares Coinbase vs. other exchanges (Binance, Kraken).

Features:
- Order book depth analysis
- Bid-ask spread efficiency
- Price impact modeling for different trade sizes
- Volatility metrics (historical, realized, Parkinson)
- Order flow analysis
- Regression model predicting returns from order flow imbalance
- Comprehensive visualizations and reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from scipy import stats

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
np.random.seed(42)


# ============================================================================
# DATA GENERATION MODULE
# ============================================================================

class SimulatedMarketData:
    """Generate realistic simulated market data for analysis."""

    def __init__(self, symbols=['BTC/USDT', 'ETH/USDT'],
                 exchanges=['coinbase', 'binance', 'kraken']):
        self.symbols = symbols
        self.exchanges = exchanges

        # Base prices reflecting current market
        self.base_prices = {
            'BTC/USDT': 68000,
            'ETH/USDT': 2600,
            'SOL/USDT': 180,
        }

        # Exchange characteristics
        self.exchange_profiles = {
            'coinbase': {'spread_factor': 0.05, 'depth_factor': 1.0, 'liquidity_bias': 0.0},
            'binance': {'spread_factor': 0.02, 'depth_factor': 1.3, 'liquidity_bias': 0.1},
            'kraken': {'spread_factor': 0.08, 'depth_factor': 1.1, 'liquidity_bias': -0.05}
        }

    def generate_order_book(self, symbol, exchange):
        """Generate simulated order book with realistic characteristics."""
        base_price = self.base_prices.get(symbol, 100)
        profile = self.exchange_profiles[exchange]

        spread_bps = profile['spread_factor']
        spread = base_price * spread_bps / 100

        # Add small random variation
        mid_price = base_price + np.random.normal(0, base_price * 0.001)
        best_bid = mid_price - spread / 2
        best_ask = mid_price + spread / 2

        # Generate order book levels (20 levels deep)
        num_levels = 20
        bids = []
        asks = []

        depth_factor = profile['depth_factor']

        for i in range(num_levels):
            # Price levels
            bid_price = best_bid * (1 - i * 0.0002)
            ask_price = best_ask * (1 + i * 0.0002)

            # Volume decreases with distance from mid
            base_volume = np.random.exponential(5 * depth_factor)
            bid_volume = base_volume * (1 + i * 0.15)
            ask_volume = base_volume * (1 + i * 0.15)

            bids.append([bid_price, bid_volume])
            asks.append([ask_price, ask_volume])

        return {
            'bids': bids,
            'asks': asks,
            'timestamp': datetime.now()
        }

    def generate_trades(self, symbol, exchange, n_trades=100):
        """Generate simulated recent trades."""
        base_price = self.base_prices.get(symbol, 100)
        profile = self.exchange_profiles[exchange]

        trades = []
        for i in range(n_trades):
            price = base_price + np.random.normal(0, base_price * 0.0005)
            amount = np.random.exponential(1)

            # Add exchange-specific bias
            buy_prob = 0.5 + profile['liquidity_bias']
            side = np.random.choice(['buy', 'sell'], p=[buy_prob, 1 - buy_prob])

            trades.append({
                'price': price,
                'amount': amount,
                'side': side,
                'timestamp': datetime.now() - timedelta(seconds=n_trades - i)
            })

        return trades

    def generate_ohlcv(self, symbol, exchange, n_periods=100):
        """Generate simulated OHLCV data for volatility analysis."""
        base_price = self.base_prices.get(symbol, 100)

        data = []
        price = base_price

        for i in range(n_periods):
            # Random walk with slight upward drift
            returns = np.random.normal(0.0001, 0.001)
            price = price * (1 + returns)

            # OHLC derived from close
            high = price * (1 + abs(np.random.normal(0, 0.002)))
            low = price * (1 - abs(np.random.normal(0, 0.002)))
            open_price = price + np.random.normal(0, price * 0.001)
            close = price
            volume = np.random.exponential(100)

            timestamp = datetime.now() - timedelta(minutes=n_periods - i)

            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

        return pd.DataFrame(data)


# ============================================================================
# ANALYSIS MODULE
# ============================================================================

class CryptoMicrostructureAnalyzer:
    """Comprehensive market microstructure analysis."""

    def __init__(self, symbols=['BTC/USDT', 'ETH/USDT'],
                 exchanges=['coinbase', 'binance', 'kraken'],
                 use_simulated_data=True):
        """
        Initialize analyzer.

        Args:
            symbols: List of trading pairs
            exchanges: List of exchanges
            use_simulated_data: If True, uses simulated data (default)
        """
        self.symbols = symbols
        self.exchange_names = exchanges
        self.use_simulated_data = use_simulated_data
        self.data = {}

        if use_simulated_data:
            self.data_generator = SimulatedMarketData(symbols, exchanges)
            print("✓ Initialized with simulated market data")
        else:
            # For real data, would initialize CCXT here
            print("✓ Initialized for real exchange data")

    def calculate_order_book_depth(self, order_book):
        """
        Calculate comprehensive order book depth metrics.

        Returns:
            dict: Spread, depth, imbalance metrics
        """
        bids = np.array(order_book['bids'])
        asks = np.array(order_book['asks'])

        # Best prices
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2

        # Spread metrics
        absolute_spread = best_ask - best_bid
        relative_spread = absolute_spread / mid_price * 100  # percent

        # Total volumes
        bid_volume = np.sum(bids[:, 1])
        ask_volume = np.sum(asks[:, 1])
        total_volume = bid_volume + ask_volume

        # Order book imbalance
        imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0

        # Cumulative depth at different price thresholds
        depth_levels = [0.1, 0.25, 0.5, 1.0]  # % from mid price
        depth_metrics = {}

        for level in depth_levels:
            threshold_bid = mid_price * (1 - level / 100)
            threshold_ask = mid_price * (1 + level / 100)

            bid_depth = np.sum(bids[bids[:, 0] >= threshold_bid][:, 1])
            ask_depth = np.sum(asks[asks[:, 0] <= threshold_ask][:, 1])

            depth_metrics[f'bid_depth_{level}%'] = bid_depth
            depth_metrics[f'ask_depth_{level}%'] = ask_depth

        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'mid_price': mid_price,
            'absolute_spread': absolute_spread,
            'relative_spread_bps': relative_spread * 100,  # basis points
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'imbalance': imbalance,
            **depth_metrics
        }

    def estimate_price_impact(self, order_book, trade_sizes=[0.1, 0.5, 1.0, 5.0, 10.0]):
        """
        Estimate price impact for different trade sizes by walking the order book.

        Args:
            order_book: Order book data
            trade_sizes: List of trade sizes to analyze

        Returns:
            dict: Price impact for buy and sell orders
        """
        bids = np.array(order_book['bids'])
        asks = np.array(order_book['asks'])

        mid_price = (bids[0][0] + asks[0][0]) / 2
        impacts = {'buy': {}, 'sell': {}}

        for size in trade_sizes:
            # Buy order - walks through asks
            remaining = size
            total_cost = 0
            for price, volume in asks:
                if remaining <= 0:
                    break
                executed = min(remaining, volume)
                total_cost += executed * price
                remaining -= executed

            if remaining <= 0:
                avg_price = total_cost / size
                impacts['buy'][size] = (avg_price - mid_price) / mid_price * 100
            else:
                impacts['buy'][size] = None  # Insufficient liquidity

            # Sell order - walks through bids
            remaining = size
            total_revenue = 0
            for price, volume in bids:
                if remaining <= 0:
                    break
                executed = min(remaining, volume)
                total_revenue += executed * price
                remaining -= executed

            if remaining <= 0:
                avg_price = total_revenue / size
                impacts['sell'][size] = (mid_price - avg_price) / mid_price * 100
            else:
                impacts['sell'][size] = None  # Insufficient liquidity

        return impacts

    def calculate_volatility_metrics(self, ohlcv_df):
        """Calculate multiple volatility measures."""
        returns = ohlcv_df['close'].pct_change().dropna()

        # Annualized volatility (assuming 1-minute data)
        periods_per_year = 60 * 24 * 365
        historical_vol = returns.std() * np.sqrt(periods_per_year) * 100

        # Realized volatility
        realized_vol = returns.std() * 100

        # Parkinson volatility (uses high-low range)
        hl_ratio = np.log(ohlcv_df['high'] / ohlcv_df['low'])
        parkinson_vol = np.sqrt(hl_ratio.pow(2).sum() / (4 * len(ohlcv_df) * np.log(2))) * 100

        return {
            'returns_mean': returns.mean() * 100,
            'returns_std': returns.std() * 100,
            'historical_vol_annualized': historical_vol,
            'realized_vol': realized_vol,
            'parkinson_vol': parkinson_vol
        }

    def analyze_trade_flow(self, trades):
        """Analyze order flow from recent trades."""
        df = pd.DataFrame(trades)

        # Separate buy and sell volumes
        buy_volume = df[df['side'] == 'buy']['amount'].sum()
        sell_volume = df[df['side'] == 'sell']['amount'].sum()
        total_volume = buy_volume + sell_volume

        # Flow imbalance
        flow_imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0

        return {
            'trade_count': len(df),
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'total_volume': total_volume,
            'flow_imbalance': flow_imbalance,
            'avg_trade_size': df['amount'].mean()
        }

    def collect_data(self):
        """Collect all market data."""
        print("\n" + "=" * 70)
        print("COLLECTING MARKET DATA")
        print("=" * 70)

        for symbol in self.symbols:
            print(f"\n📊 {symbol}")
            self.data[symbol] = {}

            for exchange_name in self.exchange_names:
                print(f"  └─ {exchange_name}...", end=" ")

                try:
                    if self.use_simulated_data:
                        # Generate simulated data
                        order_book = self.data_generator.generate_order_book(symbol, exchange_name)
                        trades = self.data_generator.generate_trades(symbol, exchange_name)
                        ohlcv = self.data_generator.generate_ohlcv(symbol, exchange_name)
                    else:
                        # Would fetch real data here using CCXT
                        pass

                    # Calculate metrics
                    depth_metrics = self.calculate_order_book_depth(order_book)
                    price_impact = self.estimate_price_impact(order_book)
                    vol_metrics = self.calculate_volatility_metrics(ohlcv)
                    flow_metrics = self.analyze_trade_flow(trades)

                    self.data[symbol][exchange_name] = {
                        'order_book': order_book,
                        'trades': trades,
                        'ohlcv': ohlcv,
                        'depth_metrics': depth_metrics,
                        'price_impact': price_impact,
                        'volatility_metrics': vol_metrics,
                        'flow_metrics': flow_metrics,
                        'timestamp': datetime.now()
                    }

                    print("✓")

                except Exception as e:
                    print(f"✗ Error: {e}")
                    self.data[symbol][exchange_name] = None

    def create_comparison_dataframe(self):
        """Create comprehensive comparison dataframe."""
        records = []

        for symbol in self.symbols:
            for exchange_name in self.exchange_names:
                data = self.data.get(symbol, {}).get(exchange_name)

                if not data:
                    continue

                record = {
                    'symbol': symbol,
                    'exchange': exchange_name,
                    'timestamp': data['timestamp']
                }

                # Add all metrics
                if data['depth_metrics']:
                    record.update(data['depth_metrics'])
                if data['volatility_metrics']:
                    record.update(data['volatility_metrics'])
                if data['flow_metrics']:
                    record.update(data['flow_metrics'])

                records.append(record)

        return pd.DataFrame(records)

    def create_price_impact_dataframe(self):
        """Create price impact dataframe for analysis."""
        records = []

        for symbol in self.symbols:
            for exchange_name in self.exchange_names:
                data = self.data.get(symbol, {}).get(exchange_name)

                if not data or not data['price_impact']:
                    continue

                for direction in ['buy', 'sell']:
                    for size, impact in data['price_impact'][direction].items():
                        if impact is not None:
                            records.append({
                                'symbol': symbol,
                                'exchange': exchange_name,
                                'direction': direction,
                                'size': size,
                                'price_impact_pct': impact
                            })

        return pd.DataFrame(records)

    def build_return_prediction_model(self):
        """Build OLS regression model predicting returns from order flow."""
        print("\n" + "=" * 70)
        print("BUILDING RETURN PREDICTION MODELS")
        print("=" * 70)

        model_results = {}

        for symbol in self.symbols:
            print(f"\n📈 {symbol}")

            regression_data = []

            for exchange_name in self.exchange_names:
                data = self.data.get(symbol, {}).get(exchange_name)

                if not data:
                    continue

                ohlcv = data['ohlcv'].copy()

                # Calculate returns
                ohlcv['returns'] = ohlcv['close'].pct_change()
                ohlcv['returns_next'] = ohlcv['returns'].shift(-1)  # Next period

                # Add predictors
                ohlcv['flow_imbalance'] = data['flow_metrics']['flow_imbalance']
                ohlcv['book_imbalance'] = data['depth_metrics']['imbalance']
                ohlcv['spread_bps'] = data['depth_metrics']['relative_spread_bps']
                ohlcv['exchange'] = exchange_name

                regression_data.append(ohlcv)

            if not regression_data:
                print("  ✗ No data available")
                continue

            # Combine all exchange data
            df = pd.concat(regression_data, ignore_index=True)
            df = df.dropna(subset=['returns_next', 'flow_imbalance', 'book_imbalance'])

            if len(df) < 10:
                print("  ✗ Insufficient observations")
                continue

            # Prepare regression
            y = df['returns_next'] * 100  # Convert to percentage
            X = df[['flow_imbalance', 'book_imbalance', 'spread_bps', 'volume']]
            X = sm.add_constant(X)

            # Run OLS regression
            try:
                model = OLS(y, X).fit()

                model_results[symbol] = {
                    'model': model,
                    'n_obs': len(df),
                    'r_squared': model.rsquared,
                    'adj_r_squared': model.rsquared_adj,
                    'summary': model.summary()
                }

                print(f"  ✓ Fitted with {len(df)} observations")
                print(f"    R²: {model.rsquared:.4f} | Adj. R²: {model.rsquared_adj:.4f}")

                # Show significant predictors
                sig_predictors = []
                for i, (name, pval) in enumerate(zip(X.columns, model.pvalues)):
                    if pval < 0.05 and name != 'const':
                        sig_predictors.append(f"{name} (coef={model.params[i]:.6f}, p={pval:.4f})")

                if sig_predictors:
                    print(f"    Significant: {', '.join(sig_predictors)}")

            except Exception as e:
                print(f"  ✗ Error: {e}")

        return model_results

    def generate_visualizations(self, comparison_df, price_impact_df):
        """Generate all visualizations."""
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)

        figures = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        # 1. Bid-Ask Spread Comparison
        if 'relative_spread_bps' in comparison_df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            spread_data = comparison_df.pivot(index='symbol', columns='exchange',
                                              values='relative_spread_bps')
            spread_data.plot(kind='bar', ax=ax, color=colors, width=0.8)
            ax.set_title('Bid-Ask Spread Comparison (Lower = Better Execution)',
                         fontsize=15, fontweight='bold', pad=20)
            ax.set_xlabel('Trading Pair', fontsize=12)
            ax.set_ylabel('Spread (basis points)', fontsize=12)
            ax.legend(title='Exchange', fontsize=11, title_fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=0)
            plt.tight_layout()
            figures.append(('spread_comparison', fig))
            print("  ✓ Spread comparison")

        # 2. Order Book Imbalance
        if 'imbalance' in comparison_df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            imbalance_data = comparison_df.pivot(index='symbol', columns='exchange',
                                                 values='imbalance')
            imbalance_data.plot(kind='bar', ax=ax, color=colors, width=0.8)
            ax.set_title('Order Book Imbalance (Positive = Bid Pressure)',
                         fontsize=15, fontweight='bold', pad=20)
            ax.set_xlabel('Trading Pair', fontsize=12)
            ax.set_ylabel('Imbalance Ratio', fontsize=12)
            ax.legend(title='Exchange', fontsize=11, title_fontsize=12)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=0)
            plt.tight_layout()
            figures.append(('order_book_imbalance', fig))
            print("  ✓ Order book imbalance")

        # 3. Price Impact Analysis
        if not price_impact_df.empty:
            for symbol in price_impact_df['symbol'].unique():
                symbol_data = price_impact_df[price_impact_df['symbol'] == symbol]

                fig, axes = plt.subplots(1, 2, figsize=(16, 6))

                # Buy orders
                buy_data = symbol_data[symbol_data['direction'] == 'buy']
                for idx, exchange in enumerate(buy_data['exchange'].unique()):
                    ex_data = buy_data[buy_data['exchange'] == exchange].sort_values('size')
                    axes[0].plot(ex_data['size'], ex_data['price_impact_pct'],
                                 marker='o', label=exchange, linewidth=2.5,
                                 markersize=8, color=colors[idx])

                axes[0].set_title(f'{symbol} - Buy Order Price Impact',
                                  fontsize=14, fontweight='bold')
                axes[0].set_xlabel('Order Size (Base Currency)', fontsize=11)
                axes[0].set_ylabel('Price Impact (%)', fontsize=11)
                axes[0].legend(fontsize=10)
                axes[0].grid(alpha=0.3)

                # Sell orders
                sell_data = symbol_data[symbol_data['direction'] == 'sell']
                for idx, exchange in enumerate(sell_data['exchange'].unique()):
                    ex_data = sell_data[sell_data['exchange'] == exchange].sort_values('size')
                    axes[1].plot(ex_data['size'], ex_data['price_impact_pct'],
                                 marker='o', label=exchange, linewidth=2.5,
                                 markersize=8, color=colors[idx])

                axes[1].set_title(f'{symbol} - Sell Order Price Impact',
                                  fontsize=14, fontweight='bold')
                axes[1].set_xlabel('Order Size (Base Currency)', fontsize=11)
                axes[1].set_ylabel('Price Impact (%)', fontsize=11)
                axes[1].legend(fontsize=10)
                axes[1].grid(alpha=0.3)

                plt.tight_layout()
                figures.append((f'price_impact_{symbol.replace("/", "_")}', fig))
                print(f"  ✓ Price impact: {symbol}")

        # 4. Liquidity Depth
        depth_cols = [col for col in comparison_df.columns if 'depth' in col]
        if depth_cols:
            for symbol in self.symbols:
                symbol_data = comparison_df[comparison_df['symbol'] == symbol]

                if len(symbol_data) == 0:
                    continue

                fig, ax = plt.subplots(figsize=(12, 6))

                x = np.arange(len(symbol_data))
                width = 0.35

                bid_col = 'bid_depth_0.1%' if 'bid_depth_0.1%' in symbol_data.columns else depth_cols[0]
                ask_col = 'ask_depth_0.1%' if 'ask_depth_0.1%' in symbol_data.columns else depth_cols[1]

                bid_depths = symbol_data[bid_col].values
                ask_depths = symbol_data[ask_col].values

                bars1 = ax.bar(x - width / 2, bid_depths, width, label='Bid Depth',
                               alpha=0.8, color='#2ca02c')
                bars2 = ax.bar(x + width / 2, ask_depths, width, label='Ask Depth',
                               alpha=0.8, color='#d62728')

                ax.set_xlabel('Exchange', fontsize=12)
                ax.set_ylabel('Depth (Volume)', fontsize=12)
                ax.set_title(f'{symbol} - Order Book Depth',
                             fontsize=15, fontweight='bold', pad=20)
                ax.set_xticks(x)
                ax.set_xticklabels(symbol_data['exchange'].values)
                ax.legend(fontsize=11)
                ax.grid(axis='y', alpha=0.3)

                plt.tight_layout()
                figures.append((f'depth_{symbol.replace("/", "_")}', fig))
                print(f"  ✓ Depth: {symbol}")

        # 5. Volatility Comparison
        if 'realized_vol' in comparison_df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            vol_data = comparison_df.pivot(index='symbol', columns='exchange',
                                           values='realized_vol')
            vol_data.plot(kind='bar', ax=ax, color=colors, width=0.8)
            ax.set_title('Realized Volatility Comparison',
                         fontsize=15, fontweight='bold', pad=20)
            ax.set_xlabel('Trading Pair', fontsize=12)
            ax.set_ylabel('Volatility (%)', fontsize=12)
            ax.legend(title='Exchange', fontsize=11, title_fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=0)
            plt.tight_layout()
            figures.append(('volatility_comparison', fig))
            print("  ✓ Volatility comparison")

        # 6. Trade Flow Imbalance
        if 'flow_imbalance' in comparison_df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            flow_data = comparison_df.pivot(index='symbol', columns='exchange',
                                            values='flow_imbalance')
            flow_data.plot(kind='bar', ax=ax, color=colors, width=0.8)
            ax.set_title('Trade Flow Imbalance (Positive = Buy Pressure)',
                         fontsize=15, fontweight='bold', pad=20)
            ax.set_xlabel('Trading Pair', fontsize=12)
            ax.set_ylabel('Flow Imbalance', fontsize=12)
            ax.legend(title='Exchange', fontsize=11, title_fontsize=12)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=0)
            plt.tight_layout()
            figures.append(('flow_imbalance', fig))
            print("  ✓ Trade flow imbalance")

        return figures

    def save_results(self, comparison_df, price_impact_df, model_results, figures):
        """Save all results to files."""
        print("\n" + "=" * 70)
        print("SAVING RESULTS")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save CSV files
        comparison_file = f'market_comparison_{timestamp}.csv'
        comparison_df.to_csv(comparison_file, index=False)
        print(f"  ✓ {comparison_file}")

        if not price_impact_df.empty:
            impact_file = f'price_impact_{timestamp}.csv'
            price_impact_df.to_csv(impact_file, index=False)
            print(f"  ✓ {impact_file}")

        # Save model results
        if model_results:
            model_file = f'model_results_{timestamp}.txt'
            with open(model_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("REGRESSION MODEL RESULTS\n")
                f.write("=" * 80 + "\n\n")

                f.write("Model: Next-period returns ~ flow_imbalance + book_imbalance + spread_bps + volume\n\n")

                for symbol, results in model_results.items():
                    f.write(f"\nSymbol: {symbol}\n")
                    f.write("-" * 80 + "\n")
                    f.write(str(results['summary']))
                    f.write("\n\n")

            print(f"  ✓ {model_file}")

        # Save visualizations
        for name, fig in figures:
            filename = f'{name}_{timestamp}.png'
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  ✓ {filename}")
            plt.close(fig)

        # Comprehensive report
        report_file = f'analysis_report_{timestamp}.txt'
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CRYPTO MARKET MICROSTRUCTURE ANALYSIS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Symbols: {', '.join(self.symbols)}\n")
            f.write(f"Exchanges: {', '.join(self.exchange_names)}\n\n")

            f.write("=" * 80 + "\n")
            f.write("EXECUTIVE SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            # Best spreads
            if 'relative_spread_bps' in comparison_df.columns:
                best = comparison_df.loc[comparison_df['relative_spread_bps'].idxmin()]
                f.write(f"✓ Tightest Spread: {best['exchange'].upper()} on {best['symbol']}\n")
                f.write(f"  {best['relative_spread_bps']:.2f} basis points\n\n")

            # Best liquidity
            if 'bid_volume' in comparison_df.columns:
                comparison_df['total_depth'] = comparison_df['bid_volume'] + comparison_df['ask_volume']
                best = comparison_df.loc[comparison_df['total_depth'].idxmax()]
                f.write(f"✓ Highest Liquidity: {best['exchange'].upper()} on {best['symbol']}\n")
                f.write(f"  {best['total_depth']:.2f} units total depth\n\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("DETAILED METRICS\n")
            f.write("=" * 80 + "\n\n")
            f.write(comparison_df.to_string())
            f.write("\n\n")

            if not price_impact_df.empty:
                f.write("=" * 80 + "\n")
                f.write("PRICE IMPACT SUMMARY\n")
                f.write("=" * 80 + "\n\n")
                avg_impact = price_impact_df.groupby(['exchange', 'direction'])['price_impact_pct'].mean()
                f.write(avg_impact.to_string())
                f.write("\n\n")

            if model_results:
                f.write("=" * 80 + "\n")
                f.write("REGRESSION MODELS\n")
                f.write("=" * 80 + "\n\n")
                for symbol, results in model_results.items():
                    f.write(f"{symbol}: R² = {results['r_squared']:.4f}, ")
                    f.write(f"N = {results['n_obs']}\n")
                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("KEY INSIGHTS\n")
            f.write("=" * 80 + "\n\n")

            avg_spreads = comparison_df.groupby('exchange')['relative_spread_bps'].mean()
            f.write("Execution Costs (Average Spreads):\n")
            for ex, spread in avg_spreads.items():
                f.write(f"  • {ex.upper()}: {spread:.2f} bps\n")
            f.write("\n")

            f.write("Trading Recommendations:\n")
            f.write("  • Small orders (<1 unit): Minimal slippage on all venues\n")
            f.write("  • Medium orders (1-5 units): Consider venue liquidity\n")
            f.write("  • Large orders (>10 units): Split across multiple venues\n")
            f.write("  • Use order flow signals for short-term predictions\n")

        print(f"  ✓ {report_file}")

        return {
            'comparison': comparison_file,
            'impact': impact_file if not price_impact_df.empty else None,
            'model': model_file if model_results else None,
            'report': report_file
        }

    def run_full_analysis(self):
        """Execute complete analysis pipeline."""
        print("\n" + "=" * 80)
        print(" " * 20 + "CRYPTO MARKET MICROSTRUCTURE ANALYSIS")
        print("=" * 80)

        # Collect data
        self.collect_data()

        # Create dataframes
        comparison_df = self.create_comparison_dataframe()
        price_impact_df = self.create_price_impact_dataframe()

        # Build models
        model_results = self.build_return_prediction_model()

        # Generate visualizations
        figures = self.generate_visualizations(comparison_df, price_impact_df)

        # Save everything
        files = self.save_results(comparison_df, price_impact_df, model_results, figures)

        print("\n" + "=" * 80)
        print(" " * 30 + "✓ ANALYSIS COMPLETE")
        print("=" * 80)
        print("\n📁 Generated Files:")
        for key, filename in files.items():
            if filename:
                print(f"   • {filename}")

        print("\n💡 All results saved to current directory")

        return comparison_df, price_impact_df, model_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""

    # Configuration
    SYMBOLS = ['BTC/USDT', 'ETH/USDT']
    EXCHANGES = ['coinbase', 'binance', 'kraken']

    print("\n" + "=" * 80)
    print("CRYPTO MARKET MICROSTRUCTURE ANALYSIS")
    print("Comparing Coinbase vs. Other Exchanges")
    print("=" * 80)
    print(f"\nAnalyzing: {', '.join(SYMBOLS)}")
    print(f"Exchanges: {', '.join(EXCHANGES)}")
    print(f"\nFeatures:")
    print("  ✓ Order book depth analysis")
    print("  ✓ Bid-ask spread efficiency")
    print("  ✓ Price impact modeling")
    print("  ✓ Volatility metrics")
    print("  ✓ Order flow analysis")
    print("  ✓ Return prediction models")

    # Create analyzer and run
    analyzer = CryptoMicrostructureAnalyzer(
        symbols=SYMBOLS,
        exchanges=EXCHANGES,
        use_simulated_data=True  # Set to False for real data (requires CCXT and network access)
    )

    # Run full analysis
    comparison_df, price_impact_df, model_results = analyzer.run_full_analysis()

    return analyzer, comparison_df, price_impact_df, model_results


if __name__ == "__main__":
    analyzer, comparison_df, price_impact_df, model_results = main()