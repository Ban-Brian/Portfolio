"""
Crypto Market Microstructure Analysis
==============================================================

Analyzes order book depth, volatility, volume, price impact, bid-ask spreads,
and creates regression models for short-term returns prediction.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import warnings
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
np.random.seed(42)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ExchangeProfile:
    """Exchange-specific characteristics."""
    spread_factor: float
    depth_factor: float
    liquidity_bias: float


@dataclass
class AnalysisConfig:
    """Configuration for analysis."""
    symbols: List[str] = field(default_factory=lambda: ['BTC/USDT', 'ETH/USDT'])
    exchanges: List[str] = field(default_factory=lambda: ['coinbase', 'binance', 'kraken'])
    base_prices: Dict[str, float] = field(default_factory=lambda: {
        'BTC/USDT': 68000,
        'ETH/USDT': 2600,
        'SOL/USDT': 180,
    })
    trade_sizes: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 5.0, 10.0])
    order_book_levels: int = 20
    num_trades: int = 100
    num_ohlcv_periods: int = 100

    exchange_profiles: Dict[str, ExchangeProfile] = field(default_factory=lambda: {
        'coinbase': ExchangeProfile(spread_factor=0.05, depth_factor=1.0, liquidity_bias=0.0),
        'binance': ExchangeProfile(spread_factor=0.02, depth_factor=1.3, liquidity_bias=0.1),
        'kraken': ExchangeProfile(spread_factor=0.08, depth_factor=1.1, liquidity_bias=-0.05)
    })


# ============================================================================
# DATA GENERATION
# ============================================================================

class MarketDataGenerator:
    """Generate realistic simulated market data."""

    def __init__(self, config: AnalysisConfig):
        self.config = config

    def generate_order_book(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """Generate simulated order book with realistic characteristics."""
        base_price = self.config.base_prices.get(symbol, 100)
        profile = self.config.exchange_profiles[exchange]

        # Calculate spread
        spread_bps = profile.spread_factor
        spread = base_price * spread_bps / 100
        mid_price = base_price + np.random.normal(0, base_price * 0.001)
        best_bid = mid_price - spread / 2
        best_ask = mid_price + spread / 2

        # Generate order book levels
        bids = self._generate_order_levels(
            best_bid, profile.depth_factor, direction='bid'
        )
        asks = self._generate_order_levels(
            best_ask, profile.depth_factor, direction='ask'
        )

        return {
            'bids': bids,
            'asks': asks,
            'timestamp': datetime.now()
        }

    def _generate_order_levels(
        self,
        base_price: float,
        depth_factor: float,
        direction: str
    ) -> List[List[float]]:
        """Generate price and volume levels for order book."""
        levels = []
        sign = -1 if direction == 'bid' else 1

        for i in range(self.config.order_book_levels):
            price = base_price * (1 + sign * i * 0.0002)
            volume = np.random.exponential(5 * depth_factor) * (1 + i * 0.15)
            levels.append([price, volume])

        return levels

    def generate_trades(self, symbol: str, exchange: str) -> List[Dict[str, Any]]:
        """Generate simulated recent trades."""
        base_price = self.config.base_prices.get(symbol, 100)
        profile = self.config.exchange_profiles[exchange]

        trades = []
        buy_prob = 0.5 + profile.liquidity_bias

        for i in range(self.config.num_trades):
            price = base_price + np.random.normal(0, base_price * 0.0005)
            amount = np.random.exponential(1)
            side = np.random.choice(['buy', 'sell'], p=[buy_prob, 1 - buy_prob])

            trades.append({
                'price': price,
                'amount': amount,
                'side': side,
                'timestamp': datetime.now() - timedelta(seconds=self.config.num_trades - i)
            })

        return trades

    def generate_ohlcv(self, symbol: str, exchange: str) -> pd.DataFrame:
        """Generate simulated OHLCV data for volatility analysis."""
        base_price = self.config.base_prices.get(symbol, 100)
        data = []
        price = base_price

        for i in range(self.config.num_ohlcv_periods):
            returns = np.random.normal(0.0001, 0.001)
            price = price * (1 + returns)

            high = price * (1 + abs(np.random.normal(0, 0.002)))
            low = price * (1 - abs(np.random.normal(0, 0.002)))
            open_price = price + np.random.normal(0, price * 0.001)
            volume = np.random.exponential(100)

            data.append({
                'timestamp': datetime.now() - timedelta(minutes=self.config.num_ohlcv_periods - i),
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })

        return pd.DataFrame(data)


# ============================================================================
# METRICS CALCULATORS
# ============================================================================

class OrderBookMetrics:
    """Calculate order book related metrics."""

    @staticmethod
    def calculate_depth_metrics(order_book: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive order book depth metrics."""
        bids = np.array(order_book['bids'])
        asks = np.array(order_book['asks'])

        best_bid, best_ask = bids[0][0], asks[0][0]
        mid_price = (best_bid + best_ask) / 2

        absolute_spread = best_ask - best_bid
        relative_spread_bps = (absolute_spread / mid_price) * 10000

        bid_volume = np.sum(bids[:, 1])
        ask_volume = np.sum(asks[:, 1])
        total_volume = bid_volume + ask_volume
        imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0

        # Calculate depth at different levels
        depth_metrics = OrderBookMetrics._calculate_depth_levels(
            bids, asks, mid_price, [0.1, 0.25, 0.5, 1.0]
        )

        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'mid_price': mid_price,
            'absolute_spread': absolute_spread,
            'relative_spread_bps': relative_spread_bps,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'imbalance': imbalance,
            **depth_metrics
        }

    @staticmethod
    def _calculate_depth_levels(
        bids: np.ndarray,
        asks: np.ndarray,
        mid_price: float,
        levels: List[float]
    ) -> Dict[str, float]:
        """Calculate cumulative depth at different price thresholds."""
        depth_metrics = {}

        for level in levels:
            threshold_bid = mid_price * (1 - level / 100)
            threshold_ask = mid_price * (1 + level / 100)

            bid_depth = np.sum(bids[bids[:, 0] >= threshold_bid][:, 1])
            ask_depth = np.sum(asks[asks[:, 0] <= threshold_ask][:, 1])

            depth_metrics[f'bid_depth_{level}%'] = bid_depth
            depth_metrics[f'ask_depth_{level}%'] = ask_depth

        return depth_metrics

    @staticmethod
    def estimate_price_impact(
        order_book: Dict[str, Any],
        trade_sizes: List[float]
    ) -> Dict[str, Dict[float, Optional[float]]]:
        """Estimate price impact for different trade sizes."""
        bids = np.array(order_book['bids'])
        asks = np.array(order_book['asks'])
        mid_price = (bids[0][0] + asks[0][0]) / 2

        impacts = {'buy': {}, 'sell': {}}

        for size in trade_sizes:
            impacts['buy'][size] = OrderBookMetrics._calculate_impact(
                asks, size, mid_price, direction='buy'
            )
            impacts['sell'][size] = OrderBookMetrics._calculate_impact(
                bids, size, mid_price, direction='sell'
            )

        return impacts

    @staticmethod
    def _calculate_impact(
        levels: np.ndarray,
        size: float,
        mid_price: float,
        direction: str
    ) -> Optional[float]:
        """Calculate price impact for a single order."""
        remaining = size
        total_value = 0

        for price, volume in levels:
            if remaining <= 0:
                break
            executed = min(remaining, volume)
            total_value += executed * price
            remaining -= executed

        if remaining > 0:
            return None  # Insufficient liquidity

        avg_price = total_value / size
        if direction == 'buy':
            return (avg_price - mid_price) / mid_price * 100
        else:
            return (mid_price - avg_price) / mid_price * 100


class VolatilityMetrics:
    """Calculate volatility related metrics."""

    @staticmethod
    def calculate_all(ohlcv_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate multiple volatility measures."""
        returns = ohlcv_df['close'].pct_change().dropna()

        periods_per_year = 60 * 24 * 365
        historical_vol = returns.std() * np.sqrt(periods_per_year) * 100
        realized_vol = returns.std() * 100

        # Parkinson volatility
        hl_ratio = np.log(ohlcv_df['high'] / ohlcv_df['low'])
        parkinson_vol = np.sqrt(
            hl_ratio.pow(2).sum() / (4 * len(ohlcv_df) * np.log(2))
        ) * 100

        return {
            'returns_mean': returns.mean() * 100,
            'returns_std': returns.std() * 100,
            'historical_vol_annualized': historical_vol,
            'realized_vol': realized_vol,
            'parkinson_vol': parkinson_vol
        }


class OrderFlowMetrics:
    """Calculate order flow related metrics."""

    @staticmethod
    def analyze(trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze order flow from recent trades."""
        df = pd.DataFrame(trades)

        buy_volume = df[df['side'] == 'buy']['amount'].sum()
        sell_volume = df[df['side'] == 'sell']['amount'].sum()
        total_volume = buy_volume + sell_volume

        flow_imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0

        return {
            'trade_count': len(df),
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'total_volume': total_volume,
            'flow_imbalance': flow_imbalance,
            'avg_trade_size': df['amount'].mean()
        }


# ============================================================================
# MAIN ANALYZER
# ============================================================================

class CryptoMicrostructureAnalyzer:
    """Comprehensive market microstructure analysis."""

    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.data_generator = MarketDataGenerator(self.config)
        self.data: Dict[str, Dict[str, Any]] = {}

    def collect_data(self) -> None:
        """Collect all market data."""
        print("\n" + "=" * 70)
        print("COLLECTING MARKET DATA")
        print("=" * 70)

        for symbol in self.config.symbols:
            print(f"\n{symbol}")
            self.data[symbol] = {}

            for exchange in self.config.exchanges:
                print(f"  {exchange}...", end=" ")

                try:
                    self.data[symbol][exchange] = self._collect_exchange_data(symbol, exchange)
                    print("✓")
                except Exception as e:
                    print(f"✗ Error: {e}")
                    self.data[symbol][exchange] = None

    def _collect_exchange_data(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """Collect data for a single exchange."""
        order_book = self.data_generator.generate_order_book(symbol, exchange)
        trades = self.data_generator.generate_trades(symbol, exchange)
        ohlcv = self.data_generator.generate_ohlcv(symbol, exchange)

        return {
            'order_book': order_book,
            'trades': trades,
            'ohlcv': ohlcv,
            'depth_metrics': OrderBookMetrics.calculate_depth_metrics(order_book),
            'price_impact': OrderBookMetrics.estimate_price_impact(
                order_book, self.config.trade_sizes
            ),
            'volatility_metrics': VolatilityMetrics.calculate_all(ohlcv),
            'flow_metrics': OrderFlowMetrics.analyze(trades),
            'timestamp': datetime.now()
        }

    def create_comparison_dataframe(self) -> pd.DataFrame:
        """Create comprehensive comparison dataframe."""
        records = []

        for symbol in self.config.symbols:
            for exchange in self.config.exchanges:
                data = self.data.get(symbol, {}).get(exchange)
                if not data:
                    continue

                record = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'timestamp': data['timestamp'],
                    **data['depth_metrics'],
                    **data['volatility_metrics'],
                    **data['flow_metrics']
                }
                records.append(record)

        return pd.DataFrame(records)

    def create_price_impact_dataframe(self) -> pd.DataFrame:
        """Create price impact dataframe for analysis."""
        records = []

        for symbol in self.config.symbols:
            for exchange in self.config.exchanges:
                data = self.data.get(symbol, {}).get(exchange)
                if not data or not data['price_impact']:
                    continue

                for direction in ['buy', 'sell']:
                    for size, impact in data['price_impact'][direction].items():
                        if impact is not None:
                            records.append({
                                'symbol': symbol,
                                'exchange': exchange,
                                'direction': direction,
                                'size': size,
                                'price_impact_pct': impact
                            })

        return pd.DataFrame(records)

    def build_return_prediction_model(self) -> Dict[str, Dict[str, Any]]:
        """Build OLS regression model predicting returns from order flow."""
        print("\n" + "=" * 70)
        print("BUILDING RETURN PREDICTION MODELS")
        print("=" * 70)

        model_results = {}

        for symbol in self.config.symbols:
            print(f"\n{symbol}")

            regression_data = self._prepare_regression_data(symbol)
            if not regression_data:
                print("  ✗ No data available")
                continue

            df = pd.concat(regression_data, ignore_index=True)
            df = df.dropna(subset=['returns_next', 'flow_imbalance', 'book_imbalance'])

            if len(df) < 10:
                print("  ✗ Insufficient observations")
                continue

            try:
                model = self._fit_ols_model(df)
                model_results[symbol] = self._extract_model_results(model, len(df))
                self._print_model_summary(model, len(df))
            except Exception as e:
                print(f"  ✗ Error: {e}")

        return model_results

    def _prepare_regression_data(self, symbol: str) -> List[pd.DataFrame]:
        """Prepare data for regression analysis."""
        regression_data = []

        for exchange in self.config.exchanges:
            data = self.data.get(symbol, {}).get(exchange)
            if not data:
                continue

            ohlcv = data['ohlcv'].copy()
            ohlcv['returns'] = ohlcv['close'].pct_change()
            ohlcv['returns_next'] = ohlcv['returns'].shift(-1)
            ohlcv['flow_imbalance'] = data['flow_metrics']['flow_imbalance']
            ohlcv['book_imbalance'] = data['depth_metrics']['imbalance']
            ohlcv['spread_bps'] = data['depth_metrics']['relative_spread_bps']
            ohlcv['exchange'] = exchange

            regression_data.append(ohlcv)

        return regression_data

    def _fit_ols_model(self, df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
        """Fit OLS regression model."""
        y = df['returns_next'] * 100
        X = df[['flow_imbalance', 'book_imbalance', 'spread_bps', 'volume']]
        X = sm.add_constant(X)
        return OLS(y, X).fit()

    def _extract_model_results(
        self,
        model: sm.regression.linear_model.RegressionResultsWrapper,
        n_obs: int
    ) -> Dict[str, Any]:
        """Extract key results from fitted model."""
        return {
            'model': model,
            'n_obs': n_obs,
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'summary': model.summary()
        }

    def _print_model_summary(
        self,
        model: sm.regression.linear_model.RegressionResultsWrapper,
        n_obs: int
    ) -> None:
        """Print model summary statistics."""
        print(f"  ✓ Fitted with {n_obs} observations")
        print(f"    R²: {model.rsquared:.4f} | Adj. R²: {model.rsquared_adj:.4f}")

        sig_predictors = [
            f"{name} (coef={coef:.6f}, p={pval:.4f})"
            for name, coef, pval in zip(model.params.index, model.params, model.pvalues)
            if pval < 0.05 and name != 'const'
        ]

        if sig_predictors:
            print(f"    Significant: {', '.join(sig_predictors)}")

    def run_full_analysis(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Execute complete analysis pipeline."""
        print("\n" + "=" * 80)
        print(" " * 20 + "CRYPTO MARKET MICROSTRUCTURE ANALYSIS")
        print("=" * 80)

        self.collect_data()
        comparison_df = self.create_comparison_dataframe()
        price_impact_df = self.create_price_impact_dataframe()
        model_results = self.build_return_prediction_model()

        visualizer = Visualizer(self.config)
        figures = visualizer.generate_all(comparison_df, price_impact_df)

        reporter = Reporter(self.config)
        files = reporter.save_all(comparison_df, price_impact_df, model_results, figures)

        print("\n" + "=" * 80)
        print(" " * 30 + "✓ ANALYSIS COMPLETE")
        print("=" * 80)
        print("\nGenerated Files:")
        for filename in files.values():
            if filename:
                print(f"   {filename}")

        return comparison_df, price_impact_df, model_results


# ============================================================================
# VISUALIZATION
# ============================================================================

class Visualizer:
    """Handle all visualization generation."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    def generate_all(
        self,
        comparison_df: pd.DataFrame,
        price_impact_df: pd.DataFrame
    ) -> List[Tuple[str, plt.Figure]]:
        """Generate all visualizations."""
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)

        figures = []

        figures.extend(self._create_spread_comparison(comparison_df))
        figures.extend(self._create_imbalance_plots(comparison_df))
        figures.extend(self._create_price_impact_plots(price_impact_df))
        figures.extend(self._create_depth_plots(comparison_df))
        figures.extend(self._create_volatility_comparison(comparison_df))
        figures.extend(self._create_flow_imbalance(comparison_df))

        return figures

    def _create_spread_comparison(self, df: pd.DataFrame) -> List[Tuple[str, plt.Figure]]:
        """Create bid-ask spread comparison chart."""
        if 'relative_spread_bps' not in df.columns:
            return []

        fig, ax = plt.subplots(figsize=(12, 6))
        spread_data = df.pivot(index='symbol', columns='exchange', values='relative_spread_bps')
        spread_data.plot(kind='bar', ax=ax, color=self.colors, width=0.8)

        ax.set_title('Bid-Ask Spread Comparison', fontsize=15, fontweight='bold', pad=20)
        ax.set_xlabel('Trading Pair', fontsize=12)
        ax.set_ylabel('Spread (basis points)', fontsize=12)
        ax.legend(title='Exchange', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=0)
        plt.tight_layout()

        print("  ✓ Spread comparison")
        return [('spread_comparison', fig)]

    def _create_imbalance_plots(self, df: pd.DataFrame) -> List[Tuple[str, plt.Figure]]:
        """Create order book imbalance plots."""
        if 'imbalance' not in df.columns:
            return []

        fig, ax = plt.subplots(figsize=(12, 6))
        imbalance_data = df.pivot(index='symbol', columns='exchange', values='imbalance')
        imbalance_data.plot(kind='bar', ax=ax, color=self.colors, width=0.8)

        ax.set_title('Order Book Imbalance', fontsize=15, fontweight='bold', pad=20)
        ax.set_xlabel('Trading Pair', fontsize=12)
        ax.set_ylabel('Imbalance Ratio', fontsize=12)
        ax.legend(title='Exchange', fontsize=11)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=0)
        plt.tight_layout()

        print("  ✓ Order book imbalance")
        return [('order_book_imbalance', fig)]

    def _create_price_impact_plots(self, df: pd.DataFrame) -> List[Tuple[str, plt.Figure]]:
        """Create price impact analysis plots."""
        if df.empty:
            return []

        figures = []

        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol]
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            for idx, direction in enumerate(['buy', 'sell']):
                direction_data = symbol_data[symbol_data['direction'] == direction]

                for ex_idx, exchange in enumerate(direction_data['exchange'].unique()):
                    ex_data = direction_data[direction_data['exchange'] == exchange].sort_values('size')
                    axes[idx].plot(
                        ex_data['size'],
                        ex_data['price_impact_pct'],
                        marker='o',
                        label=exchange,
                        linewidth=2.5,
                        markersize=8,
                        color=self.colors[ex_idx]
                    )

                axes[idx].set_title(
                    f'{symbol} - {direction.capitalize()} Order Price Impact',
                    fontsize=14,
                    fontweight='bold'
                )
                axes[idx].set_xlabel('Order Size', fontsize=11)
                axes[idx].set_ylabel('Price Impact (%)', fontsize=11)
                axes[idx].legend(fontsize=10)
                axes[idx].grid(alpha=0.3)

            plt.tight_layout()
            figures.append((f'price_impact_{symbol.replace("/", "_")}', fig))
            print(f"  ✓ Price impact: {symbol}")

        return figures

    def _create_depth_plots(self, df: pd.DataFrame) -> List[Tuple[str, plt.Figure]]:
        """Create liquidity depth plots."""
        depth_cols = [col for col in df.columns if 'depth' in col]
        if not depth_cols:
            return []

        figures = []

        for symbol in self.config.symbols:
            symbol_data = df[df['symbol'] == symbol]
            if len(symbol_data) == 0:
                continue

            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(symbol_data))
            width = 0.35

            bid_col = 'bid_depth_0.1%' if 'bid_depth_0.1%' in symbol_data.columns else depth_cols[0]
            ask_col = 'ask_depth_0.1%' if 'ask_depth_0.1%' in symbol_data.columns else depth_cols[1]

            ax.bar(x - width/2, symbol_data[bid_col], width, label='Bid Depth',
                   alpha=0.8, color='#2ca02c')
            ax.bar(x + width/2, symbol_data[ask_col], width, label='Ask Depth',
                   alpha=0.8, color='#d62728')

            ax.set_xlabel('Exchange', fontsize=12)
            ax.set_ylabel('Depth (Volume)', fontsize=12)
            ax.set_title(f'{symbol} - Order Book Depth', fontsize=15, fontweight='bold', pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels(symbol_data['exchange'].values)
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            figures.append((f'depth_{symbol.replace("/", "_")}', fig))
            print(f"  ✓ Depth: {symbol}")

        return figures

    def _create_volatility_comparison(self, df: pd.DataFrame) -> List[Tuple[str, plt.Figure]]:
        """Create volatility comparison chart."""
        if 'realized_vol' not in df.columns:
            return []

        fig, ax = plt.subplots(figsize=(12, 6))
        vol_data = df.pivot(index='symbol', columns='exchange', values='realized_vol')
        vol_data.plot(kind='bar', ax=ax, color=self.colors, width=0.8)

        ax.set_title('Realized Volatility Comparison', fontsize=15, fontweight='bold', pad=20)
        ax.set_xlabel('Trading Pair', fontsize=12)
        ax.set_ylabel('Volatility (%)', fontsize=12)
        ax.legend(title='Exchange', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=0)
        plt.tight_layout()

        print("  ✓ Volatility comparison")
        return [('volatility_comparison', fig)]

    def _create_flow_imbalance(self, df: pd.DataFrame) -> List[Tuple[str, plt.Figure]]:
        """Create trade flow imbalance chart."""
        if 'flow_imbalance' not in df.columns:
            return []

        fig, ax = plt.subplots(figsize=(12, 6))
        flow_data = df.pivot(index='symbol', columns='exchange', values='flow_imbalance')
        flow_data.plot(kind='bar', ax=ax, color=self.colors, width=0.8)

        ax.set_title('Trade Flow Imbalance', fontsize=15, fontweight='bold', pad=20)
        ax.set_xlabel('Trading Pair', fontsize=12)
        ax.set_ylabel('Flow Imbalance', fontsize=12)
        ax.legend(title='Exchange', fontsize=11)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=0)
        plt.tight_layout()

        print("  ✓ Trade flow imbalance")
        return [('flow_imbalance', fig)]


# ============================================================================
# REPORTING
# ============================================================================

class Reporter:
    """Handle all file output and reporting."""

    def __init__(self, config: AnalysisConfig):
        self.config = config

    def save_all(
        self,
        comparison_df: pd.DataFrame,
        price_impact_df: pd.DataFrame,
        model_results: Dict[str, Any],
        figures: List[Tuple[str, plt.Figure]]
    ) -> Dict[str, str]:
        """Save all results to files."""
        print("\n" + "=" * 70)
        print("SAVING RESULTS")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        files = {}

        files['comparison'] = self._save_comparison_data(comparison_df, timestamp)
        files['impact'] = self._save_price_impact_data(price_impact_df, timestamp)
        files['model'] = self._save_model_results(model_results, timestamp)
        files['figures'] = self._save_figures(figures, timestamp)
        files['report'] = self._save_comprehensive_report(
            comparison_df, price_impact_df, model_results, timestamp
        )

        return files

    def _save_comparison_data(self, df: pd.DataFrame, timestamp: str) -> str:
        """Save comparison dataframe."""
        filename = f'market_comparison_{timestamp}.csv'
        df.to_csv(filename, index=False)
        print(f"  ✓ {filename}")
        return filename

    def _save_price_impact_data(self, df: pd.DataFrame, timestamp: str) -> Optional[str]:
        """Save price impact dataframe."""
        if df.empty:
            return None
        filename = f'price_impact_{timestamp}.csv'
        df.to_csv(filename, index=False)
        print(f"  ✓ {filename}")
        return filename

    def _save_model_results(self, results: Dict[str, Any], timestamp: str) -> Optional[str]:
        """Save regression model results."""
        if not results:
            return None

        filename = f'model_results_{timestamp}.txt'
        with open(filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("REGRESSION MODEL RESULTS\n")
            f.write("=" * 80 + "\n\n")
            f.write("Model: returns_next ~ flow_imbalance + book_imbalance + spread_bps + volume\n\n")

            for symbol, res in results.items():
                f.write(f"\nSymbol: {symbol}\n")
                f.write("-" * 80 + "\n")
                f.write(str(res['summary']))
                f.write("\n\n")

        print(f"  ✓ {filename}")
        return filename

    def _save_figures(self, figures: List[Tuple[str, plt.Figure]], timestamp: str) -> str:
        """Save all visualization figures."""
        for name, fig in figures:
            filename = f'{name}_{timestamp}.png'
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  ✓ {filename}")
            plt.close(fig)
        return "figures_saved"

    def _save_comprehensive_report(
        self,
        comparison_df: pd.DataFrame,
        price_impact_df: pd.DataFrame,
        model_results: Dict[str, Any],
        timestamp: str
    ) -> str:
        """Generate and save comprehensive analysis report."""
        filename = f'analysis_report_{timestamp}.txt'

        with open(filename, 'w') as f:
            self._write_report_header(f)
            self._write_executive_summary(f, comparison_df)
            self._write_detailed_metrics(f, comparison_df)
            self._write_price_impact_summary(f, price_impact_df)
            self._write_model_summary(f, model_results)
            self._write_key_insights(f, comparison_df)

        print(f"  ✓ {filename}")
        return filename

    def _write_report_header(self, f) -> None:
        """Write report header."""
        f.write("=" * 80 + "\n")
        f.write("CRYPTO MARKET MICROSTRUCTURE ANALYSIS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Symbols: {', '.join(self.config.symbols)}\n")
        f.write(f"Exchanges: {', '.join(self.config.exchanges)}\n\n")

    def _write_executive_summary(self, f, df: pd.DataFrame) -> None:
        """Write executive summary section."""
        f.write("=" * 80 + "\n")
        f.write("EXECUTIVE SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        if 'relative_spread_bps' in df.columns:
            best = df.loc[df['relative_spread_bps'].idxmin()]
            f.write(f"Tightest Spread: {best['exchange'].upper()} on {best['symbol']}\n")
            f.write(f"  {best['relative_spread_bps']:.2f} basis points\n\n")

        if 'bid_volume' in df.columns and 'ask_volume' in df.columns:
            df['total_depth'] = df['bid_volume'] + df['ask_volume']
            best = df.loc[df['total_depth'].idxmax()]
            f.write(f"Highest Liquidity: {best['exchange'].upper()} on {best['symbol']}\n")
            f.write(f"  {best['total_depth']:.2f} units total depth\n\n")

    def _write_detailed_metrics(self, f, df: pd.DataFrame) -> None:
        """Write detailed metrics section."""
        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED METRICS\n")
        f.write("=" * 80 + "\n\n")
        f.write(df.to_string())
        f.write("\n\n")

    def _write_price_impact_summary(self, f, df: pd.DataFrame) -> None:
        """Write price impact summary."""
        if df.empty:
            return

        f.write("=" * 80 + "\n")
        f.write("PRICE IMPACT SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        avg_impact = df.groupby(['exchange', 'direction'])['price_impact_pct'].mean()
        f.write(avg_impact.to_string())
        f.write("\n\n")

    def _write_model_summary(self, f, results: Dict[str, Any]) -> None:
        """Write regression model summary."""
        if not results:
            return

        f.write("=" * 80 + "\n")
        f.write("REGRESSION MODELS\n")
        f.write("=" * 80 + "\n\n")
        for symbol, res in results.items():
            f.write(f"{symbol}: R² = {res['r_squared']:.4f}, N = {res['n_obs']}\n")
        f.write("\n")

    def _write_key_insights(self, f, df: pd.DataFrame) -> None:
        """Write key insights section."""
        f.write("=" * 80 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("=" * 80 + "\n\n")

        avg_spreads = df.groupby('exchange')['relative_spread_bps'].mean()
        f.write("Execution Costs (Average Spreads):\n")
        for ex, spread in avg_spreads.items():
            f.write(f"  {ex.upper()}: {spread:.2f} bps\n")
        f.write("\n")

        f.write("Trading Recommendations:\n")
        f.write("  Small orders (<1 unit): Minimal slippage on all venues\n")
        f.write("  Medium orders (1-5 units): Consider venue liquidity\n")
        f.write("  Large orders (>10 units): Split across multiple venues\n")
        f.write("  Use order flow signals for short-term predictions\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    config = AnalysisConfig()

    print("\n" + "=" * 80)
    print("CRYPTO MARKET MICROSTRUCTURE ANALYSIS")
    print("=" * 80)
    print(f"\nAnalyzing: {', '.join(config.symbols)}")
    print(f"Exchanges: {', '.join(config.exchanges)}")

    analyzer = CryptoMicrostructureAnalyzer(config)
    comparison_df, price_impact_df, model_results = analyzer.run_full_analysis()

    return analyzer, comparison_df, price_impact_df, model_results


if __name__ == "__main__":
    analyzer, comparison_df, price_impact_df, model_results = main()