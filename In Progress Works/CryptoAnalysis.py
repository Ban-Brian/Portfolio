import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import warnings
import sys

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AnalysisResult:
    """Container for analysis results."""
    symbol: str
    timestamp: datetime
    price_data: pd.DataFrame
    order_book: Dict[str, Any]
    metrics: Dict[str, Any]
    model_results: Optional[Dict[str, Any]]
    figures: List[Tuple[str, plt.Figure]]


# ============================================================================
# DATA FETCHER
# ============================================================================

class CryptoDataFetcher:
    """Fetch real market data from public APIs (CoinGecko + Binance)."""
    
    def __init__(self):
        self.coingecko = 'https://api.coingecko.com/api/v3'
        self.binance = 'https://api.binance.com/api/v3'
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'CryptoAnalyzer/1.0'})
    
    def search_coin(self, query: str) -> List[Dict]:
        """Search for a coin by name or symbol."""
        resp = self.session.get(f"{self.coingecko}/search", params={'query': query}, timeout=10)
        resp.raise_for_status()
        return resp.json().get('coins', [])[:10]
    
    def get_market_data(self, coin_id: str) -> Dict[str, Any]:
        """Fetch current market data from CoinGecko."""
        params = {
            'localization': 'false',
            'tickers': 'true',
            'market_data': 'true',
            'community_data': 'false',
            'developer_data': 'false'
        }
        resp = self.session.get(f"{self.coingecko}/coins/{coin_id}", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    
    def get_binance_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Fetch order book from Binance."""
        resp = self.session.get(
            f"{self.binance}/depth",
            params={'symbol': symbol.upper(), 'limit': limit},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            'bids': [[float(p), float(q)] for p, q in data['bids']],
            'asks': [[float(p), float(q)] for p, q in data['asks']],
            'timestamp': datetime.now()
        }
    
    def get_binance_trades(self, symbol: str, limit: int = 500) -> List[Dict]:
        """Fetch recent trades from Binance."""
        resp = self.session.get(
            f"{self.binance}/trades",
            params={'symbol': symbol.upper(), 'limit': limit},
            timeout=10
        )
        resp.raise_for_status()
        return [
            {
                'price': float(t['price']),
                'amount': float(t['qty']),
                'side': 'buy' if t['isBuyerMaker'] else 'sell',
                'timestamp': datetime.fromtimestamp(t['time'] / 1000)
            }
            for t in resp.json()
        ]
    
    def get_binance_klines(self, symbol: str, interval: str = '1h', limit: int = 500) -> pd.DataFrame:
        """Fetch OHLCV data from Binance."""
        resp = self.session.get(
            f"{self.binance}/klines",
            params={'symbol': symbol.upper(), 'interval': interval, 'limit': limit},
            timeout=15
        )
        resp.raise_for_status()
        
        df = pd.DataFrame(resp.json(), columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = df[col].astype(float)
        df.set_index('timestamp', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']]
    
    def get_coingecko_ohlc(self, coin_id: str, days: int = 30) -> pd.DataFrame:
        """Fallback: fetch OHLC from CoinGecko (no volume)."""
        # CoinGecko only supports: 1, 7, 14, 30, 90, 180, 365, max
        valid_days = [1, 7, 14, 30, 90, 180, 365]
        actual_days = min([d for d in valid_days if d >= days], default=30)
        
        resp = self.session.get(
            f"{self.coingecko}/coins/{coin_id}/ohlc",
            params={'vs_currency': 'usd', 'days': actual_days},
            timeout=15
        )
        resp.raise_for_status()
        df = pd.DataFrame(resp.json(), columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['volume'] = 0
        return df


# ============================================================================
# METRICS CALCULATORS
# ============================================================================

class MetricsCalculator:
    """Calculate market microstructure metrics."""
    
    @staticmethod
    def order_book_metrics(order_book: Dict[str, Any]) -> Dict[str, float]:
        """Calculate order book depth and spread metrics."""
        bids = np.array(order_book['bids'])
        asks = np.array(order_book['asks'])
        
        if len(bids) == 0 or len(asks) == 0:
            return {}
        
        best_bid, best_ask = bids[0][0], asks[0][0]
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_bps = (spread / mid_price) * 10000
        
        bid_volume = np.sum(bids[:, 1])
        ask_volume = np.sum(asks[:, 1])
        total_volume = bid_volume + ask_volume
        imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
        
        depth_metrics = {}
        for pct in [0.1, 0.5, 1.0, 2.0]:
            bid_threshold = mid_price * (1 - pct / 100)
            ask_threshold = mid_price * (1 + pct / 100)
            bid_depth = np.sum(bids[bids[:, 0] >= bid_threshold][:, 1])
            ask_depth = np.sum(asks[asks[:, 0] <= ask_threshold][:, 1])
            depth_metrics[f'bid_depth_{pct}pct'] = bid_depth
            depth_metrics[f'ask_depth_{pct}pct'] = ask_depth
            depth_metrics[f'total_depth_{pct}pct'] = bid_depth + ask_depth
        
        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'mid_price': mid_price,
            'spread': spread,
            'spread_bps': spread_bps,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'book_imbalance': imbalance,
            **depth_metrics
        }
    
    @staticmethod
    def price_impact(order_book: Dict[str, Any], sizes: List[float]) -> Dict[str, Dict[float, Optional[float]]]:
        """Estimate price impact for various order sizes."""
        bids = np.array(order_book['bids'])
        asks = np.array(order_book['asks'])
        mid_price = (bids[0][0] + asks[0][0]) / 2
        
        impacts = {'buy': {}, 'sell': {}}
        
        for size in sizes:
            for direction, levels in [('buy', asks), ('sell', bids)]:
                remaining = size
                total = 0
                for price, volume in levels:
                    if remaining <= 0:
                        break
                    executed = min(remaining, volume)
                    total += executed * price
                    remaining -= executed
                
                if remaining <= 0:
                    avg_price = total / size
                    if direction == 'buy':
                        impacts[direction][size] = (avg_price - mid_price) / mid_price * 100
                    else:
                        impacts[direction][size] = (mid_price - avg_price) / mid_price * 100
                else:
                    impacts[direction][size] = None
        
        return impacts
    
    @staticmethod
    def volatility_metrics(ohlcv: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility measures."""
        returns = ohlcv['close'].pct_change().dropna()
        if len(returns) < 2:
            return {}
        
        avg_delta = (ohlcv.index[-1] - ohlcv.index[0]) / len(ohlcv)
        periods_per_year = timedelta(days=365) / avg_delta if avg_delta.total_seconds() > 0 else 8760
        
        historical_vol = returns.std() * np.sqrt(periods_per_year) * 100
        
        hl_ratio = np.log(ohlcv['high'] / ohlcv['low'])
        parkinson_vol = np.sqrt(hl_ratio.pow(2).mean() / (4 * np.log(2))) * np.sqrt(periods_per_year) * 100
        
        log_hl = np.log(ohlcv['high'] / ohlcv['low']) ** 2
        log_co = np.log(ohlcv['close'] / ohlcv['open']) ** 2
        gk_vol = np.sqrt((0.5 * log_hl - (2 * np.log(2) - 1) * log_co).mean()) * np.sqrt(periods_per_year) * 100
        
        peak = ohlcv['close'].expanding().max()
        max_dd = ((ohlcv['close'] - peak) / peak).min() * 100
        
        sharpe = (returns.mean() / returns.std()) * np.sqrt(periods_per_year) if returns.std() > 0 else 0
        
        return {
            'returns_mean': returns.mean() * 100,
            'returns_std': returns.std() * 100,
            'historical_vol_annualized': historical_vol,
            'parkinson_vol_annualized': parkinson_vol,
            'garman_klass_vol_annualized': gk_vol,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe
        }
    
    @staticmethod
    def trade_flow_metrics(trades: List[Dict]) -> Dict[str, float]:
        """Analyze trade flow."""
        if not trades:
            return {}
        
        df = pd.DataFrame(trades)
        buy_vol = df[df['side'] == 'buy']['amount'].sum()
        sell_vol = df[df['side'] == 'sell']['amount'].sum()
        total_vol = buy_vol + sell_vol
        
        buy_value = (df[df['side'] == 'buy']['amount'] * df[df['side'] == 'buy']['price']).sum()
        sell_value = (df[df['side'] == 'sell']['amount'] * df[df['side'] == 'sell']['price']).sum()
        
        flow_imbalance = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0
        vwap = (df['amount'] * df['price']).sum() / df['amount'].sum() if df['amount'].sum() > 0 else 0
        
        return {
            'trade_count': len(df),
            'buy_volume': buy_vol,
            'sell_volume': sell_vol,
            'total_volume': total_vol,
            'buy_value_usd': buy_value,
            'sell_value_usd': sell_value,
            'flow_imbalance': flow_imbalance,
            'avg_trade_size': df['amount'].mean(),
            'vwap': vwap
        }


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

class TechnicalIndicators:
    """Calculate technical analysis indicators."""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return {'macd': macd_line, 'signal': signal_line, 'histogram': histogram}
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        return {
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev),
            'bandwidth': ((sma + std * std_dev) - (sma - std * std_dev)) / sma * 100
        }
    
    @staticmethod
    def calculate_stochastic(ohlcv: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator."""
        low_min = ohlcv['low'].rolling(window=k_period).min()
        high_max = ohlcv['high'].rolling(window=k_period).max()
        k = 100 * (ohlcv['close'] - low_min) / (high_max - low_min)
        d = k.rolling(window=d_period).mean()
        return {'k': k, 'd': d}
    
    @staticmethod
    def calculate_atr(ohlcv: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range for volatility."""
        high_low = ohlcv['high'] - ohlcv['low']
        high_close = abs(ohlcv['high'] - ohlcv['close'].shift())
        low_close = abs(ohlcv['low'] - ohlcv['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    @staticmethod
    def find_support_resistance(ohlcv: pd.DataFrame, lookback: int = 50) -> Dict[str, List[float]]:
        """Find support and resistance levels."""
        recent = ohlcv.tail(lookback)
        current_price = ohlcv['close'].iloc[-1]
        
        # Find local minima (support) and maxima (resistance)
        highs = recent['high'].values
        lows = recent['low'].values
        
        supports = []
        resistances = []
        
        for i in range(2, len(recent) - 2):
            # Local minimum = support
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                if lows[i] < current_price:
                    supports.append(lows[i])
            # Local maximum = resistance
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                if highs[i] > current_price:
                    resistances.append(highs[i])
        
        supports = sorted(set(supports), reverse=True)[:3]  # Top 3 nearest supports
        resistances = sorted(set(resistances))[:3]  # Top 3 nearest resistances
        
        return {'support': supports, 'resistance': resistances}
    
    @staticmethod
    def fibonacci_retracements(ohlcv: pd.DataFrame, lookback: int = 100) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels."""
        recent = ohlcv.tail(lookback)
        high = recent['high'].max()
        low = recent['low'].min()
        diff = high - low
        
        return {
            '0.0%': high,
            '23.6%': high - diff * 0.236,
            '38.2%': high - diff * 0.382,
            '50.0%': high - diff * 0.5,
            '61.8%': high - diff * 0.618,
            '78.6%': high - diff * 0.786,
            '100.0%': low
        }


# ============================================================================
# INVESTMENT SCORER
# ============================================================================

class InvestmentScorer:
    """Score coins and provide investment recommendations."""
    
    def __init__(self, fetcher=None):
        self.fetcher = fetcher
        self.indicators = TechnicalIndicators()
    
    def calculate_score(self, metrics: Dict, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive investment score (0-100)."""
        signals = {'bullish': [], 'bearish': [], 'neutral': []}
        
        # Technical signals (30 points max)
        tech_score, tech_signals = self._technical_score(ohlcv)
        signals.update({k: signals[k] + v for k, v in tech_signals.items()})
        
        # Volatility/Risk signals (20 points max)
        risk_score, risk_signals = self._risk_score(metrics.get('volatility', {}))
        signals.update({k: signals[k] + v for k, v in risk_signals.items()})
        
        # Market strength signals (20 points max)
        market_score, market_signals = self._market_score(metrics.get('market', {}))
        signals.update({k: signals[k] + v for k, v in market_signals.items()})
        
        # Order book signals (15 points max)
        orderbook_score, ob_signals = self._orderbook_score(metrics.get('order_book', {}))
        signals.update({k: signals[k] + v for k, v in ob_signals.items()})
        
        # Momentum signals (15 points max)
        momentum_score, mom_signals = self._momentum_score(ohlcv)
        signals.update({k: signals[k] + v for k, v in mom_signals.items()})
        
        total_score = tech_score + risk_score + market_score + orderbook_score + momentum_score
        total_score = max(0, min(100, total_score))
        
        # Calculate price targets
        targets = self._calculate_targets(ohlcv)
        
        # Risk assessment
        risk_level = self._assess_risk(metrics, ohlcv)
        
        return {
            'total_score': total_score,
            'recommendation': self._get_recommendation(total_score),
            'risk_level': risk_level,
            'signals': signals,
            'breakdown': {
                'technical': tech_score,
                'risk': risk_score,
                'market': market_score,
                'orderbook': orderbook_score,
                'momentum': momentum_score
            },
            'targets': targets,
            'indicators': self._get_current_indicators(ohlcv)
        }
    
    def _technical_score(self, ohlcv: pd.DataFrame) -> Tuple[float, Dict[str, List[str]]]:
        """Score based on technical indicators (max 30 points)."""
        score = 15  # Start neutral
        signals = {'bullish': [], 'bearish': [], 'neutral': []}
        
        if len(ohlcv) < 20:
            return score, signals
        
        current = ohlcv['close'].iloc[-1]
        
        # Moving Average Analysis
        ma20 = ohlcv['close'].rolling(20).mean().iloc[-1]
        if len(ohlcv) >= 50:
            ma50 = ohlcv['close'].rolling(50).mean().iloc[-1]
            
            if ma20 > ma50 and current > ma20:
                score += 6
                signals['bullish'].append("Golden Cross: MA20 > MA50, price above both")
            elif ma20 < ma50 and current < ma20:
                score -= 6
                signals['bearish'].append("Death Cross: MA20 < MA50, price below both")
            elif current > ma20:
                score += 3
                signals['bullish'].append("Price above MA20")
        
        # RSI Analysis
        rsi = TechnicalIndicators.calculate_rsi(ohlcv['close']).iloc[-1]
        if not np.isnan(rsi):
            if rsi < 30:
                score += 5
                signals['bullish'].append(f"RSI oversold at {rsi:.1f}")
            elif rsi > 70:
                score -= 5
                signals['bearish'].append(f"RSI overbought at {rsi:.1f}")
            elif 40 < rsi < 60:
                score += 2
                signals['neutral'].append(f"RSI neutral at {rsi:.1f}")
        
        # MACD Analysis
        macd = TechnicalIndicators.calculate_macd(ohlcv['close'])
        if not np.isnan(macd['histogram'].iloc[-1]):
            hist_now = macd['histogram'].iloc[-1]
            hist_prev = macd['histogram'].iloc[-2] if len(macd['histogram']) > 1 else 0
            
            if hist_now > 0 and hist_now > hist_prev:
                score += 4
                signals['bullish'].append("MACD bullish momentum increasing")
            elif hist_now < 0 and hist_now < hist_prev:
                score -= 4
                signals['bearish'].append("MACD bearish momentum increasing")
        
        # Bollinger Bands Analysis
        bb = TechnicalIndicators.calculate_bollinger_bands(ohlcv['close'])
        if not np.isnan(bb['lower'].iloc[-1]):
            if current <= bb['lower'].iloc[-1]:
                score += 3
                signals['bullish'].append("Price at lower Bollinger Band (potential bounce)")
            elif current >= bb['upper'].iloc[-1]:
                score -= 3
                signals['bearish'].append("Price at upper Bollinger Band (potential pullback)")
        
        return max(0, min(30, score)), signals
    
    def _risk_score(self, vol_metrics: Dict) -> Tuple[float, Dict[str, List[str]]]:
        """Score based on risk metrics (max 20 points)."""
        score = 10  # Start neutral
        signals = {'bullish': [], 'bearish': [], 'neutral': []}
        
        if not vol_metrics:
            return score, signals
        
        # Volatility assessment
        vol = vol_metrics.get('historical_vol_annualized', 50)
        if vol > 100:
            score -= 6
            signals['bearish'].append(f"Extreme volatility: {vol:.1f}% annualized")
        elif vol > 70:
            score -= 3
            signals['bearish'].append(f"High volatility: {vol:.1f}% annualized")
        elif vol < 30:
            score += 4
            signals['bullish'].append(f"Low volatility: {vol:.1f}% annualized")
        else:
            signals['neutral'].append(f"Moderate volatility: {vol:.1f}% annualized")
        
        # Drawdown assessment
        dd = abs(vol_metrics.get('max_drawdown', 0))
        if dd > 40:
            score -= 5
            signals['bearish'].append(f"Severe drawdown: {dd:.1f}%")
        elif dd > 20:
            score -= 2
            signals['bearish'].append(f"Significant drawdown: {dd:.1f}%")
        elif dd < 10:
            score += 3
            signals['bullish'].append(f"Limited drawdown: {dd:.1f}%")
        
        # Sharpe ratio assessment
        sharpe = vol_metrics.get('sharpe_ratio', 0)
        if sharpe > 2:
            score += 5
            signals['bullish'].append(f"Excellent risk-adjusted returns: Sharpe {sharpe:.2f}")
        elif sharpe > 1:
            score += 3
            signals['bullish'].append(f"Good risk-adjusted returns: Sharpe {sharpe:.2f}")
        elif sharpe < -1:
            score -= 4
            signals['bearish'].append(f"Poor risk-adjusted returns: Sharpe {sharpe:.2f}")
        
        return max(0, min(20, score)), signals
    
    def _market_score(self, market_metrics: Dict) -> Tuple[float, Dict[str, List[str]]]:
        """Score based on market metrics (max 20 points)."""
        score = 10  # Start neutral
        signals = {'bullish': [], 'bearish': [], 'neutral': []}
        
        if not market_metrics:
            return score, signals
        
        change_24h = market_metrics.get('price_change_24h', 0)
        if change_24h > 10:
            score += 4
            signals['bullish'].append(f"Strong 24h gain: +{change_24h:.1f}%")
        elif change_24h > 3:
            score += 2
            signals['bullish'].append(f"Positive 24h: +{change_24h:.1f}%")
        elif change_24h < -10:
            score -= 4
            signals['bearish'].append(f"Sharp 24h drop: {change_24h:.1f}%")
        elif change_24h < -3:
            score -= 2
            signals['bearish'].append(f"Negative 24h: {change_24h:.1f}%")
        
        change_7d = market_metrics.get('price_change_7d', 0)
        if change_7d and change_7d > 20:
            score += 3
            signals['bullish'].append(f"Strong weekly gain: +{change_7d:.1f}%")
        elif change_7d and change_7d < -20:
            score -= 3
            signals['bearish'].append(f"Sharp weekly drop: {change_7d:.1f}%")
        
        market_cap = market_metrics.get('market_cap', 0)
        if market_cap > 10_000_000_000:  # > $10B
            score += 3
            signals['bullish'].append("Large cap: More stable and liquid")
        elif market_cap < 100_000_000:  # < $100M
            score -= 2
            signals['bearish'].append("Small cap: Higher risk")
        
        return max(0, min(20, score)), signals
    
    def _orderbook_score(self, ob_metrics: Dict) -> Tuple[float, Dict[str, List[str]]]:
        score = 7.5  # Start neutral
        signals = {'bullish': [], 'bearish': [], 'neutral': []}
        
        if not ob_metrics:
            signals['neutral'].append("Order book data unavailable")
            return score, signals
        
        # Book imbalance
        imbalance = ob_metrics.get('book_imbalance', 0)
        if imbalance > 0.3:
            score += 4
            signals['bullish'].append(f"Strong bid pressure: {imbalance:.2f}")
        elif imbalance > 0.1:
            score += 2
            signals['bullish'].append(f"Moderate bid pressure: {imbalance:.2f}")
        elif imbalance < -0.3:
            score -= 4
            signals['bearish'].append(f"Strong ask pressure: {imbalance:.2f}")
        elif imbalance < -0.1:
            score -= 2
            signals['bearish'].append(f"Moderate ask pressure: {imbalance:.2f}")
        
        # Spread analysis
        spread_bps = ob_metrics.get('spread_bps', 0)
        if spread_bps < 5:
            score += 3
            signals['bullish'].append(f"Tight spread: {spread_bps:.1f} bps (highly liquid)")
        elif spread_bps > 50:
            score -= 3
            signals['bearish'].append(f"Wide spread: {spread_bps:.1f} bps (illiquid)")
        
        return max(0, min(15, score)), signals
    
    def _momentum_score(self, ohlcv: pd.DataFrame) -> Tuple[float, Dict[str, List[str]]]:
        """Score based on momentum (max 15 points)."""
        score = 7.5  # Start neutral
        signals = {'bullish': [], 'bearish': [], 'neutral': []}
        
        if len(ohlcv) < 10:
            return score, signals
        
        returns = ohlcv['close'].pct_change()
        
        # Recent trend (last 10 periods)
        recent_return = returns.tail(10).sum()
        if recent_return > 0.05:
            score += 4
            signals['bullish'].append(f"Strong recent momentum: +{recent_return*100:.1f}%")
        elif recent_return > 0.02:
            score += 2
            signals['bullish'].append(f"Positive momentum: +{recent_return*100:.1f}%")
        elif recent_return < -0.05:
            score -= 4
            signals['bearish'].append(f"Negative momentum: {recent_return*100:.1f}%")
        elif recent_return < -0.02:
            score -= 2
            signals['bearish'].append(f"Weak momentum: {recent_return*100:.1f}%")
        
        # Trend consistency (% of positive days)
        positive_days = (returns.tail(20) > 0).sum() / 20
        if positive_days > 0.6:
            score += 3
            signals['bullish'].append(f"Consistent uptrend: {positive_days*100:.0f}% green days")
        elif positive_days < 0.4:
            score -= 3
            signals['bearish'].append(f"Consistent downtrend: {(1-positive_days)*100:.0f}% red days")
        
        # Volume trend
        if 'volume' in ohlcv.columns and ohlcv['volume'].sum() > 0:
            vol_trend = ohlcv['volume'].tail(5).mean() / ohlcv['volume'].tail(20).mean()
            if vol_trend > 1.5:
                score += 2
                signals['bullish'].append("Volume surge: Recent volume above average")
            elif vol_trend < 0.5:
                score -= 1
                signals['bearish'].append("Volume declining: Low interest")
        
        return max(0, min(15, score)), signals
    
    def _calculate_targets(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """Calculate price targets and stop loss levels."""
        current = ohlcv['close'].iloc[-1]
        
        # Support/Resistance levels
        sr_levels = TechnicalIndicators.find_support_resistance(ohlcv)
        
        # Fibonacci levels
        fib_levels = TechnicalIndicators.fibonacci_retracements(ohlcv)
        
        # ATR for stop loss calculation
        atr = TechnicalIndicators.calculate_atr(ohlcv).iloc[-1]
        if np.isnan(atr):
            atr = current * 0.03  # Default to 3% if ATR unavailable
        
        # Calculate targets
        stop_loss = sr_levels['support'][0] if sr_levels['support'] else current - (2 * atr)
        target1 = sr_levels['resistance'][0] if sr_levels['resistance'] else current + (2 * atr)
        target2 = sr_levels['resistance'][1] if len(sr_levels['resistance']) > 1 else current + (3 * atr)
        
        # Risk/Reward calculation
        risk = current - stop_loss
        reward = target1 - current
        rr_ratio = reward / risk if risk > 0 else 0
        
        return {
            'current_price': current,
            'stop_loss': stop_loss,
            'target_1': target1,
            'target_2': target2,
            'risk_reward_ratio': rr_ratio,
            'support_levels': sr_levels['support'],
            'resistance_levels': sr_levels['resistance'],
            'fibonacci': fib_levels,
            'atr': atr
        }
    
    def _assess_risk(self, metrics: Dict, ohlcv: pd.DataFrame) -> str:
        """Comprehensive risk assessment."""
        risk_points = 0
        
        # Volatility risk
        vol = metrics.get('volatility', {}).get('historical_vol_annualized', 50)
        if vol > 100:
            risk_points += 3
        elif vol > 60:
            risk_points += 2
        elif vol > 40:
            risk_points += 1
        
        # Drawdown risk
        dd = abs(metrics.get('volatility', {}).get('max_drawdown', 0))
        if dd > 40:
            risk_points += 2
        elif dd > 20:
            risk_points += 1
        
        # Market cap risk
        mcap = metrics.get('market', {}).get('market_cap', 0)
        if mcap < 100_000_000:  # < $100M
            risk_points += 2
        elif mcap < 1_000_000_000:  # < $1B
            risk_points += 1
        
        # Liquidity risk
        spread = metrics.get('order_book', {}).get('spread_bps', 0)
        if spread > 50:
            risk_points += 2
        elif spread > 20:
            risk_points += 1
        
        if risk_points >= 6:
            return 'EXTREME'
        elif risk_points >= 4:
            return 'HIGH'
        elif risk_points >= 2:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _get_recommendation(self, score: float) -> str:
        """Convert score to investment recommendation."""
        if score >= 75:
            return "STRONG BUY"
        elif score >= 60:
            return "BUY"
        elif score >= 45:
            return "HOLD"
        elif score >= 30:
            return "CAUTION"
        else:
            return "AVOID"
    
    def _get_current_indicators(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """Get current values of all technical indicators."""
        indicators = {}
        
        if len(ohlcv) >= 14:
            rsi = TechnicalIndicators.calculate_rsi(ohlcv['close']).iloc[-1]
            indicators['rsi'] = rsi if not np.isnan(rsi) else None
        
        if len(ohlcv) >= 26:
            macd = TechnicalIndicators.calculate_macd(ohlcv['close'])
            indicators['macd'] = macd['macd'].iloc[-1] if not np.isnan(macd['macd'].iloc[-1]) else None
            indicators['macd_signal'] = macd['signal'].iloc[-1] if not np.isnan(macd['signal'].iloc[-1]) else None
        
        if len(ohlcv) >= 20:
            bb = TechnicalIndicators.calculate_bollinger_bands(ohlcv['close'])
            indicators['bb_upper'] = bb['upper'].iloc[-1]
            indicators['bb_lower'] = bb['lower'].iloc[-1]
            
            stoch = TechnicalIndicators.calculate_stochastic(ohlcv)
            indicators['stoch_k'] = stoch['k'].iloc[-1] if not np.isnan(stoch['k'].iloc[-1]) else None
        
        return indicators
    
    def recommend_position_size(self, score: float, risk_level: str, 
                                 portfolio_value: float = 10000) -> Dict[str, Any]:
        """Suggest how much to invest based on score and risk."""
        # Base allocation by score
        if score >= 75:
            base_alloc = 0.10  # 10%
        elif score >= 60:
            base_alloc = 0.07  # 7%
        elif score >= 45:
            base_alloc = 0.05  # 5%
        elif score >= 30:
            base_alloc = 0.02  # 2%
        else:
            base_alloc = 0.0  # Avoid
        
        # Adjust by risk level
        risk_multipliers = {'LOW': 1.2, 'MEDIUM': 1.0, 'HIGH': 0.6, 'EXTREME': 0.3}
        adjusted_alloc = base_alloc * risk_multipliers.get(risk_level, 1.0)
        
        # Cap at 15% max per position
        final_alloc = min(adjusted_alloc, 0.15)
        
        return {
            'recommended_allocation': f"{final_alloc * 100:.1f}%",
            'dollar_amount': portfolio_value * final_alloc,
            'max_loss_amount': portfolio_value * final_alloc * 0.1,  # Assuming 10% stop loss
            'rationale': self._position_rationale(score, risk_level, final_alloc)
        }
    
    def _position_rationale(self, score: float, risk_level: str, alloc: float) -> str:
        """Generate rationale for position sizing."""
        if alloc == 0:
            return "Score too low - avoid this position"
        elif risk_level == 'EXTREME':
            return f"High risk limits position size despite score of {score:.0f}"
        elif score >= 70:
            return f"Strong signals justify larger position with {risk_level.lower()} risk"
        else:
            return f"Moderate signals with {risk_level.lower()} risk - standard sizing"
    
    def compare_to_bitcoin(self, ohlcv: pd.DataFrame, btc_ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """Compare coin performance vs Bitcoin."""
        if len(ohlcv) < 10 or len(btc_ohlcv) < 10:
            return {}
        
        # Align data
        min_len = min(len(ohlcv), len(btc_ohlcv))
        coin_returns = ohlcv['close'].pct_change().tail(min_len).dropna()
        btc_returns = btc_ohlcv['close'].pct_change().tail(min_len).dropna()
        
        # Align by taking minimum length
        common_len = min(len(coin_returns), len(btc_returns))
        coin_returns = coin_returns.tail(common_len)
        btc_returns = btc_returns.tail(common_len)
        
        # Beta calculation
        covariance = coin_returns.cov(btc_returns)
        btc_variance = btc_returns.var()
        beta = covariance / btc_variance if btc_variance > 0 else 1
        
        # Total returns
        coin_total = (1 + coin_returns).prod() - 1
        btc_total = (1 + btc_returns).prod() - 1
        
        # Correlation
        correlation = coin_returns.corr(btc_returns)
        
        return {
            'beta': beta,
            'correlation': correlation,
            'coin_return': coin_total * 100,
            'btc_return': btc_total * 100,
            'outperformance': (coin_total - btc_total) * 100,
            'interpretation': self._interpret_beta(beta, coin_total - btc_total)
        }
    
    def _interpret_beta(self, beta: float, outperformance: float) -> str:
        """Interpret beta and outperformance."""
        if beta > 1.5:
            risk_profile = "Very high risk (amplified BTC moves)"
        elif beta > 1:
            risk_profile = "Higher risk than BTC"
        elif beta > 0.5:
            risk_profile = "Moderate correlation with BTC"
        else:
            risk_profile = "Low BTC correlation (unique drivers)"
        
        perf = "Outperforming" if outperformance > 0 else "Underperforming"
        return f"{perf} BTC by {abs(outperformance):.1f}%. {risk_profile}"


# ============================================================================
# RETURN PREDICTOR
# ============================================================================

class ReturnPredictor:
    """Build models to predict returns from microstructure features."""
    
    @staticmethod
    def prepare_features(ohlcv: pd.DataFrame, book_imbalance: float = 0, flow_imbalance: float = 0) -> pd.DataFrame:
        """Prepare feature matrix."""
        df = ohlcv.copy()
        
        df['returns'] = df['close'].pct_change()
        df['returns_next'] = df['returns'].shift(-1)
        
        for lag in [1, 2, 3, 5]:
            df[f'returns_lag{lag}'] = df['returns'].shift(lag)
        
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_10'] = df['returns'].rolling(10).std()
        
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        
        if 'volume' in df.columns and df['volume'].sum() > 0:
            df['log_volume'] = np.log1p(df['volume'])
            df['volume_ma5'] = df['volume'].rolling(5).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma5']
        
        df['range'] = (df['high'] - df['low']) / df['close']
        df['book_imbalance'] = book_imbalance
        df['flow_imbalance'] = flow_imbalance
        
        return df
    
    @staticmethod
    def fit_model(df: pd.DataFrame) -> Dict[str, Any]:
        """Fit OLS model."""
        try:
            import statsmodels.api as sm
        except ImportError:
            return {'error': 'statsmodels not installed'}
        
        feature_cols = [
            'returns_lag1', 'returns_lag2', 'returns_lag3',
            'volatility_5', 'momentum_5', 'momentum_10',
            'book_imbalance', 'flow_imbalance'
        ]
        
        if 'log_volume' in df.columns:
            feature_cols.extend(['log_volume', 'volume_ratio'])
        
        available = [c for c in feature_cols if c in df.columns]
        df_clean = df.dropna(subset=['returns_next'] + available)
        
        if len(df_clean) < 20:
            return {'error': f'Insufficient data: {len(df_clean)} observations'}
        
        y = df_clean['returns_next'] * 100
        X = sm.add_constant(df_clean[available])
        model = sm.OLS(y, X).fit(cov_type='HC1')
        
        return {
            'model': model,
            'n_obs': len(df_clean),
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'features': available,
            'coefficients': dict(zip(model.params.index, model.params.values)),
            'pvalues': dict(zip(model.pvalues.index, model.pvalues.values)),
            'significant': [f for f, p in zip(model.params.index, model.pvalues) if p < 0.05 and f != 'const']
        }


# ============================================================================
# VISUALIZER
# ============================================================================

class Visualizer:
    """Generate analysis visualizations."""
    
    COLORS = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']
    
    @classmethod
    def create_all(cls, symbol: str, ohlcv: pd.DataFrame, order_book: Dict,
                   metrics: Dict, trades: List[Dict]) -> List[Tuple[str, plt.Figure]]:
        """Generate all visualizations."""
        figures = [
            cls._price_chart(symbol, ohlcv),
            cls._volume_analysis(symbol, ohlcv),
            cls._volatility_chart(symbol, ohlcv),
        ]
        
        if order_book.get('bids'):
            figures.append(cls._order_book_depth(symbol, order_book))
        
        if trades:
            figures.append(cls._trade_flow_chart(symbol, trades))
        
        return figures
    
    @classmethod
    def _price_chart(cls, symbol: str, ohlcv: pd.DataFrame) -> Tuple[str, plt.Figure]:
        """Price chart with moving averages."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
        
        ax1 = axes[0]
        ax1.plot(ohlcv.index, ohlcv['close'], label='Close', linewidth=1.5, color=cls.COLORS[2])
        
        if len(ohlcv) >= 20:
            ax1.plot(ohlcv.index, ohlcv['close'].rolling(20).mean(), label='MA20', linewidth=1, alpha=0.8, color=cls.COLORS[3])
        if len(ohlcv) >= 50:
            ax1.plot(ohlcv.index, ohlcv['close'].rolling(50).mean(), label='MA50', linewidth=1, alpha=0.8, color=cls.COLORS[4])
        
        ax1.fill_between(ohlcv.index, ohlcv['low'], ohlcv['high'], alpha=0.2, color=cls.COLORS[2])
        ax1.set_title(f'{symbol} Price Chart', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (USD)')
        ax1.legend(loc='upper left')
        ax1.grid(alpha=0.3)
        
        ax2 = axes[1]
        if 'volume' in ohlcv.columns and ohlcv['volume'].sum() > 0:
            colors = [cls.COLORS[0] if c >= o else cls.COLORS[1] for c, o in zip(ohlcv['close'], ohlcv['open'])]
            ax2.bar(ohlcv.index, ohlcv['volume'], color=colors, alpha=0.7)
            ax2.set_ylabel('Volume')
        ax2.set_xlabel('Date')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        return ('price_chart', fig)
    
    @classmethod
    def _order_book_depth(cls, symbol: str, order_book: Dict) -> Tuple[str, plt.Figure]:
        """Order book depth visualization."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bids = np.array(order_book['bids'])
        asks = np.array(order_book['asks'])
        
        bid_cum = np.cumsum(bids[:, 1])
        ask_cum = np.cumsum(asks[:, 1])
        
        ax.fill_between(bids[:, 0], bid_cum, alpha=0.5, color=cls.COLORS[0], label='Bids')
        ax.fill_between(asks[:, 0], ask_cum, alpha=0.5, color=cls.COLORS[1], label='Asks')
        ax.plot(bids[:, 0], bid_cum, color=cls.COLORS[0], linewidth=2)
        ax.plot(asks[:, 0], ask_cum, color=cls.COLORS[1], linewidth=2)
        
        mid = (bids[0][0] + asks[0][0]) / 2
        ax.axvline(mid, color='gray', linestyle='--', alpha=0.7, label=f'Mid: ${mid:,.2f}')
        
        ax.set_title(f'{symbol} Order Book Depth', fontsize=14, fontweight='bold')
        ax.set_xlabel('Price (USD)')
        ax.set_ylabel('Cumulative Volume')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return ('order_book_depth', fig)
    
    @classmethod
    def _volume_analysis(cls, symbol: str, ohlcv: pd.DataFrame) -> Tuple[str, plt.Figure]:
        """Volume and returns analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        returns = ohlcv['close'].pct_change().dropna()
        
        ax1 = axes[0, 0]
        ax1.hist(returns * 100, bins=50, color=cls.COLORS[2], alpha=0.7, edgecolor='white')
        ax1.axvline(returns.mean() * 100, color='red', linestyle='--', label=f'Mean: {returns.mean()*100:.3f}%')
        ax1.set_title('Returns Distribution', fontweight='bold')
        ax1.set_xlabel('Return (%)')
        ax1.legend()
        
        ax2 = axes[0, 1]
        if len(returns) >= 20:
            vol_20 = returns.rolling(20).std() * np.sqrt(365 * 24) * 100
            ax2.plot(vol_20.index, vol_20, color=cls.COLORS[4], linewidth=1.5)
        ax2.set_title('Rolling 20-Period Annualized Volatility', fontweight='bold')
        ax2.set_ylabel('Volatility (%)')
        ax2.grid(alpha=0.3)
        
        ax3 = axes[1, 0]
        if 'volume' in ohlcv.columns and ohlcv['volume'].sum() > 0:
            ax3.plot(ohlcv.index, ohlcv['volume'], color=cls.COLORS[3], linewidth=1)
            if len(ohlcv) >= 20:
                ax3.plot(ohlcv.index, ohlcv['volume'].rolling(20).mean(), color='red', linewidth=1.5, label='MA20')
                ax3.legend()
        ax3.set_title('Volume Over Time', fontweight='bold')
        ax3.grid(alpha=0.3)
        
        ax4 = axes[1, 1]
        cum_returns = (1 + returns).cumprod() - 1
        ax4.plot(cum_returns.index, cum_returns * 100, color=cls.COLORS[0], linewidth=1.5)
        ax4.fill_between(cum_returns.index, cum_returns * 100, alpha=0.3, color=cls.COLORS[0])
        ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax4.set_title('Cumulative Returns', fontweight='bold')
        ax4.set_ylabel('Return (%)')
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        return ('volume_analysis', fig)
    
    @classmethod
    def _volatility_chart(cls, symbol: str, ohlcv: pd.DataFrame) -> Tuple[str, plt.Figure]:
        """Volatility analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        returns = ohlcv['close'].pct_change().dropna()
        
        ax1 = axes[0]
        for i, w in enumerate([5, 10, 20]):
            if len(returns) >= w:
                vol = returns.rolling(w).std() * 100
                ax1.plot(vol.index, vol, label=f'{w}-period', linewidth=1.5, color=cls.COLORS[i])
        ax1.set_title('Rolling Volatility', fontweight='bold')
        ax1.set_ylabel('Volatility (%)')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        ax2 = axes[1]
        hl_range = (ohlcv['high'] - ohlcv['low']) / ohlcv['close'] * 100
        ax2.plot(ohlcv.index, hl_range, color=cls.COLORS[3], linewidth=1)
        if len(ohlcv) >= 10:
            ax2.plot(ohlcv.index, hl_range.rolling(10).mean(), color='red', linewidth=1.5, label='MA10')
            ax2.legend()
        ax2.set_title('High-Low Range (% of Close)', fontweight='bold')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        return ('volatility_analysis', fig)
    
    @classmethod
    def _trade_flow_chart(cls, symbol: str, trades: List[Dict]) -> Tuple[str, plt.Figure]:
        """Trade flow visualization."""
        df = pd.DataFrame(trades)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1 = axes[0]
        buy_vol = df[df['side'] == 'buy']['amount'].sum()
        sell_vol = df[df['side'] == 'sell']['amount'].sum()
        bars = ax1.bar(['Buy', 'Sell'], [buy_vol, sell_vol], color=[cls.COLORS[0], cls.COLORS[1]], alpha=0.8)
        ax1.set_title('Buy vs Sell Volume', fontweight='bold')
        for bar, val in zip(bars, [buy_vol, sell_vol]):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:,.2f}', ha='center', va='bottom')
        
        ax2 = axes[1]
        ax2.hist(df['amount'], bins=30, color=cls.COLORS[2], alpha=0.7, edgecolor='white')
        ax2.axvline(df['amount'].mean(), color='red', linestyle='--', label=f'Mean: {df["amount"].mean():.4f}')
        ax2.set_title('Trade Size Distribution', fontweight='bold')
        ax2.legend()
        
        plt.tight_layout()
        return ('trade_flow', fig)


# ============================================================================
# MAIN ANALYZER
# ============================================================================

class CryptoAnalyzer:
    """Main analyzer - orchestrates data fetching, metrics, and visualization."""
    
    def __init__(self):
        self.fetcher = CryptoDataFetcher()
    
    def search(self, query: str) -> List[Dict]:
        """Search for a cryptocurrency."""
        print(f"\nSearching for '{query}'...")
        results = self.fetcher.search_coin(query)
        if not results:
            print("No results found.")
            return []
        
        print(f"\nFound {len(results)} results:")
        for i, coin in enumerate(results):
            print(f"  {i+1}. {coin['name']} ({coin['symbol'].upper()}) - ID: {coin['id']}")
        return results
    
    def analyze(self, coin_id: str, binance_symbol: Optional[str] = None,
                days: int = 30, save_plots: bool = True) -> AnalysisResult:
        """
        Run full analysis on a cryptocurrency.
        
        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin', 'ethereum', 'solana')
            binance_symbol: Binance trading pair (e.g., 'BTCUSDT'). Auto-detected if None.
            days: Days of historical data
            save_plots: Whether to save plots to files
        """
        print(f"\n{'='*70}")
        print(f"ANALYZING: {coin_id.upper()}")
        print(f"{'='*70}")
        
        # Market data from CoinGecko
        print("\nFetching market data...")
        market_data = self.fetcher.get_market_data(coin_id)
        symbol = market_data['symbol'].upper()
        price = market_data['market_data']['current_price']['usd']
        print(f"  {market_data['name']} ({symbol})")
        print(f"  Price: ${price:,.2f}")
        print(f"  Market Cap Rank: #{market_data.get('market_cap_rank', 'N/A')}")
        
        if binance_symbol is None:
            binance_symbol = f"{symbol}USDT"
        
        # OHLCV data
        print(f"\nFetching {days} days of price data...")
        try:
            ohlcv = self.fetcher.get_binance_klines(binance_symbol, '1h', min(days * 24, 1000))
            print(f"  Got {len(ohlcv)} hourly candles from Binance")
        except Exception as e:
            print(f"  Binance failed ({e}), using CoinGecko...")
            ohlcv = self.fetcher.get_coingecko_ohlc(coin_id, days)
            print(f"  Got {len(ohlcv)} candles from CoinGecko")
        
        # Order book
        print("\nFetching order book...")
        try:
            order_book = self.fetcher.get_binance_order_book(binance_symbol, 100)
            print(f"  Got {len(order_book['bids'])} bid levels, {len(order_book['asks'])} ask levels")
        except Exception as e:
            print(f"  Could not fetch order book: {e}")
            order_book = {'bids': [], 'asks': [], 'timestamp': datetime.now()}
        
        # Recent trades
        print("\nFetching recent trades...")
        try:
            trades = self.fetcher.get_binance_trades(binance_symbol, 500)
            print(f"  Got {len(trades)} recent trades")
        except Exception as e:
            print(f"  Could not fetch trades: {e}")
            trades = []
        
        # Calculate metrics
        print("\nCalculating metrics...")
        metrics = {
            'market': {
                'name': market_data['name'],
                'symbol': symbol,
                'price': price,
                'market_cap': market_data['market_data']['market_cap']['usd'],
                'volume_24h': market_data['market_data']['total_volume']['usd'],
                'price_change_24h': market_data['market_data']['price_change_percentage_24h'],
                'price_change_7d': market_data['market_data'].get('price_change_percentage_7d'),
            }
        }
        
        if order_book['bids']:
            metrics['order_book'] = MetricsCalculator.order_book_metrics(order_book)
            sizes = [0.1, 0.5, 1, 5, 10, 50]
            metrics['price_impact'] = MetricsCalculator.price_impact(order_book, sizes)
        
        metrics['volatility'] = MetricsCalculator.volatility_metrics(ohlcv)
        
        if trades:
            metrics['trade_flow'] = MetricsCalculator.trade_flow_metrics(trades)
        
        # Build prediction model
        print("\nBuilding prediction model...")
        book_imb = metrics.get('order_book', {}).get('book_imbalance', 0)
        flow_imb = metrics.get('trade_flow', {}).get('flow_imbalance', 0)
        feature_df = ReturnPredictor.prepare_features(ohlcv, book_imb, flow_imb)
        model_results = ReturnPredictor.fit_model(feature_df)
        
        if 'error' not in model_results:
            print(f"  R-squared: {model_results['r_squared']:.4f}")
            print(f"  Observations: {model_results['n_obs']}")
            if model_results['significant']:
                print(f"  Significant features: {', '.join(model_results['significant'])}")
        else:
            print(f"  {model_results['error']}")
        
        # Investment Scoring
        print("\nCalculating investment score...")
        scorer = InvestmentScorer(self.fetcher)
        investment = scorer.calculate_score(metrics, ohlcv)
        position = scorer.recommend_position_size(
            investment['total_score'], 
            investment['risk_level'],
            portfolio_value=10000
        )
        print(f"  Score: {investment['total_score']:.0f}/100")
        print(f"  Recommendation: {investment['recommendation']}")
        print(f"  Risk Level: {investment['risk_level']}")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        figures = Visualizer.create_all(f"{symbol}/USD", ohlcv, order_book, metrics, trades)
        
        if save_plots:
            for name, fig in figures:
                filename = f"{coin_id}_{name}.png"
                fig.savefig(filename, dpi=150, bbox_inches='tight')
                print(f"  Saved: {filename}")
        
        self._print_summary(metrics, model_results)
        self._print_investment_report(metrics, investment, position, ohlcv)
        
        # Store investment data in result
        result = AnalysisResult(
            symbol=symbol,
            timestamp=datetime.now(),
            price_data=ohlcv,
            order_book=order_book,
            metrics=metrics,
            model_results=model_results,
            figures=figures
        )
        result.investment = investment
        result.position = position
        
        return result
    
    def _print_summary(self, metrics: Dict, model_results: Dict) -> None:
        """Print analysis summary."""
        print(f"\n{'='*70}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*70}")
        
        m = metrics['market']
        print(f"\nMarket Overview:")
        print(f"  Price: ${m['price']:,.2f}")
        print(f"  24h Change: {m['price_change_24h']:.2f}%")
        print(f"  24h Volume: ${m['volume_24h']:,.0f}")
        print(f"  Market Cap: ${m['market_cap']:,.0f}")
        
        if 'order_book' in metrics:
            ob = metrics['order_book']
            print(f"\nOrder Book:")
            print(f"  Spread: {ob['spread_bps']:.2f} bps (${ob['spread']:.4f})")
            print(f"  Book Imbalance: {ob['book_imbalance']:.3f} ({'bullish' if ob['book_imbalance'] > 0 else 'bearish'})")
            print(f"  Bid Volume: {ob['bid_volume']:,.2f}")
            print(f"  Ask Volume: {ob['ask_volume']:,.2f}")
        
        if 'volatility' in metrics:
            v = metrics['volatility']
            print(f"\nVolatility:")
            print(f"  Annualized (Historical): {v['historical_vol_annualized']:.2f}%")
            print(f"  Annualized (Parkinson): {v['parkinson_vol_annualized']:.2f}%")
            print(f"  Max Drawdown: {v['max_drawdown']:.2f}%")
            print(f"  Sharpe Ratio: {v['sharpe_ratio']:.3f}")
        
        if 'trade_flow' in metrics:
            tf = metrics['trade_flow']
            print(f"\nTrade Flow:")
            print(f"  Recent Trades: {tf['trade_count']}")
            print(f"  Flow Imbalance: {tf['flow_imbalance']:.3f} ({'buying pressure' if tf['flow_imbalance'] > 0 else 'selling pressure'})")
            print(f"  VWAP: ${tf['vwap']:,.2f}")
        
        if 'price_impact' in metrics:
            print(f"\nPrice Impact (estimated slippage):")
            for size, impact in list(metrics['price_impact']['buy'].items())[:4]:
                if impact is not None:
                    print(f"  Buy {size:,.1f} units: {impact:.4f}%")
    
    def _print_investment_report(self, metrics: Dict, investment: Dict, 
                                  position: Dict, ohlcv: pd.DataFrame) -> None:
        """Print comprehensive investment recommendation report."""
        print(f"\n{'='*70}")
        print("INVESTMENT RECOMMENDATION")
        print(f"{'='*70}")
        
        # Main recommendation
        rec = investment['recommendation']
        score = investment['total_score']
        risk = investment['risk_level']
        
        # Color coding for recommendation (using text indicators)
        rec_emoji = {
            'STRONG BUY': '🟢🟢',
            'BUY': '🟢',
            'HOLD': '🟡',
            'CAUTION': '🟠',
            'AVOID': '🔴'
        }.get(rec, '')
        
        print(f"\n  {rec_emoji} RECOMMENDATION: {rec}")
        print(f"  INVESTMENT SCORE: {score:.0f}/100")
        print(f"  RISK LEVEL: {risk}")
        
        # Score breakdown
        breakdown = investment['breakdown']
        print(f"\n  Score Breakdown:")
        print(f"    Technical Analysis: {breakdown['technical']:.1f}/30")
        print(f"    Risk Assessment:    {breakdown['risk']:.1f}/20")
        print(f"    Market Strength:    {breakdown['market']:.1f}/20")
        print(f"    Order Book:         {breakdown['orderbook']:.1f}/15")
        print(f"    Momentum:           {breakdown['momentum']:.1f}/15")
        
        # Signals
        signals = investment['signals']
        if signals['bullish']:
            print(f"\n  ✓ BULLISH SIGNALS:")
            for sig in signals['bullish'][:5]:
                print(f"    + {sig}")
        
        if signals['bearish']:
            print(f"\n  ✗ BEARISH SIGNALS:")
            for sig in signals['bearish'][:5]:
                print(f"    - {sig}")
        
        # Technical Indicators
        indicators = investment.get('indicators', {})
        if indicators:
            print(f"\n  Technical Indicators:")
            if indicators.get('rsi') is not None:
                rsi = indicators['rsi']
                rsi_status = 'Oversold' if rsi < 30 else ('Overbought' if rsi > 70 else 'Neutral')
                print(f"    RSI(14): {rsi:.1f} ({rsi_status})")
            if indicators.get('macd') is not None:
                macd_signal = 'Bullish' if indicators['macd'] > (indicators.get('macd_signal') or 0) else 'Bearish'
                print(f"    MACD: {indicators['macd']:.4f} ({macd_signal})")
            if indicators.get('stoch_k') is not None:
                stoch = indicators['stoch_k']
                stoch_status = 'Oversold' if stoch < 20 else ('Overbought' if stoch > 80 else 'Neutral')
                print(f"    Stochastic: {stoch:.1f} ({stoch_status})")
        
        # Price Targets
        targets = investment['targets']
        print(f"\n  Price Targets:")
        print(f"    Current:    ${targets['current_price']:,.2f}")
        print(f"    Stop Loss:  ${targets['stop_loss']:,.2f} ({((targets['stop_loss']/targets['current_price'])-1)*100:+.1f}%)")
        print(f"    Target 1:   ${targets['target_1']:,.2f} ({((targets['target_1']/targets['current_price'])-1)*100:+.1f}%)")
        print(f"    Target 2:   ${targets['target_2']:,.2f} ({((targets['target_2']/targets['current_price'])-1)*100:+.1f}%)")
        print(f"    Risk/Reward: {targets['risk_reward_ratio']:.2f}:1")
        
        # Position Sizing
        print(f"\n  Position Sizing (for $10,000 portfolio):")
        print(f"    Recommended: {position['recommended_allocation']}")
        print(f"    Dollar Amount: ${position['dollar_amount']:,.2f}")
        print(f"    Max Risk: ${position['max_loss_amount']:,.2f}")
        print(f"    Rationale: {position['rationale']}")
        
        # Support/Resistance
        if targets['support_levels']:
            print(f"\n  Support Levels: {', '.join([f'${s:,.2f}' for s in targets['support_levels']])}")
        if targets['resistance_levels']:
            print(f"  Resistance Levels: {', '.join([f'${r:,.2f}' for r in targets['resistance_levels']])}")
        
        # Bottom line
        print(f"\n  {'─'*60}")
        self._print_bottom_line(rec, score, risk, targets, signals)
    def _print_bottom_line(self, rec: str, score: float, risk: str, 
                           targets: Dict, signals: Dict) -> None:
        """Print the bottom line summary."""
        print("  BOTTOM LINE:")
        
        bullish_count = len(signals['bullish'])
        bearish_count = len(signals['bearish'])
        
        if rec in ['STRONG BUY', 'BUY']:
            print(f"    This coin shows {bullish_count} bullish signals vs {bearish_count} bearish.")
            print(f"    Consider entering at current levels with stop loss at ${targets['stop_loss']:,.2f}.")
            if targets['risk_reward_ratio'] > 2:
                print(f"    Risk/reward of {targets['risk_reward_ratio']:.1f}:1 is favorable.")
            print(f"    Target: ${targets['target_1']:,.2f} for first profit-taking.")
        elif rec == 'HOLD':
            print(f"    Mixed signals ({bullish_count} bullish, {bearish_count} bearish).")
            print("    Wait for clearer setup before adding to position.")
            print("    If already holding, maintain position with stops in place.")
        elif rec == 'CAUTION':
            print(f"    Warning: {bearish_count} bearish signals detected.")
            print("    Reduce exposure or tighten stop losses.")
            print("    Wait for conditions to improve before adding.")
        else:  # AVOID
            print(f"    High risk with {bearish_count} bearish signals and {risk} risk level.")
            print("    Avoid new positions at this time.")
            print("    If holding, consider taking profits or cutting losses.")


# ============================================================================
# CLI
# ============================================================================

def interactive_mode(analyzer: CryptoAnalyzer) -> Optional[AnalysisResult]:
    """Run the analyzer in interactive mode."""
    print("\n" + "="*70)
    print("INTERACTIVE CRYPTOCURRENCY ANALYZER")
    print("="*70)
    print("\nEnter a coin name or symbol to analyze (e.g., 'bitcoin', 'eth', 'solana').")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("Enter coin name: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            return None
        
        if user_input in ['quit', 'exit', 'q']:
            print("Goodbye!")
            return None
        
        if not user_input:
            print("Please enter a coin name.")
            continue
        
        # Search for the coin
        results = analyzer.fetcher.search_coin(user_input)
        
        if not results:
            print(f"No coins found matching '{user_input}'. Try again.")
            continue
        
        # If exact match or single result, use it
        coin = None
        for r in results:
            if r['id'] == user_input or r['symbol'].lower() == user_input:
                coin = r
                break
        
        if not coin:
            # Show results and let user pick
            print(f"\nFound {len(results)} results:")
            for i, c in enumerate(results[:5], 1):
                print(f"  {i}. {c['name']} ({c['symbol'].upper()}) - ID: {c['id']}")
            
            print("\nEnter the number to select, or the coin ID directly:")
            try:
                selection = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            
            if selection.isdigit():
                idx = int(selection) - 1
                if 0 <= idx < len(results):
                    coin = results[idx]
                else:
                    print("Invalid selection.")
                    continue
            else:
                # Assume it's a coin ID
                coin = {'id': selection, 'symbol': selection, 'name': selection}
        
        print(f"\nAnalyzing {coin['name']} ({coin['symbol'].upper()})...\n")
        
        try:
            result = analyzer.analyze(
                coin_id=coin['id'],
                days=7,
                save_plots=True
            )
            
            # Ask if user wants to continue
            print("\n" + "-"*40)
            another = input("\nAnalyze another coin? (y/n): ").strip().lower()
            if another not in ['y', 'yes']:
                print("Goodbye!")
                return result
                
        except Exception as e:
            print(f"\nError analyzing {coin['id']}: {e}")
            print("Try a different coin or check the coin ID.")


def main():
    parser = argparse.ArgumentParser(
        description='Cryptocurrency Market Microstructure Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python CryptoAnalysis.py                          # Interactive mode
  python CryptoAnalysis.py --coin ethereum          # Analyze Ethereum directly
  python CryptoAnalysis.py --coin solana --days 14  # Solana with 14 days of data
  python CryptoAnalysis.py --search dogecoin        # Search for a coin
        """
    )
    parser.add_argument('--coin', help='CoinGecko coin ID (e.g., bitcoin, ethereum, solana)')
    parser.add_argument('--binance', help='Binance symbol (e.g., BTCUSDT). Auto-detected if not provided.')
    parser.add_argument('--days', type=int, default=7, help='Days of historical data (default: 7)')
    parser.add_argument('--search', help='Search for a coin by name/symbol')
    parser.add_argument('--no-plots', action='store_true', help='Skip saving plot files')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    analyzer = CryptoAnalyzer()
    
    print("\n" + "="*70)
    print("CRYPTOCURRENCY MARKET MICROSTRUCTURE ANALYZER")
    print("="*70)
    print("\nData sources: CoinGecko (market data), Binance (order book, trades, OHLCV)")
    
    if args.search:
        analyzer.search(args.search)
        return
    
    # If no coin specified, run interactive mode
    if args.coin is None or args.interactive:
        return interactive_mode(analyzer)
    
    result = analyzer.analyze(
        coin_id=args.coin,
        binance_symbol=args.binance,
        days=args.days,
        save_plots=not args.no_plots
    )
    
    return result


if __name__ == "__main__":
    main()