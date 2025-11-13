#!/usr/bin/env python3
"""
Advanced Binance Scalping Bot with Bollinger Bands and OCO Orders

This bot implements:
- Bollinger Bands (with SMA and standard deviation) for precise entry/exit signals
- OCO (One-Cancels-the-Other) orders for automated stop-loss and take-profit management
- Optimized for fast execution with minimal latency
- Robust real-time data handling for multiple symbols via WebSocket
- Capitalizes on short-term volatility, bid-ask spreads, and market inefficiencies
"""

import json
import logging
import time
import threading
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException
from binance import ThreadedWebsocketManager
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BollingerBands:
    """Calculate Bollinger Bands indicator with SMA and standard deviation"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev
    
    def calculate(self, prices: List[float]) -> Tuple[float, float, float]:
        """
        Calculate Bollinger Bands
        
        Args:
            prices: List of closing prices
            
        Returns:
            Tuple of (upper_band, middle_band/SMA, lower_band)
        """
        if len(prices) < self.period:
            return None, None, None
        
        prices_array = np.array(prices[-self.period:])
        sma = np.mean(prices_array)
        std = np.std(prices_array)
        
        upper_band = sma + (self.std_dev * std)
        lower_band = sma - (self.std_dev * std)
        
        return upper_band, sma, lower_band


class RSI:
    """Calculate Relative Strength Index (RSI) indicator"""
    
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate(self, prices: List[float]) -> Optional[float]:
        """
        Calculate RSI
        
        Args:
            prices: List of closing prices
            
        Returns:
            RSI value (0-100) or None if insufficient data
        """
        if len(prices) < self.period + 1:
            return None
        
        prices_array = np.array(prices[-(self.period + 1):])
        deltas = np.diff(prices_array)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


class ScalpingBot:
    """Advanced Scalping Bot for Binance"""
    
    def __init__(self, config_file: str = 'config.json'):
        """Initialize the scalping bot with configuration"""
        self.config = self._load_config(config_file)
        self.client = self._init_client()
        self.twm = None
        
        # Data storage for each symbol
        self.price_data: Dict[str, List[float]] = defaultdict(list)
        self.volume_data: Dict[str, List[float]] = defaultdict(list)
        self.current_positions: Dict[str, dict] = {}
        self.lock = threading.Lock()
        
        # Bollinger Bands calculator
        bb_config = self.config['bollinger_bands']
        self.bb_calculator = BollingerBands(
            period=bb_config['period'],
            std_dev=bb_config['std_dev']
        )
        
        # RSI calculator
        rsi_config = self.config.get('rsi', {'period': 14, 'enabled': True})
        self.rsi_enabled = rsi_config.get('enabled', True)
        self.rsi_calculator = RSI(period=rsi_config.get('period', 14))
        self.rsi_oversold = rsi_config.get('oversold_threshold', 30)
        self.rsi_overbought = rsi_config.get('overbought_threshold', 70)
        
        # Execution optimization
        self.websocket_enabled = self.config['execution']['websocket_enabled']
        self.order_retry_attempts = self.config['execution']['order_retry_attempts']
        
        # Account balance cache
        self.account_balance_cache = {}
        self.balance_cache_time = 0
        self.balance_cache_ttl = 30  # Cache for 30 seconds
        
        logger.info("Scalping bot initialized successfully")
    
    def _load_config(self, config_file: str) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file {config_file} not found")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in configuration file {config_file}")
            raise
    
    def _init_client(self) -> Client:
        """Initialize Binance client"""
        api_key = self.config['api_key']
        api_secret = self.config['api_secret']
        
        if self.config.get('testnet', False):
            # Use testnet
            client = Client(api_key, api_secret, testnet=True)
            logger.info("Connected to Binance Testnet")
        else:
            client = Client(api_key, api_secret)
            logger.info("Connected to Binance Live")
        
        return client
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol with minimal latency"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def get_bid_ask_spread(self, symbol: str) -> Tuple[float, float, float]:
        """
        Get bid-ask spread for a symbol
        
        Returns:
            Tuple of (bid_price, ask_price, spread_percentage)
        """
        try:
            orderbook = self.client.get_orderbook_ticker(symbol=symbol)
            bid_price = float(orderbook['bidPrice'])
            ask_price = float(orderbook['askPrice'])
            spread_pct = ((ask_price - bid_price) / bid_price) * 100
            return bid_price, ask_price, spread_pct
        except BinanceAPIException as e:
            logger.error(f"Error getting spread for {symbol}: {e}")
            return None, None, None
    
    def get_account_balance(self, quote_asset: str = 'USDT') -> Optional[float]:
        """
        Get account balance for a specific asset with caching
        
        Args:
            quote_asset: Asset to check balance for (default: USDT)
            
        Returns:
            Available balance or None if error
        """
        current_time = time.time()
        
        # Use cache if recent
        if current_time - self.balance_cache_time < self.balance_cache_ttl:
            if quote_asset in self.account_balance_cache:
                return self.account_balance_cache[quote_asset]
        
        try:
            account_info = self.client.get_account()
            for balance in account_info['balances']:
                if balance['asset'] == quote_asset:
                    free_balance = float(balance['free'])
                    # Update cache
                    self.account_balance_cache[quote_asset] = free_balance
                    self.balance_cache_time = current_time
                    return free_balance
            return 0.0
        except BinanceAPIException as e:
            logger.error(f"Error getting account balance: {e}")
            return None
    
    def get_symbol_volume_volatility(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Get 24h volume and price volatility for a symbol
        
        Returns:
            Tuple of (24h_volume_usd, price_change_percent)
        """
        try:
            ticker = self.client.get_ticker(symbol=symbol)
            volume = float(ticker['quoteVolume'])  # 24h volume in quote asset (usually USDT)
            price_change_pct = float(ticker['priceChangePercent'])
            return volume, abs(price_change_pct)
        except BinanceAPIException as e:
            logger.error(f"Error getting volume/volatility for {symbol}: {e}")
            return None, None
    
    def select_dynamic_pairs(self) -> List[str]:
        """
        Dynamically select trading pairs based on volume and volatility
        
        Returns:
            List of selected trading pairs
        """
        pair_selection = self.config.get('pair_selection', {})
        
        if not pair_selection.get('dynamic_enabled', False):
            # Return static list from config
            return self.config.get('symbols', [])
        
        try:
            quote_asset = pair_selection.get('quote_asset', 'USDT')
            min_volume = pair_selection.get('min_volume_usd', 10000000)  # $10M default
            min_volatility = pair_selection.get('min_volatility_percent', 1.0)  # 1% default
            max_pairs = pair_selection.get('max_pairs', 10)
            
            # Get all tickers
            tickers = self.client.get_ticker()
            
            # Filter pairs
            candidates = []
            for ticker in tickers:
                symbol = ticker['symbol']
                
                # Only consider pairs with the specified quote asset
                if not symbol.endswith(quote_asset):
                    continue
                
                # Skip stable coins and leverage tokens
                base_asset = symbol.replace(quote_asset, '')
                skip_assets = ['USDC', 'BUSD', 'TUSD', 'PAX', 'UP', 'DOWN', 'BULL', 'BEAR']
                if any(skip in base_asset for skip in skip_assets):
                    continue
                
                volume = float(ticker['quoteVolume'])
                price_change = abs(float(ticker['priceChangePercent']))
                
                # Apply filters
                if volume >= min_volume and price_change >= min_volatility:
                    candidates.append({
                        'symbol': symbol,
                        'volume': volume,
                        'volatility': price_change,
                        'score': volume * price_change  # Simple scoring
                    })
            
            # Sort by score and take top pairs
            candidates.sort(key=lambda x: x['score'], reverse=True)
            selected_pairs = [c['symbol'] for c in candidates[:max_pairs]]
            
            logger.info(f"Selected {len(selected_pairs)} dynamic pairs: {selected_pairs}")
            return selected_pairs
            
        except BinanceAPIException as e:
            logger.error(f"Error selecting dynamic pairs: {e}")
            # Fallback to static list
            return self.config.get('symbols', [])
    
    def update_price_data(self, symbol: str, price: float, volume: float = 0):
        """Update price data for a symbol (thread-safe)"""
        with self.lock:
            self.price_data[symbol].append(price)
            self.volume_data[symbol].append(volume)
            # Keep only recent data to optimize memory
            max_history = max(self.config['bollinger_bands']['period'], 
                            self.config.get('rsi', {}).get('period', 14)) * 2
            if len(self.price_data[symbol]) > max_history:
                self.price_data[symbol] = self.price_data[symbol][-max_history:]
                self.volume_data[symbol] = self.volume_data[symbol][-max_history:]
    
    def analyze_signal(self, symbol: str) -> Optional[str]:
        """
        Analyze Bollinger Bands and RSI for trading signals
        
        Returns:
            'BUY' if price touches/crosses lower band (oversold) and RSI confirms
            'SELL' if price touches/crosses upper band (overbought) and RSI confirms
            None if no signal
        """
        with self.lock:
            prices = self.price_data.get(symbol, [])
        
        if len(prices) < self.config['bollinger_bands']['period']:
            return None
        
        upper, middle, lower = self.bb_calculator.calculate(prices)
        
        if upper is None:
            return None
        
        current_price = prices[-1]
        
        # Calculate RSI if enabled
        rsi_value = None
        if self.rsi_enabled:
            rsi_value = self.rsi_calculator.calculate(prices)
        
        # Buy signal: price at or below lower band (oversold)
        if current_price <= lower * 1.001:  # Small tolerance
            # If RSI enabled, confirm with RSI
            if self.rsi_enabled and rsi_value is not None:
                if rsi_value > self.rsi_oversold:
                    # RSI not oversold, skip signal
                    return None
                logger.info(f"{symbol} BUY signal: Price {current_price:.8f} <= Lower BB {lower:.8f}, RSI {rsi_value:.2f}")
            else:
                logger.info(f"{symbol} BUY signal: Price {current_price:.8f} <= Lower BB {lower:.8f}")
            return 'BUY'
        
        # Sell signal: price at or above upper band (overbought)
        elif current_price >= upper * 0.999:  # Small tolerance
            # If RSI enabled, confirm with RSI
            if self.rsi_enabled and rsi_value is not None:
                if rsi_value < self.rsi_overbought:
                    # RSI not overbought, skip signal
                    return None
                logger.info(f"{symbol} SELL signal: Price {current_price:.8f} >= Upper BB {upper:.8f}, RSI {rsi_value:.2f}")
            else:
                logger.info(f"{symbol} SELL signal: Price {current_price:.8f} >= Upper BB {upper:.8f}")
            return 'SELL'
        
        return None
    
    def calculate_position_size(self, symbol: str, price: float) -> Optional[float]:
        """
        Calculate position size based on account balance and risk percentage
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            price: Current price
            
        Returns:
            Position size in base asset quantity, or None if insufficient funds
        """
        # Extract quote asset from symbol (usually USDT)
        quote_asset = 'USDT'  # Default
        if symbol.endswith('USDT'):
            quote_asset = 'USDT'
        elif symbol.endswith('BUSD'):
            quote_asset = 'BUSD'
        elif symbol.endswith('BTC'):
            quote_asset = 'BTC'
        
        # Get account balance
        balance = self.get_account_balance(quote_asset)
        if balance is None or balance == 0:
            logger.warning(f"Insufficient {quote_asset} balance")
            return None
        
        # Calculate position size
        risk_config = self.config['risk_management']
        
        # Use percentage-based sizing if enabled
        if risk_config.get('position_size_type', 'fixed') == 'percentage':
            risk_percentage = risk_config.get('risk_percentage_per_trade', 1.0)
            max_position_usd = balance * (risk_percentage / 100)
        else:
            # Use fixed USD amount (legacy)
            max_position_usd = risk_config.get('max_position_size_usd', 100)
        
        # Ensure we don't exceed available balance
        max_position_usd = min(max_position_usd, balance * 0.95)  # Leave 5% buffer
        
        quantity = max_position_usd / price
        
        # Get symbol info for precision
        try:
            info = self.client.get_symbol_info(symbol)
            step_size = None
            min_qty = None
            
            for filter in info['filters']:
                if filter['filterType'] == 'LOT_SIZE':
                    step_size = float(filter['stepSize'])
                    min_qty = float(filter['minQty'])
            
            if step_size and min_qty:
                # Round down to step size
                precision = len(str(step_size).rstrip('0').split('.')[-1])
                quantity = max(min_qty, round(quantity - (quantity % step_size), precision))
                
                # Final check: ensure we have enough balance
                required_balance = quantity * price
                if required_balance > balance:
                    logger.warning(f"Insufficient balance for {symbol}. Required: {required_balance:.2f}, Available: {balance:.2f}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return None
        
        return quantity
    
    def place_oco_order(self, symbol: str, side: str, quantity: float, 
                        price: float) -> Optional[dict]:
        """
        Place separate limit (take-profit) and stop-loss orders
        
        This replaces the OCO order functionality with two separate orders:
        1. A limit order for take-profit
        2. A stop-loss limit order for stop-loss
        
        Args:
            symbol: Trading pair
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            price: Current market price
        
        Returns:
            Dict with both order responses or None if failed
        """
        trade_params = self.config['trade_params']
        
        if side == 'BUY':
            # For a BUY, we need to SELL to close (take profit above, stop loss below)
            exit_side = SIDE_SELL
            stop_price = price * (1 - trade_params['stop_loss_percentage'] / 100)
            take_profit_price = price * (1 + trade_params['take_profit_percentage'] / 100)
        else:
            # For a SELL, we need to BUY to close (take profit below, stop loss above)
            exit_side = SIDE_BUY
            stop_price = price * (1 + trade_params['stop_loss_percentage'] / 100)
            take_profit_price = price * (1 - trade_params['take_profit_percentage'] / 100)
        
        # Get price precision
        try:
            info = self.client.get_symbol_info(symbol)
            tick_size = None
            
            for filter in info['filters']:
                if filter['filterType'] == 'PRICE_FILTER':
                    tick_size = float(filter['tickSize'])
            
            if tick_size:
                precision = len(str(tick_size).rstrip('0').split('.')[-1])
                stop_price = round(stop_price, precision)
                take_profit_price = round(take_profit_price, precision)
                stop_limit_price = round(stop_price * 0.995, precision) if side == 'BUY' else round(stop_price * 1.005, precision)
        except Exception as e:
            logger.error(f"Error getting price precision for {symbol}: {e}")
            return None
        
        # Place separate orders with retry logic
        take_profit_order = None
        stop_loss_order = None
        
        for attempt in range(self.order_retry_attempts):
            try:
                logger.info(f"Placing separate TP and SL orders for {symbol}: TP={take_profit_price:.8f}, SL={stop_price:.8f}")
                
                # Place take-profit limit order
                take_profit_order = self.client.create_order(
                    symbol=symbol,
                    side=exit_side,
                    type=ORDER_TYPE_LIMIT,
                    quantity=quantity,
                    price=take_profit_price,
                    timeInForce=TIME_IN_FORCE_GTC
                )
                
                logger.info(f"Take-profit order placed for {symbol}: ID={take_profit_order['orderId']}")
                
                # Place stop-loss limit order
                stop_loss_order = self.client.create_order(
                    symbol=symbol,
                    side=exit_side,
                    type=ORDER_TYPE_STOP_LOSS_LIMIT,
                    quantity=quantity,
                    price=stop_limit_price,
                    stopPrice=stop_price,
                    timeInForce=TIME_IN_FORCE_GTC
                )
                
                logger.info(f"Stop-loss order placed for {symbol}: ID={stop_loss_order['orderId']}")
                
                # Return both orders in a compatible format
                return {
                    'take_profit_order': take_profit_order,
                    'stop_loss_order': stop_loss_order,
                    'take_profit_order_id': take_profit_order['orderId'],
                    'stop_loss_order_id': stop_loss_order['orderId']
                }
                
            except BinanceAPIException as e:
                logger.error(f"Attempt {attempt + 1}/{self.order_retry_attempts} - Error placing orders for {symbol}: {e}")
                
                # If take-profit order was placed but stop-loss failed, cancel take-profit
                if take_profit_order and not stop_loss_order:
                    try:
                        self.client.cancel_order(symbol=symbol, orderId=take_profit_order['orderId'])
                        logger.info(f"Cancelled take-profit order {take_profit_order['orderId']} after stop-loss placement failed")
                    except Exception as cancel_error:
                        logger.error(f"Error cancelling take-profit order: {cancel_error}")
                
                if attempt < self.order_retry_attempts - 1:
                    time.sleep(0.1)  # Brief delay before retry
                    take_profit_order = None
                    stop_loss_order = None
                else:
                    return None
        
        return None
    
    def execute_trade(self, symbol: str, signal: str):
        """Execute a trade based on the signal with balance and safety checks"""
        # Check if we've reached max open positions
        if len(self.current_positions) >= self.config['trade_params']['max_open_positions']:
            logger.warning(f"Max open positions reached. Skipping trade for {symbol}")
            return
        
        # Check if we already have a position in this symbol
        if symbol in self.current_positions:
            logger.info(f"Already have an open position in {symbol}. Skipping.")
            return
        
        # Check bid-ask spread
        bid, ask, spread_pct = self.get_bid_ask_spread(symbol)
        min_spread = self.config['risk_management']['min_spread_percentage']
        
        if spread_pct is None or spread_pct > min_spread:
            logger.warning(f"Spread too wide for {symbol}: {spread_pct:.4f}% > {min_spread}%")
            return
        
        # Get current price
        current_price = self.get_current_price(symbol)
        if current_price is None:
            return
        
        # Calculate position size (includes balance check)
        quantity = self.calculate_position_size(symbol, current_price)
        if quantity is None:
            logger.warning(f"Cannot calculate position size for {symbol} - insufficient balance or error")
            return
        
        try:
            # Place market order to enter position
            if signal == 'BUY':
                logger.info(f"Executing BUY order for {symbol}: {quantity} @ {current_price:.8f}")
                market_order = self.client.create_order(
                    symbol=symbol,
                    side=SIDE_BUY,
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity
                )
            else:  # SELL
                logger.info(f"Executing SELL order for {symbol}: {quantity} @ {current_price:.8f}")
                market_order = self.client.create_order(
                    symbol=symbol,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity
                )
            
            # Get actual fill price
            fill_price = float(market_order['fills'][0]['price']) if market_order.get('fills') else current_price
            
            # Place separate TP and SL orders for exit
            exit_orders = self.place_oco_order(symbol, signal, quantity, fill_price)
            
            if exit_orders:
                # Track position
                with self.lock:
                    self.current_positions[symbol] = {
                        'entry_price': fill_price,
                        'quantity': quantity,
                        'side': signal,
                        'entry_time': time.time(),
                        'take_profit_order_id': exit_orders['take_profit_order_id'],
                        'stop_loss_order_id': exit_orders['stop_loss_order_id']
                    }
                logger.info(f"Position opened for {symbol}: {signal} {quantity} @ {fill_price:.8f}")
            else:
                logger.error(f"Failed to place exit orders for {symbol}. Position may be at risk.")
                
        except BinanceAPIException as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
    
    def check_oco_status(self, symbol: str):
        """Check and update order status for take-profit and stop-loss orders"""
        position = self.current_positions.get(symbol)
        if not position:
            return
        
        try:
            # Check take-profit order status
            tp_order = self.client.get_order(
                symbol=symbol,
                orderId=position['take_profit_order_id']
            )
            
            # Check stop-loss order status
            sl_order = self.client.get_order(
                symbol=symbol,
                orderId=position['stop_loss_order_id']
            )
            
            # Check if either order is filled
            tp_filled = tp_order['status'] == 'FILLED'
            sl_filled = sl_order['status'] == 'FILLED'
            
            if tp_filled or sl_filled:
                # One order executed, cancel the other
                if tp_filled:
                    logger.info(f"Take-profit order filled for {symbol}")
                    # Cancel stop-loss order
                    try:
                        self.client.cancel_order(symbol=symbol, orderId=position['stop_loss_order_id'])
                        logger.info(f"Cancelled stop-loss order for {symbol}")
                    except BinanceAPIException as e:
                        # Order might already be cancelled or filled
                        logger.warning(f"Could not cancel stop-loss order for {symbol}: {e}")
                
                elif sl_filled:
                    logger.info(f"Stop-loss order filled for {symbol}")
                    # Cancel take-profit order
                    try:
                        self.client.cancel_order(symbol=symbol, orderId=position['take_profit_order_id'])
                        logger.info(f"Cancelled take-profit order for {symbol}")
                    except BinanceAPIException as e:
                        # Order might already be cancelled or filled
                        logger.warning(f"Could not cancel take-profit order for {symbol}: {e}")
                
                # Position closed
                logger.info(f"Position closed for {symbol}")
                with self.lock:
                    del self.current_positions[symbol]
                    
        except BinanceAPIException as e:
            logger.error(f"Error checking order status for {symbol}: {e}")
    
    def process_kline(self, symbol: str, kline: dict):
        """Process kline data for a symbol"""
        close_price = float(kline['c'])
        is_closed = kline['x']
        
        # Update price data
        if is_closed:
            self.update_price_data(symbol, close_price)
            
            # Check for signals
            signal = self.analyze_signal(symbol)
            
            if signal:
                # Execute trade
                self.execute_trade(symbol, signal)
        
        # Check status of open positions
        if symbol in self.current_positions:
            self.check_oco_status(symbol)
    
    def on_kline_message(self, msg):
        """WebSocket callback for kline messages"""
        if msg['e'] == 'error':
            logger.error(f"WebSocket error: {msg}")
            return
        
        if msg['e'] == 'kline':
            symbol = msg['s']
            kline = msg['k']
            self.process_kline(symbol, kline)
    
    def start_websocket(self):
        """Start WebSocket streams for real-time data"""
        if not self.websocket_enabled:
            logger.info("WebSocket disabled in config")
            return
        
        logger.info("Starting WebSocket connections...")
        self.twm = ThreadedWebsocketManager(
            api_key=self.config['api_key'],
            api_secret=self.config['api_secret'],
            testnet=self.config.get('testnet', False)
        )
        self.twm.start()
        
        # Start kline streams for all symbols
        for symbol in self.config['symbols']:
            self.twm.start_kline_socket(
                callback=self.on_kline_message,
                symbol=symbol.lower(),
                interval=self.config['timeframe']
            )
            logger.info(f"Started WebSocket stream for {symbol}")
        
        logger.info(f"WebSocket streams started for {len(self.config['symbols'])} symbols")
    
    def fetch_historical_data(self, symbol: str, limit: int = None):
        """Fetch historical kline data to initialize price history"""
        if limit is None:
            limit = self.config['bollinger_bands']['period'] + 10
        
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=self.config['timeframe'],
                limit=limit
            )
            
            # Extract closing prices
            for kline in klines:
                close_price = float(kline[4])
                self.update_price_data(symbol, close_price)
            
            logger.info(f"Loaded {len(klines)} historical klines for {symbol}")
            
        except BinanceAPIException as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
    
    def run_polling_mode(self):
        """Run bot in polling mode (fallback if WebSocket disabled)"""
        logger.info("Starting polling mode...")
        
        while True:
            try:
                for symbol in self.config['symbols']:
                    # Get latest kline
                    klines = self.client.get_klines(
                        symbol=symbol,
                        interval=self.config['timeframe'],
                        limit=1
                    )
                    
                    if klines:
                        kline = klines[0]
                        kline_dict = {
                            'c': kline[4],  # close
                            'x': True  # is closed
                        }
                        self.process_kline(symbol, kline_dict)
                
                # Sleep to avoid rate limits
                time.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("Stopping bot...")
                break
            except Exception as e:
                logger.error(f"Error in polling mode: {e}")
                time.sleep(5)
    
    def run(self):
        """Main entry point to run the bot"""
        logger.info("=" * 60)
        logger.info("Starting Advanced Binance Scalping Bot")
        logger.info("=" * 60)
        logger.info(f"Symbols: {', '.join(self.config['symbols'])}")
        logger.info(f"Timeframe: {self.config['timeframe']}")
        logger.info(f"Bollinger Bands: Period={self.config['bollinger_bands']['period']}, StdDev={self.config['bollinger_bands']['std_dev']}")
        logger.info(f"WebSocket: {'Enabled' if self.websocket_enabled else 'Disabled'}")
        logger.info("=" * 60)
        
        # Fetch historical data for all symbols
        for symbol in self.config['symbols']:
            self.fetch_historical_data(symbol)
        
        # Start WebSocket or polling mode
        if self.websocket_enabled:
            self.start_websocket()
            
            # Keep main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Stopping bot...")
                if self.twm:
                    self.twm.stop()
        else:
            self.run_polling_mode()
        
        logger.info("Bot stopped")


class BacktestSimulator:
    """Backtesting simulator for the scalping strategy"""
    
    def __init__(self, config_file: str = 'config.json'):
        """Initialize backtesting simulator"""
        self.config = self._load_config(config_file)
        self.client = self._init_client()
        
        # Indicators
        bb_config = self.config['bollinger_bands']
        self.bb_calculator = BollingerBands(
            period=bb_config['period'],
            std_dev=bb_config['std_dev']
        )
        
        rsi_config = self.config.get('rsi', {'period': 14, 'enabled': True})
        self.rsi_enabled = rsi_config.get('enabled', True)
        self.rsi_calculator = RSI(period=rsi_config.get('period', 14))
        self.rsi_oversold = rsi_config.get('oversold_threshold', 30)
        self.rsi_overbought = rsi_config.get('overbought_threshold', 70)
        
        # Results tracking
        self.trades = []
        self.balance = self.config.get('backtest', {}).get('initial_balance', 10000)
        self.initial_balance = self.balance
        
    def _load_config(self, config_file: str) -> dict:
        """Load configuration from JSON file"""
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def _init_client(self) -> Client:
        """Initialize Binance client"""
        api_key = self.config['api_key']
        api_secret = self.config['api_secret']
        return Client(api_key, api_secret, testnet=self.config.get('testnet', False))
    
    def run_backtest(self, symbol: str, days: int = 7):
        """
        Run backtest on historical data
        
        Args:
            symbol: Trading pair to backtest
            days: Number of days of historical data to use
        """
        logger.info(f"Starting backtest for {symbol} over {days} days")
        logger.info(f"Initial balance: ${self.initial_balance:.2f}")
        
        # Fetch historical data
        interval = self.config['timeframe']
        limit = days * 24 * 60  # Assuming 1m intervals
        if interval == '5m':
            limit = days * 24 * 12
        elif interval == '15m':
            limit = days * 24 * 4
        elif interval == '1h':
            limit = days * 24
        
        limit = min(limit, 1000)  # API limit
        
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
        except BinanceAPIException as e:
            logger.error(f"Error fetching historical data: {e}")
            return
        
        # Process klines
        price_history = []
        position = None
        
        for i, kline in enumerate(klines):
            close_price = float(kline[4])
            price_history.append(close_price)
            
            # Need enough data for indicators
            if len(price_history) < max(self.config['bollinger_bands']['period'], 
                                       self.config.get('rsi', {}).get('period', 14)) + 1:
                continue
            
            # Calculate indicators
            upper, middle, lower = self.bb_calculator.calculate(price_history)
            rsi_value = self.rsi_calculator.calculate(price_history) if self.rsi_enabled else None
            
            if upper is None:
                continue
            
            # Check for entry signals if no position
            if position is None:
                # Buy signal
                if close_price <= lower * 1.001:
                    if not self.rsi_enabled or (rsi_value and rsi_value <= self.rsi_oversold):
                        # Calculate position size
                        risk_pct = self.config['risk_management'].get('risk_percentage_per_trade', 1.0)
                        position_size_usd = self.balance * (risk_pct / 100)
                        quantity = position_size_usd / close_price
                        
                        position = {
                            'type': 'BUY',
                            'entry_price': close_price,
                            'quantity': quantity,
                            'entry_index': i
                        }
                        logger.info(f"[Backtest] BUY at {close_price:.8f}, RSI: {rsi_value:.2f if rsi_value else 'N/A'}")
                
                # Sell signal
                elif close_price >= upper * 0.999:
                    if not self.rsi_enabled or (rsi_value and rsi_value >= self.rsi_overbought):
                        # For short positions (not implemented in simple backtest)
                        pass
            
            # Check for exit if we have a position
            elif position:
                entry_price = position['entry_price']
                quantity = position['quantity']
                
                # Calculate profit/loss targets
                tp_pct = self.config['trade_params']['take_profit_percentage']
                sl_pct = self.config['trade_params']['stop_loss_percentage']
                
                take_profit_price = entry_price * (1 + tp_pct / 100)
                stop_loss_price = entry_price * (1 - sl_pct / 100)
                
                # Check if TP or SL hit
                if close_price >= take_profit_price:
                    # Take profit
                    pnl = quantity * (close_price - entry_price)
                    self.balance += pnl
                    
                    self.trades.append({
                        'entry_price': entry_price,
                        'exit_price': close_price,
                        'pnl': pnl,
                        'pnl_pct': ((close_price - entry_price) / entry_price) * 100,
                        'type': 'TP',
                        'duration': i - position['entry_index']
                    })
                    
                    logger.info(f"[Backtest] TP at {close_price:.8f}, PnL: ${pnl:.2f} ({self.trades[-1]['pnl_pct']:.2f}%)")
                    position = None
                    
                elif close_price <= stop_loss_price:
                    # Stop loss
                    pnl = quantity * (close_price - entry_price)
                    self.balance += pnl
                    
                    self.trades.append({
                        'entry_price': entry_price,
                        'exit_price': close_price,
                        'pnl': pnl,
                        'pnl_pct': ((close_price - entry_price) / entry_price) * 100,
                        'type': 'SL',
                        'duration': i - position['entry_index']
                    })
                    
                    logger.info(f"[Backtest] SL at {close_price:.8f}, PnL: ${pnl:.2f} ({self.trades[-1]['pnl_pct']:.2f}%)")
                    position = None
        
        # Print results
        self.print_results()
    
    def print_results(self):
        """Print backtest results"""
        logger.info("=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Initial Balance: ${self.initial_balance:.2f}")
        logger.info(f"Final Balance: ${self.balance:.2f}")
        logger.info(f"Total PnL: ${self.balance - self.initial_balance:.2f} ({((self.balance - self.initial_balance) / self.initial_balance * 100):.2f}%)")
        logger.info(f"Total Trades: {len(self.trades)}")
        
        if self.trades:
            winning_trades = [t for t in self.trades if t['pnl'] > 0]
            losing_trades = [t for t in self.trades if t['pnl'] <= 0]
            
            logger.info(f"Winning Trades: {len(winning_trades)} ({len(winning_trades)/len(self.trades)*100:.1f}%)")
            logger.info(f"Losing Trades: {len(losing_trades)} ({len(losing_trades)/len(self.trades)*100:.1f}%)")
            
            if winning_trades:
                avg_win = np.mean([t['pnl'] for t in winning_trades])
                logger.info(f"Average Win: ${avg_win:.2f}")
            
            if losing_trades:
                avg_loss = np.mean([t['pnl'] for t in losing_trades])
                logger.info(f"Average Loss: ${avg_loss:.2f}")
            
            avg_duration = np.mean([t['duration'] for t in self.trades])
            logger.info(f"Average Trade Duration: {avg_duration:.1f} candles")
        
        logger.info("=" * 60)


def main():
    """Main function"""
    import sys
    
    # Check if running in backtest mode
    if len(sys.argv) > 1 and sys.argv[1] == '--backtest':
        try:
            symbol = sys.argv[2] if len(sys.argv) > 2 else 'BTCUSDT'
            days = int(sys.argv[3]) if len(sys.argv) > 3 else 7
            
            simulator = BacktestSimulator('config.json')
            simulator.run_backtest(symbol, days)
        except Exception as e:
            logger.error(f"Backtest error: {e}", exc_info=True)
            return 1
    else:
        try:
            bot = ScalpingBot('config.json')
            bot.run()
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
