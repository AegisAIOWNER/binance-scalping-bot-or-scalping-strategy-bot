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
from binance.websocket import BinanceSocketManager
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


class ScalpingBot:
    """Advanced Scalping Bot for Binance"""
    
    def __init__(self, config_file: str = 'config.json'):
        """Initialize the scalping bot with configuration"""
        self.config = self._load_config(config_file)
        self.client = self._init_client()
        self.bsm = None
        
        # Data storage for each symbol
        self.price_data: Dict[str, List[float]] = defaultdict(list)
        self.current_positions: Dict[str, dict] = {}
        self.lock = threading.Lock()
        
        # Bollinger Bands calculator
        bb_config = self.config['bollinger_bands']
        self.bb_calculator = BollingerBands(
            period=bb_config['period'],
            std_dev=bb_config['std_dev']
        )
        
        # Execution optimization
        self.websocket_enabled = self.config['execution']['websocket_enabled']
        self.order_retry_attempts = self.config['execution']['order_retry_attempts']
        
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
    
    def update_price_data(self, symbol: str, price: float):
        """Update price data for a symbol (thread-safe)"""
        with self.lock:
            self.price_data[symbol].append(price)
            # Keep only recent data to optimize memory
            max_history = self.config['bollinger_bands']['period'] * 2
            if len(self.price_data[symbol]) > max_history:
                self.price_data[symbol] = self.price_data[symbol][-max_history:]
    
    def analyze_signal(self, symbol: str) -> Optional[str]:
        """
        Analyze Bollinger Bands for trading signals
        
        Returns:
            'BUY' if price touches/crosses lower band (oversold)
            'SELL' if price touches/crosses upper band (overbought)
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
        
        # Buy signal: price at or below lower band (oversold)
        if current_price <= lower * 1.001:  # Small tolerance
            logger.info(f"{symbol} BUY signal: Price {current_price:.8f} <= Lower BB {lower:.8f}")
            return 'BUY'
        
        # Sell signal: price at or above upper band (overbought)
        elif current_price >= upper * 0.999:  # Small tolerance
            logger.info(f"{symbol} SELL signal: Price {current_price:.8f} >= Upper BB {upper:.8f}")
            return 'SELL'
        
        return None
    
    def calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate position size based on risk management parameters"""
        max_position_usd = self.config['risk_management']['max_position_size_usd']
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
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
        
        return quantity
    
    def place_oco_order(self, symbol: str, side: str, quantity: float, 
                        price: float) -> Optional[dict]:
        """
        Place OCO (One-Cancels-the-Other) order for stop-loss and take-profit
        
        Args:
            symbol: Trading pair
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            price: Current market price
        
        Returns:
            Order response or None if failed
        """
        trade_params = self.config['trade_params']
        
        if side == 'BUY':
            # For a BUY, we need to SELL to close (take profit above, stop loss below)
            oco_side = SIDE_SELL
            stop_price = price * (1 - trade_params['stop_loss_percentage'] / 100)
            take_profit_price = price * (1 + trade_params['take_profit_percentage'] / 100)
        else:
            # For a SELL, we need to BUY to close (take profit below, stop loss above)
            oco_side = SIDE_BUY
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
                stop_limit_price = round(stop_price * 0.995, precision)  # Slightly below stop
        except Exception as e:
            logger.error(f"Error getting price precision for {symbol}: {e}")
            return None
        
        # Place OCO order with retry logic
        for attempt in range(self.order_retry_attempts):
            try:
                logger.info(f"Placing OCO order for {symbol}: TP={take_profit_price:.8f}, SL={stop_price:.8f}")
                
                order = self.client.create_oco_order(
                    symbol=symbol,
                    side=oco_side,
                    quantity=quantity,
                    price=take_profit_price,
                    stopPrice=stop_price,
                    stopLimitPrice=stop_limit_price,
                    stopLimitTimeInForce=TIME_IN_FORCE_GTC
                )
                
                logger.info(f"OCO order placed successfully for {symbol}")
                return order
                
            except BinanceAPIException as e:
                logger.error(f"Attempt {attempt + 1}/{self.order_retry_attempts} - Error placing OCO order for {symbol}: {e}")
                if attempt < self.order_retry_attempts - 1:
                    time.sleep(0.1)  # Brief delay before retry
                else:
                    return None
        
        return None
    
    def execute_trade(self, symbol: str, signal: str):
        """Execute a trade based on the signal"""
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
        
        # Calculate position size
        quantity = self.calculate_position_size(symbol, current_price)
        
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
            
            # Place OCO order for exit
            oco_order = self.place_oco_order(symbol, signal, quantity, fill_price)
            
            if oco_order:
                # Track position
                with self.lock:
                    self.current_positions[symbol] = {
                        'entry_price': fill_price,
                        'quantity': quantity,
                        'side': signal,
                        'entry_time': time.time(),
                        'oco_order_id': oco_order['orderListId']
                    }
                logger.info(f"Position opened for {symbol}: {signal} {quantity} @ {fill_price:.8f}")
            else:
                logger.error(f"Failed to place OCO order for {symbol}. Position may be at risk.")
                
        except BinanceAPIException as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
    
    def check_oco_status(self, symbol: str):
        """Check and update OCO order status"""
        position = self.current_positions.get(symbol)
        if not position:
            return
        
        try:
            order_list = self.client.get_order_list(orderListId=position['oco_order_id'])
            
            if order_list['listOrderStatus'] == 'ALL_DONE':
                # Position closed
                logger.info(f"Position closed for {symbol}")
                with self.lock:
                    del self.current_positions[symbol]
                    
        except BinanceAPIException as e:
            logger.error(f"Error checking OCO status for {symbol}: {e}")
    
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
        self.bsm = BinanceSocketManager(self.client)
        
        # Start kline streams for all symbols
        streams = []
        for symbol in self.config['symbols']:
            stream = self.bsm.kline_socket(
                symbol=symbol.lower(),
                interval=self.config['timeframe'],
                callback=self.on_kline_message
            )
            streams.append(stream)
        
        # Start all streams
        for stream in streams:
            stream.start()
        
        logger.info(f"WebSocket streams started for {len(streams)} symbols")
    
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
                if self.bsm:
                    self.bsm.close()
        else:
            self.run_polling_mode()
        
        logger.info("Bot stopped")


def main():
    """Main function"""
    try:
        bot = ScalpingBot('config.json')
        bot.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
