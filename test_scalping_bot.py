#!/usr/bin/env python3
"""
Unit tests for the Advanced Binance Scalping Bot
"""

import unittest
import json
import os
import time
import tempfile
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from scalable_scalping_bot import BollingerBands, RSI, ScalpingBot, BacktestSimulator
from binance.exceptions import BinanceAPIException


class TestBollingerBands(unittest.TestCase):
    """Test cases for Bollinger Bands calculation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.bb = BollingerBands(period=20, std_dev=2.0)
    
    def test_insufficient_data(self):
        """Test with insufficient price data"""
        prices = [100.0] * 10  # Less than period
        upper, middle, lower = self.bb.calculate(prices)
        self.assertIsNone(upper)
        self.assertIsNone(middle)
        self.assertIsNone(lower)
    
    def test_basic_calculation(self):
        """Test basic Bollinger Bands calculation"""
        # Create simple price data
        prices = [100.0] * 20
        upper, middle, lower = self.bb.calculate(prices)
        
        # With constant prices, std dev should be 0
        self.assertIsNotNone(upper)
        self.assertIsNotNone(middle)
        self.assertIsNotNone(lower)
        self.assertAlmostEqual(middle, 100.0, places=2)
        self.assertAlmostEqual(upper, 100.0, places=2)
        self.assertAlmostEqual(lower, 100.0, places=2)
    
    def test_volatile_prices(self):
        """Test with volatile price data"""
        # Create volatile price data
        prices = list(range(100, 120)) + list(range(120, 100, -1))
        prices = prices[:20]
        upper, middle, lower = self.bb.calculate(prices)
        
        self.assertIsNotNone(upper)
        self.assertIsNotNone(middle)
        self.assertIsNotNone(lower)
        # Upper should be greater than middle
        self.assertGreater(upper, middle)
        # Lower should be less than middle
        self.assertLess(lower, middle)
    
    def test_different_periods(self):
        """Test with different period settings"""
        bb_short = BollingerBands(period=10, std_dev=2.0)
        prices = list(range(100, 130))
        
        upper, middle, lower = bb_short.calculate(prices)
        self.assertIsNotNone(upper)
        self.assertIsNotNone(middle)
        self.assertIsNotNone(lower)


class TestScalpingBotConfig(unittest.TestCase):
    """Test cases for ScalpingBot configuration"""
    
    def setUp(self):
        """Create a temporary config file for testing"""
        self.config_data = {
            "api_key": "test_key",
            "api_secret": "test_secret",
            "testnet": True,
            "symbols": ["BTCUSDT"],
            "timeframe": "1m",
            "bollinger_bands": {
                "period": 20,
                "std_dev": 2
            },
            "trade_params": {
                "quantity_percentage": 0.1,
                "take_profit_percentage": 0.5,
                "stop_loss_percentage": 0.3,
                "max_open_positions": 3
            },
            "risk_management": {
                "max_position_size_usd": 100,
                "min_spread_percentage": 0.05
            },
            "execution": {
                "websocket_enabled": False,
                "order_retry_attempts": 3,
                "latency_optimization": True
            }
        }
        
        # Create temporary config file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.config_data, self.temp_file)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up temporary files"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    @patch('scalable_scalping_bot.Client')
    def test_config_loading(self, mock_client):
        """Test configuration file loading"""
        bot = ScalpingBot(self.temp_file.name)
        
        self.assertEqual(bot.config['api_key'], 'test_key')
        self.assertEqual(bot.config['symbols'], ['BTCUSDT'])
        self.assertEqual(bot.config['bollinger_bands']['period'], 20)
    
    @patch('scalable_scalping_bot.Client')
    def test_bollinger_bands_initialization(self, mock_client):
        """Test Bollinger Bands calculator initialization"""
        bot = ScalpingBot(self.temp_file.name)
        
        self.assertEqual(bot.bb_calculator.period, 20)
        self.assertEqual(bot.bb_calculator.std_dev, 2)


class TestScalpingBotSignals(unittest.TestCase):
    """Test cases for trading signal generation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_data = {
            "api_key": "test_key",
            "api_secret": "test_secret",
            "testnet": True,
            "symbols": ["BTCUSDT"],
            "timeframe": "1m",
            "bollinger_bands": {
                "period": 20,
                "std_dev": 2
            },
            "trade_params": {
                "quantity_percentage": 0.1,
                "take_profit_percentage": 0.5,
                "stop_loss_percentage": 0.3,
                "max_open_positions": 3
            },
            "risk_management": {
                "max_position_size_usd": 100,
                "min_spread_percentage": 0.05
            },
            "execution": {
                "websocket_enabled": False,
                "order_retry_attempts": 3,
                "latency_optimization": True
            }
        }
        
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.config_data, self.temp_file)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    @patch('scalable_scalping_bot.Client')
    def test_no_signal_with_insufficient_data(self, mock_client):
        """Test that no signal is generated with insufficient data"""
        bot = ScalpingBot(self.temp_file.name)
        
        # Add only a few prices
        for i in range(10):
            bot.update_price_data('BTCUSDT', 100.0 + i)
        
        signal = bot.analyze_signal('BTCUSDT')
        self.assertIsNone(signal)
    
    @patch('scalable_scalping_bot.Client')
    def test_buy_signal_at_lower_band(self, mock_client):
        """Test BUY signal when price hits lower band"""
        bot = ScalpingBot(self.temp_file.name)
        
        # Create price data that will trigger BUY signal
        # Start with stable prices around 100
        for i in range(19):
            bot.update_price_data('BTCUSDT', 100.0)
        
        # Add a price at the lower band (will be calculated)
        bot.update_price_data('BTCUSDT', 95.0)
        
        signal = bot.analyze_signal('BTCUSDT')
        self.assertEqual(signal, 'BUY')
    
    @patch('scalable_scalping_bot.Client')
    def test_sell_signal_at_upper_band(self, mock_client):
        """Test SELL signal when price hits upper band"""
        bot = ScalpingBot(self.temp_file.name)
        
        # Create price data that will trigger SELL signal
        # Start with stable prices around 100
        for i in range(19):
            bot.update_price_data('BTCUSDT', 100.0)
        
        # Add a price at the upper band
        bot.update_price_data('BTCUSDT', 105.0)
        
        signal = bot.analyze_signal('BTCUSDT')
        self.assertEqual(signal, 'SELL')
    
    @patch('scalable_scalping_bot.Client')
    def test_no_signal_in_middle_range(self, mock_client):
        """Test no signal when price is in middle range"""
        bot = ScalpingBot(self.temp_file.name)
        
        # Create price data with some variation but staying in middle range
        base_prices = [100.0 + i * 0.1 for i in range(-10, 10)]
        for price in base_prices:
            bot.update_price_data('BTCUSDT', price)
        
        signal = bot.analyze_signal('BTCUSDT')
        self.assertIsNone(signal)


class TestScalpingBotDataManagement(unittest.TestCase):
    """Test cases for data management"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_data = {
            "api_key": "test_key",
            "api_secret": "test_secret",
            "testnet": True,
            "symbols": ["BTCUSDT"],
            "timeframe": "1m",
            "bollinger_bands": {
                "period": 20,
                "std_dev": 2
            },
            "trade_params": {
                "quantity_percentage": 0.1,
                "take_profit_percentage": 0.5,
                "stop_loss_percentage": 0.3,
                "max_open_positions": 3
            },
            "risk_management": {
                "max_position_size_usd": 100,
                "min_spread_percentage": 0.05
            },
            "execution": {
                "websocket_enabled": False,
                "order_retry_attempts": 3,
                "latency_optimization": True
            }
        }
        
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.config_data, self.temp_file)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    @patch('scalable_scalping_bot.Client')
    def test_price_data_update(self, mock_client):
        """Test price data updating"""
        bot = ScalpingBot(self.temp_file.name)
        
        bot.update_price_data('BTCUSDT', 100.0)
        bot.update_price_data('BTCUSDT', 101.0)
        bot.update_price_data('BTCUSDT', 102.0)
        
        self.assertEqual(len(bot.price_data['BTCUSDT']), 3)
        self.assertEqual(bot.price_data['BTCUSDT'][-1], 102.0)
    
    @patch('scalable_scalping_bot.Client')
    def test_price_data_limit(self, mock_client):
        """Test that old price data is removed to limit memory"""
        bot = ScalpingBot(self.temp_file.name)
        
        # Add more data than max history
        max_history = bot.config['bollinger_bands']['period'] * 2
        for i in range(max_history + 10):
            bot.update_price_data('BTCUSDT', 100.0 + i)
        
        # Should keep only max_history items
        # Note: max_history is now based on max of BB and RSI periods
        expected_max = max(bot.config['bollinger_bands']['period'], 
                          bot.config.get('rsi', {}).get('period', 14)) * 2
        self.assertEqual(len(bot.price_data['BTCUSDT']), expected_max)


class TestRSI(unittest.TestCase):
    """Test cases for RSI calculation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rsi = RSI(period=14)
    
    def test_insufficient_data(self):
        """Test RSI with insufficient data"""
        prices = [100.0] * 10
        rsi_value = self.rsi.calculate(prices)
        self.assertIsNone(rsi_value)
    
    def test_rising_prices(self):
        """Test RSI with rising prices (should be high)"""
        prices = list(range(100, 130))  # Consistently rising
        rsi_value = self.rsi.calculate(prices)
        self.assertIsNotNone(rsi_value)
        self.assertGreater(rsi_value, 50)  # Should be above 50
    
    def test_falling_prices(self):
        """Test RSI with falling prices (should be low)"""
        prices = list(range(130, 100, -1))  # Consistently falling
        rsi_value = self.rsi.calculate(prices)
        self.assertIsNotNone(rsi_value)
        self.assertLess(rsi_value, 50)  # Should be below 50
    
    def test_rsi_range(self):
        """Test that RSI is always between 0 and 100"""
        # Create volatile price data
        prices = [100 + np.sin(i/5) * 10 for i in range(30)]
        rsi_value = self.rsi.calculate(prices)
        self.assertIsNotNone(rsi_value)
        self.assertGreaterEqual(rsi_value, 0)
        self.assertLessEqual(rsi_value, 100)


class TestEnhancedFeatures(unittest.TestCase):
    """Test cases for enhanced features"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_data = {
            "api_key": "test_key",
            "api_secret": "test_secret",
            "testnet": True,
            "symbols": ["BTCUSDT"],
            "timeframe": "1m",
            "bollinger_bands": {
                "period": 20,
                "std_dev": 2
            },
            "rsi": {
                "enabled": True,
                "period": 14,
                "oversold_threshold": 30,
                "overbought_threshold": 70
            },
            "trade_params": {
                "quantity_percentage": 0.1,
                "take_profit_percentage": 0.5,
                "stop_loss_percentage": 0.3,
                "max_open_positions": 3
            },
            "risk_management": {
                "position_size_type": "percentage",
                "risk_percentage_per_trade": 1.0,
                "max_position_size_usd": 100,
                "min_spread_percentage": 0.05
            },
            "pair_selection": {
                "dynamic_enabled": False,
                "quote_asset": "USDT",
                "min_volume_usd": 10000000,
                "min_volatility_percent": 1.0,
                "max_pairs": 10
            },
            "execution": {
                "websocket_enabled": False,
                "order_retry_attempts": 3,
                "latency_optimization": True
            }
        }
        
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.config_data, self.temp_file)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    @patch('scalable_scalping_bot.Client')
    def test_rsi_initialization(self, mock_client):
        """Test RSI calculator initialization"""
        bot = ScalpingBot(self.temp_file.name)
        
        self.assertTrue(bot.rsi_enabled)
        self.assertEqual(bot.rsi_calculator.period, 14)
        self.assertEqual(bot.rsi_oversold, 30)
        self.assertEqual(bot.rsi_overbought, 70)
    
    @patch('scalable_scalping_bot.Client')
    def test_account_balance_cache(self, mock_client):
        """Test account balance caching"""
        bot = ScalpingBot(self.temp_file.name)
        
        # Mock the client response
        mock_client.return_value.get_account.return_value = {
            'balances': [
                {'asset': 'USDT', 'free': '1000.0', 'locked': '0.0'}
            ]
        }
        
        # First call should hit API
        balance1 = bot.get_account_balance('USDT')
        self.assertEqual(balance1, 1000.0)
        
        # Second call should use cache
        balance2 = bot.get_account_balance('USDT')
        self.assertEqual(balance2, 1000.0)
        self.assertEqual(balance1, balance2)
    
    @patch('scalable_scalping_bot.Client')
    def test_position_sizing_percentage(self, mock_client):
        """Test percentage-based position sizing"""
        bot = ScalpingBot(self.temp_file.name)
        
        # Mock balance
        mock_client.return_value.get_account.return_value = {
            'balances': [
                {'asset': 'USDT', 'free': '10000.0', 'locked': '0.0'}
            ]
        }
        
        # Mock symbol info
        mock_client.return_value.get_symbol_info.return_value = {
            'filters': [
                {'filterType': 'LOT_SIZE', 'stepSize': '0.001', 'minQty': '0.001'}
            ]
        }
        
        # Test position sizing
        quantity = bot.calculate_position_size('BTCUSDT', 50000.0)
        
        # With 1% risk on 10000 balance = 100 USD
        # 100 / 50000 = 0.002 BTC
        self.assertIsNotNone(quantity)
        self.assertGreater(quantity, 0)


class TestPlaceOCOOrder(unittest.TestCase):
    """Test cases for the modified place_oco_order function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_data = {
            "api_key": "test_key",
            "api_secret": "test_secret",
            "testnet": True,
            "symbols": ["BTCUSDT"],
            "timeframe": "1m",
            "bollinger_bands": {
                "period": 20,
                "std_dev": 2
            },
            "trade_params": {
                "quantity_percentage": 0.1,
                "take_profit_percentage": 0.5,
                "stop_loss_percentage": 0.3,
                "max_open_positions": 3
            },
            "risk_management": {
                "max_position_size_usd": 100,
                "min_spread_percentage": 0.05
            },
            "execution": {
                "websocket_enabled": False,
                "order_retry_attempts": 3,
                "latency_optimization": True
            }
        }
        
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.config_data, self.temp_file)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    @patch('scalable_scalping_bot.Client')
    def test_place_separate_orders_buy(self, mock_client):
        """Test placing separate TP and SL orders for a BUY position"""
        bot = ScalpingBot(self.temp_file.name)
        
        # Mock symbol info
        mock_client.return_value.get_symbol_info.return_value = {
            'filters': [
                {'filterType': 'PRICE_FILTER', 'tickSize': '0.01'}
            ]
        }
        
        # Mock order creation responses
        tp_order_response = {'orderId': 12345, 'status': 'NEW'}
        sl_order_response = {'orderId': 12346, 'status': 'NEW'}
        
        mock_client.return_value.create_order.side_effect = [tp_order_response, sl_order_response]
        
        # Place orders
        result = bot.place_oco_order('BTCUSDT', 'BUY', 0.001, 50000.0)
        
        # Verify result
        self.assertIsNotNone(result)
        self.assertIn('take_profit_order_id', result)
        self.assertIn('stop_loss_order_id', result)
        self.assertEqual(result['take_profit_order_id'], 12345)
        self.assertEqual(result['stop_loss_order_id'], 12346)
    
    @patch('scalable_scalping_bot.Client')
    def test_place_separate_orders_sell(self, mock_client):
        """Test placing separate TP and SL orders for a SELL position"""
        bot = ScalpingBot(self.temp_file.name)
        
        # Mock symbol info
        mock_client.return_value.get_symbol_info.return_value = {
            'filters': [
                {'filterType': 'PRICE_FILTER', 'tickSize': '0.01'}
            ]
        }
        
        # Mock order creation responses
        tp_order_response = {'orderId': 22345, 'status': 'NEW'}
        sl_order_response = {'orderId': 22346, 'status': 'NEW'}
        
        mock_client.return_value.create_order.side_effect = [tp_order_response, sl_order_response]
        
        # Place orders
        result = bot.place_oco_order('BTCUSDT', 'SELL', 0.001, 50000.0)
        
        # Verify result
        self.assertIsNotNone(result)
        self.assertIn('take_profit_order_id', result)
        self.assertIn('stop_loss_order_id', result)
        self.assertEqual(result['take_profit_order_id'], 22345)
        self.assertEqual(result['stop_loss_order_id'], 22346)
    
    @patch('scalable_scalping_bot.Client')
    def test_place_orders_with_tp_failure(self, mock_client):
        """Test that when TP placement fails, no orders are placed"""
        bot = ScalpingBot(self.temp_file.name)
        
        # Mock symbol info
        mock_client.return_value.get_symbol_info.return_value = {
            'filters': [
                {'filterType': 'PRICE_FILTER', 'tickSize': '0.01'}
            ]
        }
        
        # Create a mock response object
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = '{"code":-1013,"msg":"Test error"}'
        
        # Mock order creation to fail
        mock_client.return_value.create_order.side_effect = BinanceAPIException(mock_response, 400, '{"code":-1013,"msg":"Test error"}')
        
        # Place orders
        result = bot.place_oco_order('BTCUSDT', 'BUY', 0.001, 50000.0)
        
        # Verify result is None on failure
        self.assertIsNone(result)
    
    @patch('scalable_scalping_bot.Client')
    def test_place_orders_with_sl_failure_cancels_tp(self, mock_client):
        """Test that when SL placement fails, TP order is cancelled"""
        bot = ScalpingBot(self.temp_file.name)
        
        # Mock symbol info
        mock_client.return_value.get_symbol_info.return_value = {
            'filters': [
                {'filterType': 'PRICE_FILTER', 'tickSize': '0.01'}
            ]
        }
        
        # Create a mock response object
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = '{"code":-1013,"msg":"Test error"}'
        
        # Mock order creation: TP succeeds, SL fails
        tp_order_response = {'orderId': 32345, 'status': 'NEW'}
        mock_client.return_value.create_order.side_effect = [
            tp_order_response,  # First call succeeds (TP)
            BinanceAPIException(mock_response, 400, '{"code":-1013,"msg":"Test error"}'),  # Second call fails (SL)
            tp_order_response,  # Retry: First call succeeds (TP)
            BinanceAPIException(mock_response, 400, '{"code":-1013,"msg":"Test error"}'),  # Retry: Second call fails (SL)
            tp_order_response,  # Retry: First call succeeds (TP)
            BinanceAPIException(mock_response, 400, '{"code":-1013,"msg":"Test error"}')  # Retry: Second call fails (SL)
        ]
        
        # Place orders
        result = bot.place_oco_order('BTCUSDT', 'BUY', 0.001, 50000.0)
        
        # Verify result is None after all retries fail
        self.assertIsNone(result)
        
        # Verify cancel_order was called for each TP order
        self.assertEqual(mock_client.return_value.cancel_order.call_count, 3)


class TestCheckOCOStatus(unittest.TestCase):
    """Test cases for the modified check_oco_status function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_data = {
            "api_key": "test_key",
            "api_secret": "test_secret",
            "testnet": True,
            "symbols": ["BTCUSDT"],
            "timeframe": "1m",
            "bollinger_bands": {
                "period": 20,
                "std_dev": 2
            },
            "trade_params": {
                "quantity_percentage": 0.1,
                "take_profit_percentage": 0.5,
                "stop_loss_percentage": 0.3,
                "max_open_positions": 3
            },
            "risk_management": {
                "max_position_size_usd": 100,
                "min_spread_percentage": 0.05
            },
            "execution": {
                "websocket_enabled": False,
                "order_retry_attempts": 3,
                "latency_optimization": True
            }
        }
        
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.config_data, self.temp_file)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    @patch('scalable_scalping_bot.Client')
    def test_check_status_tp_filled(self, mock_client):
        """Test that when TP is filled, SL is cancelled"""
        bot = ScalpingBot(self.temp_file.name)
        
        # Add a mock position
        bot.current_positions['BTCUSDT'] = {
            'entry_price': 50000.0,
            'quantity': 0.001,
            'side': 'BUY',
            'entry_time': time.time(),
            'take_profit_order_id': 12345,
            'stop_loss_order_id': 12346
        }
        
        # Mock get_order responses
        mock_client.return_value.get_order.side_effect = [
            {'status': 'FILLED', 'orderId': 12345},  # TP order filled
            {'status': 'NEW', 'orderId': 12346}  # SL order still open
        ]
        
        # Check status
        bot.check_oco_status('BTCUSDT')
        
        # Verify SL was cancelled
        mock_client.return_value.cancel_order.assert_called_once_with(symbol='BTCUSDT', orderId=12346)
        
        # Verify position was removed
        self.assertNotIn('BTCUSDT', bot.current_positions)
    
    @patch('scalable_scalping_bot.Client')
    def test_check_status_sl_filled(self, mock_client):
        """Test that when SL is filled, TP is cancelled"""
        bot = ScalpingBot(self.temp_file.name)
        
        # Add a mock position
        bot.current_positions['BTCUSDT'] = {
            'entry_price': 50000.0,
            'quantity': 0.001,
            'side': 'BUY',
            'entry_time': time.time(),
            'take_profit_order_id': 12345,
            'stop_loss_order_id': 12346
        }
        
        # Mock get_order responses
        mock_client.return_value.get_order.side_effect = [
            {'status': 'NEW', 'orderId': 12345},  # TP order still open
            {'status': 'FILLED', 'orderId': 12346}  # SL order filled
        ]
        
        # Check status
        bot.check_oco_status('BTCUSDT')
        
        # Verify TP was cancelled
        mock_client.return_value.cancel_order.assert_called_once_with(symbol='BTCUSDT', orderId=12345)
        
        # Verify position was removed
        self.assertNotIn('BTCUSDT', bot.current_positions)
    
    @patch('scalable_scalping_bot.Client')
    def test_check_status_both_open(self, mock_client):
        """Test that when both orders are open, position remains"""
        bot = ScalpingBot(self.temp_file.name)
        
        # Add a mock position
        bot.current_positions['BTCUSDT'] = {
            'entry_price': 50000.0,
            'quantity': 0.001,
            'side': 'BUY',
            'entry_time': time.time(),
            'take_profit_order_id': 12345,
            'stop_loss_order_id': 12346
        }
        
        # Mock get_order responses
        mock_client.return_value.get_order.side_effect = [
            {'status': 'NEW', 'orderId': 12345},  # TP order still open
            {'status': 'NEW', 'orderId': 12346}  # SL order still open
        ]
        
        # Check status
        bot.check_oco_status('BTCUSDT')
        
        # Verify no orders were cancelled
        mock_client.return_value.cancel_order.assert_not_called()
        
        # Verify position still exists
        self.assertIn('BTCUSDT', bot.current_positions)


if __name__ == '__main__':
    unittest.main()
