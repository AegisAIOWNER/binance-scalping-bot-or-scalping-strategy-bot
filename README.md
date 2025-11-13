# Advanced Binance Scalping Bot

An advanced scalping bot for Binance that implements sophisticated trading strategies using Bollinger Bands and OCO orders for optimal risk management.

## Features

### 1. Bollinger Bands Strategy
- **SMA-based calculation**: Uses Simple Moving Average (SMA) as the middle band
- **Standard deviation bands**: Upper and lower bands calculated using configurable standard deviation multiplier
- **Precise entry/exit signals**:
  - **BUY signal**: Triggered when price touches or crosses below the lower band (oversold condition)
  - **SELL signal**: Triggered when price touches or crosses above the upper band (overbought condition)

### 2. OCO (One-Cancels-the-Other) Orders
- **Automated risk management**: Every position is protected with both stop-loss and take-profit orders
- **Efficient execution**: When one order executes, the other is automatically cancelled
- **Configurable parameters**: Set your own take-profit and stop-loss percentages

### 3. Optimized for Fast Execution
- **WebSocket integration**: Real-time market data with minimal latency
- **Efficient data processing**: Thread-safe operations with minimal overhead
- **Retry logic**: Automatic retry for failed orders to ensure execution
- **Minimal API calls**: Optimized to reduce latency and avoid rate limits

### 4. Multi-Symbol Support
- **Concurrent monitoring**: Track multiple trading pairs simultaneously
- **Independent signals**: Each symbol analyzed independently with Bollinger Bands
- **Real-time processing**: WebSocket streams for all configured symbols

### 5. Risk Management
- **Position size limits**: Configure maximum position size in USD
- **Spread checking**: Avoid trading during wide bid-ask spreads
- **Max open positions**: Limit total concurrent positions
- **Stop-loss protection**: Every trade protected with automatic stop-loss

### 6. Market Inefficiency Exploitation
- **Bid-ask spread monitoring**: Checks spread before entering trades
- **Short-term volatility**: Bollinger Bands capture volatility-based opportunities
- **Quick execution**: Designed for numerous trades throughout the day

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AegisAIOWNER/binance-scalping-bot-or-scalping-strategy-bot.git
cd binance-scalping-bot-or-scalping-strategy-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.json` to customize the bot:

```json
{
    "api_key": "YOUR_BINANCE_API_KEY",
    "api_secret": "YOUR_BINANCE_API_SECRET",
    "testnet": true,
    "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
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
        "websocket_enabled": true,
        "order_retry_attempts": 3,
        "latency_optimization": true
    }
}
```

### Configuration Parameters

#### API Configuration
- `api_key`: Your Binance API key
- `api_secret`: Your Binance API secret
- `testnet`: Set to `true` for testing on Binance Testnet

#### Trading Pairs
- `symbols`: List of trading pairs to monitor (e.g., ["BTCUSDT", "ETHUSDT"])
- `timeframe`: Kline interval (e.g., "1m", "5m", "15m")

#### Bollinger Bands
- `period`: Number of candles for SMA calculation (default: 20)
- `std_dev`: Standard deviation multiplier (default: 2)

#### Trade Parameters
- `take_profit_percentage`: Take profit percentage (default: 0.5%)
- `stop_loss_percentage`: Stop loss percentage (default: 0.3%)
- `max_open_positions`: Maximum concurrent positions (default: 3)

#### Risk Management
- `max_position_size_usd`: Maximum position size in USD
- `min_spread_percentage`: Minimum acceptable bid-ask spread

#### Execution
- `websocket_enabled`: Enable WebSocket for real-time data (recommended: true)
- `order_retry_attempts`: Number of retry attempts for failed orders
- `latency_optimization`: Enable latency optimization features

## Usage

### Testing on Binance Testnet (Recommended)

1. Get testnet API keys from [Binance Testnet](https://testnet.binance.vision/)
2. Update `config.json` with your testnet credentials and set `"testnet": true`
3. Run the bot:

```bash
python scalable_scalping_bot.py
```

### Running on Live Binance

⚠️ **WARNING**: Only use on live Binance after thorough testing on testnet!

1. Set `"testnet": false` in `config.json`
2. Update with your live API keys
3. Start with small position sizes
4. Run the bot:

```bash
python scalable_scalping_bot.py
```

## How It Works

### Signal Generation
1. **Historical data loading**: Bot fetches recent price history for each symbol
2. **Bollinger Bands calculation**: 
   - Middle band (SMA) = Average of last N closing prices
   - Upper band = SMA + (standard deviation × multiplier)
   - Lower band = SMA - (standard deviation × multiplier)
3. **Signal detection**:
   - Price at/below lower band → BUY signal (oversold)
   - Price at/above upper band → SELL signal (overbought)

### Trade Execution
1. **Signal validation**: Check spread, position limits, existing positions
2. **Market order**: Enter position at market price for fastest execution
3. **OCO order placement**: Automatically place take-profit and stop-loss orders
4. **Position tracking**: Monitor position until OCO order executes

### Risk Management
- Every trade is protected with OCO orders (stop-loss + take-profit)
- Position sizes calculated based on maximum USD allocation
- Spread checking prevents trading during unfavorable market conditions
- Maximum position limits prevent overexposure

## Architecture

### Key Components

1. **BollingerBands Class**: Calculates Bollinger Bands using SMA and standard deviation
2. **ScalpingBot Class**: Main bot logic
   - Price data management (thread-safe)
   - Signal analysis
   - Trade execution
   - OCO order management
   - WebSocket handling

### Data Flow
```
WebSocket → Price Update → Bollinger Bands → Signal → Validation → Market Order → OCO Order
```

### Threading Model
- Main thread: WebSocket event loop
- Thread-safe: All data structures protected with locks
- Concurrent: Multiple symbols processed simultaneously

## Performance Optimization

### Latency Reduction
- **WebSocket streaming**: Real-time data with minimal delay
- **Efficient calculations**: Numpy-based Bollinger Bands calculation
- **Minimal API calls**: Reduced to essential operations only
- **Local state management**: Avoid unnecessary data fetches

### Execution Speed
- **Market orders**: Immediate execution at current price
- **Retry logic**: Ensures orders execute even during brief API issues
- **Thread-safe operations**: No bottlenecks from locking

## Monitoring and Logging

The bot provides comprehensive logging:
- Signal generation (BUY/SELL)
- Order placement (market and OCO)
- Position tracking
- Error handling
- Performance metrics

Log levels can be adjusted in the code by modifying the logging configuration.

## Safety Features

1. **Testnet support**: Test strategies without risking real funds
2. **Position limits**: Prevent overtrading
3. **OCO orders**: Every position protected
4. **Spread checking**: Avoid unfavorable entry points
5. **Error handling**: Graceful degradation on API errors
6. **Retry logic**: Ensures critical operations complete

## Disclaimer

⚠️ **IMPORTANT**: Trading cryptocurrencies carries significant risk. This bot is provided for educational purposes only. Always:
- Test thoroughly on testnet before live trading
- Start with small position sizes
- Monitor the bot regularly
- Understand that past performance doesn't guarantee future results
- Never invest more than you can afford to lose

The authors are not responsible for any financial losses incurred while using this bot.

## License

This project is open source. Use at your own risk.

## Support

For issues, questions, or contributions, please open an issue on GitHub.
