# Advanced Binance Scalping Bot

An advanced scalping bot for Binance that implements sophisticated trading strategies using Bollinger Bands, RSI, and OCO orders for optimal risk management and consistent profitability.

## Features

### 1. Bollinger Bands Strategy
- **SMA-based calculation**: Uses Simple Moving Average (SMA) as the middle band
- **Standard deviation bands**: Upper and lower bands calculated using configurable standard deviation multiplier
- **Precise entry/exit signals**:
  - **BUY signal**: Triggered when price touches or crosses below the lower band (oversold condition)
  - **SELL signal**: Triggered when price touches or crosses above the upper band (overbought condition)

### 2. RSI (Relative Strength Index) Indicator
- **Momentum confirmation**: RSI validates Bollinger Bands signals for higher accuracy
- **Configurable thresholds**: Default oversold (30) and overbought (70) levels
- **Better entry timing**: Only trades when both BB and RSI confirm the signal
- **Reduces false signals**: Filters out weak signals that don't meet RSI criteria

### 3. Account Balance Checking
- **Pre-trade validation**: Checks available balance before each trade
- **Balance caching**: 30-second cache to reduce API calls
- **Insufficient funds protection**: Prevents trades when balance is too low
- **Multi-asset support**: Works with USDT, BUSD, BTC and other quote assets

### 4. Dynamic Pair Selection
- **Volume-based filtering**: Selects pairs with high 24h volume ($10M+ default)
- **Volatility-based filtering**: Targets pairs with significant price movement (1%+ default)
- **Automatic discovery**: Scans all available pairs to find the best opportunities
- **Smart scoring**: Ranks pairs by volume × volatility for optimal selection
- **Avoids bad pairs**: Filters out stablecoins and leveraged tokens

### 5. Intelligent Position Sizing
- **Percentage-based risk**: Risk 1-2% of account balance per trade
- **Balance-aware**: Automatically adjusts position size based on available funds
- **Safety buffer**: Leaves 5% of balance unused for safety
- **Precision handling**: Respects exchange lot size and step size requirements

### 6. OCO (One-Cancels-the-Other) Orders
- **Automated risk management**: Every position is protected with both stop-loss and take-profit orders
- **Efficient execution**: When one order executes, the other is automatically cancelled
- **Configurable parameters**: Set your own take-profit and stop-loss percentages

### 7. Backtesting Simulation
- **Historical testing**: Test strategies on past data before live trading
- **Performance metrics**: Track win rate, average profit/loss, and total return
- **Risk-free validation**: Verify strategy effectiveness without risking capital
- **Easy to use**: Simple command-line interface for backtesting

### 8. Optimized for Fast Execution
- **WebSocket integration**: Real-time market data with minimal latency
- **Efficient data processing**: Thread-safe operations with minimal overhead
- **Retry logic**: Automatic retry for failed orders to ensure execution
- **Minimal API calls**: Optimized to reduce latency and avoid rate limits

### 9. Multi-Symbol Support
- **Concurrent monitoring**: Track multiple trading pairs simultaneously
- **Independent signals**: Each symbol analyzed independently with Bollinger Bands and RSI
- **Real-time processing**: WebSocket streams for all configured symbols

### 10. Comprehensive Risk Management
- **Position size limits**: Configure maximum position size by percentage or fixed USD
- **Spread checking**: Avoid trading during wide bid-ask spreads
- **Max open positions**: Limit total concurrent positions
- **Stop-loss protection**: Every trade protected with automatic stop-loss
- **Balance verification**: Ensures sufficient funds before each trade

### 11. Market Inefficiency Exploitation
- **Bid-ask spread monitoring**: Checks spread before entering trades
- **Short-term volatility**: Bollinger Bands capture volatility-based opportunities
- **Quick execution**: Designed for numerous trades throughout the day
- **High-probability setups**: Only trades when conditions are favorable

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
    "rsi": {
        "enabled": true,
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
        "dynamic_enabled": false,
        "quote_asset": "USDT",
        "min_volume_usd": 10000000,
        "min_volatility_percent": 1.0,
        "max_pairs": 10
    },
    "execution": {
        "websocket_enabled": true,
        "order_retry_attempts": 3,
        "latency_optimization": true
    },
    "backtest": {
        "initial_balance": 10000
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

#### RSI (Relative Strength Index)
- `enabled`: Enable/disable RSI confirmation (default: true)
- `period`: RSI calculation period (default: 14)
- `oversold_threshold`: RSI level considered oversold (default: 30)
- `overbought_threshold`: RSI level considered overbought (default: 70)

#### Trade Parameters
- `take_profit_percentage`: Take profit percentage (default: 0.5%)
- `stop_loss_percentage`: Stop loss percentage (default: 0.3%)
- `max_open_positions`: Maximum concurrent positions (default: 3)

#### Risk Management
- `position_size_type`: "percentage" or "fixed" - how to calculate position size
- `risk_percentage_per_trade`: Percentage of balance to risk per trade (default: 1.0%)
- `max_position_size_usd`: Maximum position size in USD (used when position_size_type is "fixed")
- `min_spread_percentage`: Minimum acceptable bid-ask spread

#### Pair Selection (Dynamic Mode)
- `dynamic_enabled`: Enable dynamic pair selection (default: false)
- `quote_asset`: Quote asset for filtering pairs (default: "USDT")
- `min_volume_usd`: Minimum 24h volume in USD (default: 10000000)
- `min_volatility_percent`: Minimum 24h price change percentage (default: 1.0)
- `max_pairs`: Maximum number of pairs to trade (default: 10)

#### Execution
- `websocket_enabled`: Enable WebSocket for real-time data (recommended: true)
- `order_retry_attempts`: Number of retry attempts for failed orders
- `latency_optimization`: Enable latency optimization features

#### Backtesting
- `initial_balance`: Starting balance for backtesting simulations (default: 10000)

## Usage

### Testing on Binance Testnet (Recommended)

1. Get testnet API keys from [Binance Testnet](https://testnet.binance.vision/)
2. Update `config.json` with your testnet credentials and set `"testnet": true`
3. Run the bot:

```bash
python scalable_scalping_bot.py
```

### Running Backtests

Before live trading, test your strategy on historical data:

```bash
# Backtest BTCUSDT for last 7 days
python scalable_scalping_bot.py --backtest BTCUSDT 7

# Backtest ETHUSDT for last 14 days
python scalable_scalping_bot.py --backtest ETHUSDT 14
```

The backtest will show:
- Initial and final balance
- Total profit/loss
- Number of winning vs losing trades
- Win rate percentage
- Average win/loss amounts

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
3. **RSI calculation** (if enabled):
   - Measures momentum using average gains vs losses over N periods
   - Values range from 0 (extremely oversold) to 100 (extremely overbought)
4. **Signal detection with confirmation**:
   - **BUY signal**: Price at/below lower band AND RSI ≤ oversold threshold (30)
   - **SELL signal**: Price at/above upper band AND RSI ≥ overbought threshold (70)
   - Both indicators must confirm for higher probability trades

### Dynamic Pair Selection (Optional)
1. **Volume filtering**: Scan all pairs for minimum 24h volume (e.g., $10M+)
2. **Volatility filtering**: Select pairs with significant price movement (e.g., 1%+)
3. **Smart scoring**: Rank pairs by volume × volatility
4. **Auto-selection**: Choose top N pairs automatically
5. **Bad pair avoidance**: Filter out stablecoins and leverage tokens

### Trade Execution
1. **Balance check**: Verify sufficient funds in account before trading
2. **Signal validation**: Check spread, position limits, existing positions
3. **Position sizing**: Calculate size based on account balance and risk percentage (1-2%)
4. **Market order**: Enter position at market price for fastest execution
5. **OCO order placement**: Automatically place take-profit and stop-loss orders
6. **Position tracking**: Monitor position until OCO order executes

### Risk Management
- **Balance verification**: Checks account balance before each trade
- **Percentage-based sizing**: Risk only 1-2% of total balance per trade
- **OCO protection**: Every trade protected with stop-loss + take-profit
- **Position size limits**: Configurable maximum position size
- **Spread checking**: Prevents trading during unfavorable market conditions
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
