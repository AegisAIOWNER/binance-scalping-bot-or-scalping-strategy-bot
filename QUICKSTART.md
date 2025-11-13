# Quick Start Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Binance account (testnet recommended for initial testing)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/AegisAIOWNER/binance-scalping-bot-or-scalping-strategy-bot.git
cd binance-scalping-bot-or-scalping-strategy-bot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Your API Credentials

Copy the example configuration file:

```bash
cp config.json.example config.json
```

Edit `config.json` and add your API credentials:

```json
{
    "api_key": "YOUR_ACTUAL_API_KEY",
    "api_secret": "YOUR_ACTUAL_API_SECRET",
    "testnet": true,
    ...
}
```

**IMPORTANT:** 
- For testing, use Binance Testnet credentials from https://testnet.binance.vision/
- Keep `"testnet": true` for safe testing
- Never commit `config.json` with real credentials to version control

### 4. Run the Bot

```bash
python scalable_scalping_bot.py
```

Or use the example script:

```bash
python example_usage.py
```

## Configuration Options

### Trading Symbols
Edit the `symbols` array in `config.json`:

```json
"symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
```

### Timeframe
Choose your preferred kline interval:

```json
"timeframe": "1m"  // Options: "1m", "5m", "15m", "1h", etc.
```

### Bollinger Bands Settings
Adjust the period and standard deviation:

```json
"bollinger_bands": {
    "period": 20,      // Number of periods for SMA
    "std_dev": 2       // Standard deviation multiplier
}
```

### Risk Management
Configure position sizes and limits:

```json
"risk_management": {
    "max_position_size_usd": 100,       // Max USD per position
    "min_spread_percentage": 0.05       // Minimum bid-ask spread to accept
}
```

### Take Profit and Stop Loss
Set your profit and loss percentages:

```json
"trade_params": {
    "take_profit_percentage": 0.5,    // 0.5% profit target
    "stop_loss_percentage": 0.3,      // 0.3% stop loss
    "max_open_positions": 3           // Max concurrent positions
}
```

## Testing

Run the unit tests to verify installation:

```bash
python -m unittest test_scalping_bot -v
```

Expected output: All tests should pass (12 tests).

## Safety Checklist

Before running with real funds:

- [ ] Tested thoroughly on Binance Testnet
- [ ] Reviewed and understood all configuration parameters
- [ ] Started with small position sizes
- [ ] Set appropriate stop-loss and take-profit levels
- [ ] Monitored the bot for several hours/days on testnet
- [ ] Understood the risks of automated trading
- [ ] Have a plan to monitor and stop the bot if needed

## Common Issues

### "No module named 'binance'"
**Solution:** Install dependencies with `pip install -r requirements.txt`

### "Configuration file not found"
**Solution:** Copy `config.json.example` to `config.json`

### "Invalid API key"
**Solution:** Double-check your API key and secret in `config.json`

### WebSocket connection errors
**Solution:** 
- Check your internet connection
- Verify API keys have appropriate permissions
- Try setting `"websocket_enabled": false` in config.json

## Getting Help

- Read the full [README.md](README.md) for detailed documentation
- Check the [Issues](https://github.com/AegisAIOWNER/binance-scalping-bot-or-scalping-strategy-bot/issues) page
- Review the code comments in `scalable_scalping_bot.py`

## Next Steps

1. **Monitor Performance:** Watch the logs to understand how the bot behaves
2. **Adjust Parameters:** Fine-tune Bollinger Bands, take-profit, and stop-loss settings
3. **Backtest:** Consider implementing backtesting with historical data
4. **Scale Gradually:** Only increase position sizes after proven success

## Disclaimer

This bot is for educational purposes only. Trading cryptocurrencies carries substantial risk. Never invest more than you can afford to lose.
