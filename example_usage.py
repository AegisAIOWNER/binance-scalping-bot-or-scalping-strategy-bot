#!/usr/bin/env python3
"""
Example usage of the Binance Scalping Bot

This script demonstrates how to set up and run the scalping bot.
Make sure to configure config.json with your API credentials before running.
"""

import sys
import signal
from scalable_scalping_bot import ScalpingBot, logger


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    logger.info("\nShutdown signal received. Stopping bot...")
    sys.exit(0)


def main():
    """Main function to run the bot"""
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=" * 70)
    print("Advanced Binance Scalping Bot")
    print("=" * 70)
    print("\nStarting bot with configuration from config.json...")
    print("\nIMPORTANT:")
    print("- Make sure you have configured your API keys in config.json")
    print("- It is HIGHLY recommended to test on testnet first")
    print("- Set 'testnet': true in config.json to use testnet")
    print("- Monitor the bot regularly and start with small positions")
    print("\nPress Ctrl+C to stop the bot at any time.")
    print("=" * 70)
    print()
    
    try:
        # Create and run the bot
        bot = ScalpingBot('config.json')
        bot.run()
        
    except FileNotFoundError:
        logger.error("Configuration file config.json not found!")
        logger.error("Please copy config.json.example to config.json and configure it.")
        return 1
        
    except KeyboardInterrupt:
        logger.info("\nBot stopped by user")
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
