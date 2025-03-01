# Market-Maker
# AI-Driven Market Maker Bot

## Overview
This AI-powered market maker bot fetches on-chain data, extracts trading signals, and runs an AI model to predict market movements for meme tokens. The bot is designed to work in Google Colab for free execution.

## Features
- Fetches wallet transactions from Etherscan
- Extracts trading signals using technical analysis (SMA, RSI)
- Aggregates wallet activity insights
- Uses a Random Forest model to predict buy/sell signals
- Connects to Ethereum via Infura

## Installation
### Prerequisites
Ensure you have Python installed. Then, install the required dependencies:
```sh
pip install web3 pandas numpy requests scikit-learn tensorflow pandas-ta ta-lib ta
```

## Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/nsain25/market-maker-bot.git
   ```
2. Navigate to the project directory:
   ```sh
   cd market-maker-bot
   ```
3. Set up API keys:
   - Get an **Infura API key** from [Infura](https://infura.io/)
   - Get an **Etherscan API key** from [Etherscan](https://etherscan.io/)
   - Export the API keys:
     ```sh
     export INFURA_KEY="your-infura-key"
     export ETHERSCAN_KEY="your-etherscan-key"
     ```
4. Run the script:
   ```sh
   python market_maker_bot.py
   ```

## Configuration
- Replace `0xYourWalletHere` in the script with an actual wallet address to analyze transactions.
- Adjust the technical indicators (SMA, RSI) if needed.

## Output
The bot will print whether the predicted signal is **BUY** or **SELL** based on on-chain data.

## Disclaimer
This bot is for educational purposes only and should not be used for financial decisions without proper risk assessment.

## License
This project is licensed under the MIT License.

