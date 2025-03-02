!pip install web3
!pip install pandas_ta # install pandas_ta to use technical analysis indicators

from IPython import get_ipython
from IPython.display import display
import os
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from web3 import Web3
import tensorflow as tf
import pandas_ta as ta  # Using pandas_ta instead of TA-Lib

# Set API keys securely (Set these in Colab before running the code)
os.environ["INFURA_KEY"] = "43bd561d8e3a4ae99005b85107e55833"
os.environ["ETHERSCAN_KEY"] = "9TGF51PS7Q8JIDAKA1WKU9IAEMMSUWPYFW"

# Access API keys
INFURA_KEY = os.getenv("INFURA_KEY")
ETHERSCAN_KEY = os.getenv("ETHERSCAN_KEY")

# Connect to Ethereum via Infura
ETH_NODE_URL = f"https://mainnet.infura.io/v3/{INFURA_KEY}"
w3 = Web3(Web3.HTTPProvider(ETH_NODE_URL))

# Function to fetch wallet transactions
def get_wallet_transactions(wallet_address):
    url = f"https://api.etherscan.io/api?module=account&action=txlist&address={wallet_address}&sort=desc&apikey={ETHERSCAN_KEY}"

    try:
        response = requests.get(url).json()
        transactions = response.get("result", [])

        if not isinstance(transactions, list):
            print("Invalid response format:", transactions)
            return []

        return transactions
    except Exception as e:
        print(f"Error fetching transactions: {e}")
        return []

# Feature Engineering: Extract wallet activity insights
def extract_features(transactions):
    if not transactions:
        print("No transactions available.")
        return pd.DataFrame()  # Return empty DataFrame

    df = pd.DataFrame(transactions)

    # Check if necessary columns exist
    required_columns = {"value", "timeStamp", "gasPrice"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return pd.DataFrame()

    df["value"] = df["value"].astype(float) / 1e18  # Convert Wei to ETH
    df["timeStamp"] = pd.to_datetime(df["timeStamp"], unit="s")
    df["gasPrice"] = df["gasPrice"].astype(float) / 1e9  # Convert to Gwei
    df["day"] = df["timeStamp"].dt.date

    # Aggregate features per day
    features = df.groupby("day").agg({
        "value": ["sum", "count", "max", "min"],
        "gasPrice": ["mean", "max"]
    })
    features.columns = ["_".join(col) for col in features.columns]

    # Add technical indicators using pandas_ta
    df.set_index("timeStamp", inplace=True)
    df["SMA_7"] = df["value"].rolling(window=7).mean()  # 7-day SMA
    df["RSI"] = ta.rsi(df["value"], length=14)  # RSI indicator

    # Merge with aggregated features
    features = features.join(df[["SMA_7", "RSI"]].dropna(), how="left")

    # Label: 1 if next day's total value is higher, else 0
    if "value_sum" in features.columns:
        features["target"] = (features["value_sum"].shift(-1) > features["value_sum"]).astype(int)
    else:
        print("Skipping target column due to missing data.")
        features["target"] = 0  # Default target

    return features.dropna()

# Train AI Model
def train_ai_model(features):
    if features.empty or "target" not in features.columns:
        print("Insufficient data for training.")
        return None, None

    X = features.drop(columns=["target"])
    y = features["target"]

    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train & test to avoid overfitting
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Print model accuracy on test set
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")

    return model, scaler

# Predict buy/sell signals
def predict_signal(model, scaler, new_data):
    if model is None or scaler is None or new_data.empty:
        print("No valid prediction due to missing model or data.")
        return np.array([])

    scaled_data = scaler.transform(new_data)
    return model.predict(scaled_data)

# Example usage
wallet_address = "0xYourWalletHere"  # Replace with actual wallet address
transactions = get_wallet_transactions(wallet_address)
features = extract_features(transactions)

if not features.empty:
    model, scaler = train_ai_model(features)

    if model is not None and scaler is not None:
        prediction = predict_signal(model, scaler, features.tail(1))
        print("Predicted Signal:", "BUY" if prediction[0] == 1 else "SELL")
    else:
        print("Unable to train model due to insufficient data.")
else:
    print("No transaction data available for this wallet.")


import asyncio
from web3 import Web3
!pip install uniswap-python
from uniswap import Uniswap  # Requires `pip install uniswap-python`
import json

# Set up Web3 connection
INFURA_WS_URL = "wss://mainnet.infura.io/ws/v3/YOUR_INFURA_KEY"
w3 = Web3(Web3.WebsocketProvider(INFURA_WS_URL))

# Uniswap trading setup
wallet_address = "0xbf2Ec62E36C3749AFc773236baaDd3a6a0A3A5E1"
private_key = "YOUR_PRIVATE_KEY"  # Use ENV variables for security
uniswap = Uniswap(address=wallet_address, private_key=private_key, version=3, web3=w3)

# Function to execute trade
def execute_trade(signal, amount_eth=0.1):
    token_out = "USDT"  # Change based on your trading pair

    if signal == "BUY":
        tx = uniswap.make_trade("ETH", token_out, amount_eth)
        print(f"✅ BUY Trade Executed: {tx}")
    elif signal == "SELL":
        tx = uniswap.make_trade(token_out, "ETH", amount_eth)
        print(f"✅ SELL Trade Executed: {tx}")
    else:
        print("❌ No trade executed (Neutral Signal)")

# WebSocket Listener for real-time transactions
async def listen_transactions():
    print("🔴 Listening to pending transactions...")

    async for event in w3.eth.subscribe("newPendingTransactions"):
        tx_hash = event.hex()
        tx = w3.eth.get_transaction(tx_hash)

        # Monitor specific wallet activity (modify as needed)
        if tx["from"].lower() == wallet_address.lower():
            print(f"📊 Detected activity from {wallet_address}: {tx}")

            # Run AI model on latest wallet transactions
            transactions = get_wallet_transactions(wallet_address)
            features = extract_features(transactions)

            if not features.empty and model is not None and scaler is not None:
                prediction = predict_signal(model, scaler, features.tail(1))
                signal = "BUY" if prediction[0] == 1 else "SELL"
                execute_trade(signal)

# Run WebSocket listener
asyncio.run(listen_transactions())
