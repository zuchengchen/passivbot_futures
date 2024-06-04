#!/usr/bin/env python3
import ccxt
import pandas as pd
import btalib as btl
import time
from datetime import datetime
from time import sleep
from btalib import Indicator, mad, sma, ema


def current_time():
    return f"{datetime.now():%Y-%m-%d %H:%M:%S}"


def get_historical_data(exchange, coin_pair, timeframe="1m", limit=120):
    """
    Get Historical data (ohlcv) from a coin_pair.
    """
    data0 = exchange.fetch_ohlcv(coin_pair, timeframe=timeframe, limit=limit)
    header = ["date", "open", "high", "low", "close", "volume"]
    data = pd.DataFrame(data0, columns=header)
    data.set_index("date", inplace=True)
    data.index = pd.to_datetime(data.index, unit="ms")
    data["price_diff"] = (data.high - data.low) / data.low
    return data


def get_price_diffs(exchange):
    valid_coin_pairs = get_valid_coin_pairs(exchange)
    price_diffs = []
    deviations = []
    valid_coin_pairs_noUSDT = []
    #     print(red(f"开始从{exchange}交易所搜寻{len(valid_coin_pairs)}个交易对."))

    for coin_pair in valid_coin_pairs:
        df = get_historical_data(exchange, coin_pair, timeframe="5m", limit=1000)
        #         print(coin_pair)
        price_diffs.append(df.price_diff.mean())
        valid_coin_pairs_noUSDT.append(get_coin(coin_pair))

    df = pd.DataFrame(
        {
            "币对": valid_coin_pairs_noUSDT,
            "价差": price_diffs,
        }
    )
    df.set_index("币对", inplace=True)
    result0 = df.sort_values(by=["价差"], ascending=False)[:20]
    result = result0.index.tolist()  # .sort()
    return result


def get_coin(coin_pair):
    coin1, coin2 = coin_pair.split("/")
    return coin1


def valid_coin_pair(coin_pair):
    if "/" not in coin_pair:
        return False
    coin1, coin2 = coin_pair.split("/")
    base_coins = ["USDT"]
    if (coin1 not in base_coins) and (coin2 in base_coins):
        return True
    else:
        return False


def get_valid_coin_pairs(exchange):
    """
    Get valid coin pairs from the exchange.
    """
    exchange.load_markets()
    coin_pairs = exchange.symbols

    # list of coin pairs which are active and use "USD" as base coin
    valid_coin_pairs = []

    for coin_pair in coin_pairs:
        if valid_coin_pair(coin_pair) and exchange.markets[coin_pair]["active"]:
            valid_coin_pairs.append(coin_pair)
    return valid_coin_pairs


binance = ccxt.binance({"options": {"defaultType": "future"}})
exchange = binance


def send_message(exchange):
    print(current_time())
    buy_message = f"{current_time()} \n"
    symbols = get_price_diffs(exchange)
    buy_message += f"{exchange}价格波动最大的是：\n\n"
    buy_message += f"{symbols}\n\n"
    print(buy_message)

    file_path = "/home/czc/projects/working/stock/passivbot_futures/manager/config.yaml"
    with open(file_path, "w") as file:
        # Write content to the file
        file.write("version: 2\n")
        file.write("defaults:\n")
        file.write('  config: "latest.json"\n')
        file.write("instances:\n")
        file.write('  - user: "binance_04"\n')
        file.write("    symbols:\n")
        for symbol in symbols:
            file.write(f'     - "{symbol}USDT"\n')


time_interval = 5 * 60  # 间隔 7*24*3600

while True:
    try:
        send_message(exchange)
        sleep(time_interval)
    except:
        send_message(exchange)
        sleep(time_interval)
