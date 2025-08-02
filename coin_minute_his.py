import requests
import pandas as pd
import json
import time
from datetime import datetime
from pathlib import Path
from data.gc_release import release_all
from data.td_dao import insert_df_to_tdengine


LAST_TIMESTAMP_FILE = Path("last_timestamp.json")


def robust_request(url, params, max_retries=3, backoff=1):
    for attempt in range(max_retries):
        try:
            proxies = {
                "http":  "http://huangchixin:19971030hcx@proxy.smartproxycn.com:1000",
                "https": "http://huangchixin:19971030hcx@proxy.smartproxycn.com:1000",
            }
            response = requests.get(url, params=params, timeout=10, proxies=proxies)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ 状态码错误: {response.status_code}")
        except Exception as e:
            print(f"⚠️ 第 {attempt + 1} 次请求失败：{e}")
        time.sleep(backoff)
    return None


def load_last_timestamp(symbol: str, tsym: str) -> int:
    try:
        if LAST_TIMESTAMP_FILE.exists():
            with open(LAST_TIMESTAMP_FILE, "r") as f:
                ts_map = json.load(f)
            return ts_map.get(f"{symbol}_{tsym}", 0)
        return 0
    except:
        return 0


def save_last_timestamp(symbol: str, tsym: str, timestamp: int):
    key = f"{symbol}_{tsym}"
    ts_map = {}
    if LAST_TIMESTAMP_FILE.exists():
        try:
            with open(LAST_TIMESTAMP_FILE, "r") as f:
                ts_map = json.load(f)
        except:
            ts_map = {}

    ts_map[key] = timestamp
    with open(LAST_TIMESTAMP_FILE, "w") as f:
        json.dump(ts_map, f)


def fetch_latest_ohlcv_to_tdengine(
    symbol: str = 'BTC',
    tsym: str = 'USDT',
    table_name: str = 'btc_usdt_kline',
    limit: int = 100
):
    url = 'https://min-api.cryptocompare.com/data/v2/histominute'
    last_ts = load_last_timestamp(symbol, tsym)
    
    params = {
        'fsym': symbol,
        'tsym': tsym,
        'limit': limit,
        'aggregate': 1,
        # 'api_key': 'a4b8b45f304ed64ff3a5f00a1347104dfd6e90ebba569531b07423dadca93a6e'
    }
    result = robust_request(url, params)
    # print(result)

    if not result:
        print(f"❌ {symbol}_{tsym} 拉取失败")
        return

    raw_data = result.get('Data', {}).get('Data', [])
    if not raw_data:
        print(f"⚠️ {symbol}_{tsym} 无数据返回")
        return

    df = pd.DataFrame(raw_data)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert('Asia/Shanghai')
    df.set_index('time', inplace=True)
    df = df[df.index.notnull()]

    if df.empty:
        print(f"⚠️ {symbol}_{tsym} 数据为空")
        return

    # 写入 TDengine
    insert_df_to_tdengine(df, table_name=table_name, symbol=symbol, market=tsym)

    # 保存最后时间戳
    max_ts = int(df.index.max().timestamp())
    save_last_timestamp(symbol, tsym, max_ts)

    print(f"✅ {symbol}_{tsym} 已写入 {len(df)} 条数据（{df.index.min()} ~ {df.index.max()}）")


if __name__ == '__main__':
    coin_pairs = [
        ("BTC", "USDT"),
        ("ETH", "USDT"),
        ("BNB", "USDT"),
        ("SOL", "USDT"),
        ("XRP", "USDT"),
        ("DOGE", "USDT"),
        ("ADA", "USDT"),
        ("TON", "USDT"),
        ("AVAX", "USDT"),
        ("SHIB", "USDT"),
        ("TRX", "USDT"),
        ("LINK", "USDT"),
        ("DOT", "USDT"),
        ("MATIC", "USDT"),
        ("LTC", "USDT"),
        ("WBTC", "USDT"),
        ("BCH", "USDT"),
        ("UNI", "USDT"),
        ("ICP", "USDT"),
        ("ETC", "USDT"),
    ]

    for base, quote in coin_pairs:
        table = f"{base.lower()}_{quote.lower()}_kline"
        fetch_latest_ohlcv_to_tdengine(symbol=base, tsym=quote, table_name=table, limit=100)
    
    release_all()
