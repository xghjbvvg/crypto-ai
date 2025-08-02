import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from typing import Optional

from data.td_dao import insert_df_to_tdengine


def fetch_cryptocompare_ohlcv_to_tdengine(
    symbol: str,
    tsym: str,
    start_dt: datetime = datetime(2025, 1, 1),
    end_dt: datetime = datetime(2025, 8, 1),
    table_name: str = "btc_usdt_kline",
    limit: int = 24
):
    """
    拉取 CryptoCompare 的历史 OHLCV 数据并保存到 TDengine。

    :param symbol: 交易对基础币种（如 'BTC'）
    :param tsym: 交易对计价币种（如 'USDT'）
    :param granularity: 时间粒度（如 '1h', '1d'）
    :param start_dt: 起始时间（北京时间）
    :param end_dt: 结束时间（北京时间）
    :param table_name: TDengine 中的表名
    :param limit: 每次请求的条数（1h 时建议 24，1d 时建议 1）
    """


    # 统一转换为 UTC（CryptoCompare 接口要求 UTC）
    start_dt_utc = (start_dt - timedelta(hours=8))
    end_dt_utc = (end_dt - timedelta(hours=8))

    to_ts = int(end_dt_utc.timestamp())

    while True:
        url = 'https://min-api.cryptocompare.com/data/v2/histohour'
        params = {
            'fsym': symbol,
            'tsym': tsym,
            'limit': limit,
            'aggregate': 1,
            'toTs': to_ts
        }

        response = requests.get(url, params=params)
        result = response.json()
        raw_data = result.get('Data', {}).get('Data', [])

        if not raw_data:
            print("❌ 没有返回数据，可能到头或被限流。")
            break

        df = pd.DataFrame(raw_data)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df['time'] = df['time'].dt.tz_convert('Asia/Shanghai')  # 转为北京时间
        df.set_index('time', inplace=True)

        if df.empty or pd.isna(df.index.min()):
            print("⚠️ 分页数据为空或无时间，终止。")
            break

        # 写入 TDengine
        insert_df_to_tdengine(df, table_name=table_name, symbol=symbol, market=tsym)

        print(f"✅ 已处理: （{df.index.min()} ~ {df.index.max()}）共 {len(df)} 条")

        # 判断是否到底
        if df.index.min().tz_localize(None) <= start_dt:
            print("✅ 到达起始日期，停止。")
            break

        # 下一页时间戳
        to_ts = int(df.index.min().tz_convert('UTC').timestamp()) - 1
        time.sleep(2)  # 防止限流


# ✅ 使用示例
if __name__ == '__main__':
    start_dt = datetime(2025, 1, 1, 0, 0, 0) - timedelta(hours=8)
    end_dt = datetime(2025, 8, 1, 23, 59, 0) - timedelta(hours=8)

    fetch_cryptocompare_ohlcv_to_tdengine(
        symbol='BTC',
        tsym='USDT',
        start_dt=start_dt,
        end_dt=end_dt,
        table_name='btc_usdt_kline',
        limit=24
    )
