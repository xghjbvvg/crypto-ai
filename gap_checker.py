import requests
import pandas as pd
from datetime import datetime, timedelta
from data.td_dao import insert_df_to_tdengine, query_df_from_tdengine

class GapChecker:
    def __init__(self, symbol: str, tsym: str, table_name: str, interval_minutes: int = 1):
        self.symbol = symbol
        self.tsym = tsym
        self.table_name = table_name
        self.interval = timedelta(minutes=interval_minutes)
        self.url = 'https://min-api.cryptocompare.com/data/v2/histominute'
        self.limit = 100

    def detect_gaps(self, df: pd.DataFrame):
        if df.empty:
            return []

        df = df.sort_index()
        time_diffs = df.index.to_series().diff()
        gaps = time_diffs[time_diffs > self.interval]

        result = []
        for i in range(len(gaps)):
            if gaps.iloc[i] > self.interval:
                gap_start = df.index[i - 1] + self.interval
                gap_end = df.index[i]
                result.append((gap_start, gap_end))
        return result

    def fetch_and_insert(self, to_ts: int):
        params = {
            'fsym': self.symbol,
            'tsym': self.tsym,
            'limit': self.limit,
            'aggregate': 1,
            'toTs': to_ts
        }

        try:
            response = requests.get(self.url, params=params, timeout=10)
            result = response.json()
            raw_data = result.get('Data', {}).get('Data', [])
        except Exception as e:
            print(f"❌ 请求失败: {e}")
            return False

        if not raw_data:
            print("⚠️ 无数据返回")
            return False

        df = pd.DataFrame(raw_data)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert('Asia/Shanghai')
        df.set_index('time', inplace=True)
        df = df[df.index.notnull()]
        if df.empty:
            print("⚠️ 返回数据为空")
            return False

        insert_df_to_tdengine(df, table_name=self.table_name, symbol=self.symbol, market=self.tsym)
        print(f"✅ 补采写入 {len(df)} 条数据：{df.index.min()} ~ {df.index.max()}")
        return True

    def repair(self, gap_start, gap_end):
        print(f"🔧 开始补采断层：{gap_start} ~ {gap_end}")
        current = gap_start
        while current < gap_end:
            to_ts = int((current + timedelta(minutes=self.limit - 1)).timestamp())
            if not self.fetch_and_insert(to_ts):
                break
            current += timedelta(minutes=self.limit)
        print(f"✅ 补采完成：{gap_start} ~ {gap_end}")

    def run(self, rows=1000):
        df = query_df_from_tdengine(self.table_name, limit=rows)
        if df is None or df.empty:
            print(f"⚠️ 无法获取 {self.table_name} 的数据")
            return

        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        gaps = self.detect_gaps(df)
        if not gaps:
            print(f"✅ 未发现断层：{self.table_name}")
            return

        print(f"⛔ 发现断层 {len(gaps)} 个：{gaps}")
        for start, end in gaps:
            self.repair(start, end)


coin_pairs = [
    ("BTC", "USDT"),
    ("ETH", "USDT"),
]

for base, quote in coin_pairs:
    table = f"{base.lower()}_{quote.lower()}_kline"
    checker = GapChecker(base, quote, table)
    checker.run(rows=1000)
