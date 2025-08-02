from datetime import datetime
from data.td_connect import cursor, conn
import pandas as pd

def insert_df_to_tdengine(df, table_name: str, symbol: str, market: str):
    for ts, row in df.iterrows():
        # 转换时间为字符串（确保是 UTC 时间或你创建表时一致）
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")

        # 提取各列数据
        open_ = row['open']
        high = row['high']
        low = row['low']
        close = row['close']
        volumefrom = row['volumefrom']
        volumeto = row['volumeto']

        sql = f"""
        INSERT INTO {table_name}
        USING kline
        TAGS ('{symbol}', '{market}')
        VALUES ('{ts_str}', {open_}, {high}, {low}, {close}, {volumefrom}, {volumeto});
        """

        cursor.execute(sql)
        # cursor.close()
        # conn.close()
        
def query_df_from_tdengine(table_name: str, tf, end_time=datetime.now(), day='100d') -> pd.DataFrame:
    # 归一化并校验 INTERVAL：支持 5 / "5" / "5m" / "30s" / "1h" 等
    # interval = str(tf).lower().strip()
    time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

    sql = f"""
   
    SELECT
      _wstart AS ts,
      first(open)  AS open,
      max(high)    AS high,
      min(low)     AS low,
      last(close)  AS close,
      sum(volumefrom) AS volume
    FROM {table_name}
    WHERE  ts < '{end_time}'
    INTERVAL('{tf}')
    ORDER BY ts desc
    """
    cursor.execute(sql)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    # 空结果时返回带索引的空 DataFrame，便于上层统一处理
    if not rows:
        empty = pd.DataFrame(columns=["time"] + [c for c in columns if c != "ts"])
        empty.set_index(pd.DatetimeIndex([], name="time"), inplace=True)
        return empty

    df = pd.DataFrame(rows, columns=columns)

    # 统一时间索引
    if "ts" in df.columns:
        df["time"] = pd.to_datetime(df["ts"])
        df.set_index("time", inplace=True)
        df.drop(columns=["ts"], inplace=True)

    # 数值列转为数值类型（防止驱动返回为字符串）
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df




