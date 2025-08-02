"""
Outcome evaluation helpers for EnhancedMultiTimeframeStrategy historical replay.
Assumes a DAO function `query_df_from_tdengine(table, timeframe, end_time)` is available.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Dict, Any
import pandas as pd

try:
    # 按你的工程组织：data/td_dao.py 提供该函数
    from data.td_dao import query_df_from_tdengine
except Exception as e:
    raise ImportError("Cannot import query_df_from_tdengine from data.td_dao. Please ensure it exists.")


def _normalize_ohlcv_df(df_like) -> pd.DataFrame:
    """将 DAO 返回的数据规范为以 UTC DatetimeIndex 为索引的 DataFrame。"""
    if df_like is None:
        return pd.DataFrame()
    df = pd.DataFrame(df_like)
    if df.empty:
        return df

    df.columns = [str(c).strip().lower() for c in df.columns]
    if 'timestamp' in df.columns:
        ts0 = pd.to_numeric(df['timestamp'].iloc[0], errors='coerce')
        if pd.notna(ts0):
            unit = 'ms' if int(ts0) > 10**11 else 's'
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit=unit, utc=True)
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df = df.set_index('timestamp')
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df.sort_index()


def first_touch_outcome_on_primary(
    strat,
    entry_time: pd.Timestamp,
    side: str,
    targets: Dict[str, Any],
    lookahead_hours: int = 48,
) -> Dict[str, Any]:
    """
    在主周期上，从 entry_time 之后开始向未来最多 lookahead_hours 小时，判断先触 TP/SL。
    规则：SL 优先（更保守）。
    返回: {'outcome': 'TP1'/'TP2'/'TP3'/'SL'/'NONE', 'exit_time': ts, 'exit_price': p}
    """
    end_eval = (entry_time.tz_convert('UTC') if entry_time.tzinfo else entry_time.replace(tzinfo=timezone.utc)) + timedelta(hours=lookahead_hours)

    # 表名与主TF
    try:
        table = strat._table_name(strat.primary_tf)
    except Exception:
        table = strat.symbol.replace('/', '_').lower() + "_kline"

    df_eval = query_df_from_tdengine(table, strat.primary_tf, end_eval)
    df = _normalize_ohlcv_df(df_eval)
    if df.empty:
        return {'outcome': 'NONE', 'exit_time': None, 'exit_price': None}

    future = df[df.index > entry_time]
    if future.empty:
        return {'outcome': 'NONE', 'exit_time': None, 'exit_price': None}

    sl = float(targets.get('stop_loss', 0.0))
    tps = [float(x) for x in targets.get('take_profit', [])]

    for ts, row in future.iterrows():
        hi = float(row['high']); lo = float(row['low'])
        if side == 'long':
            if lo <= sl:
                return {'outcome': 'SL', 'exit_time': ts, 'exit_price': sl}
            for i, tp in enumerate(tps, start=1):
                if hi >= tp:
                    return {'outcome': f'TP{i}', 'exit_time': ts, 'exit_price': tp}
        else:  # short
            if hi >= sl:
                return {'outcome': 'SL', 'exit_time': ts, 'exit_price': sl}
            for i, tp in enumerate(tps, start=1):
                if lo <= tp:
                    return {'outcome': f'TP{i}', 'exit_time': ts, 'exit_price': tp}

    return {'outcome': 'NONE', 'exit_time': None, 'exit_price': None}
