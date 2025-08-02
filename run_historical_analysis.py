import argparse
from datetime import datetime, timedelta, timezone
import pandas as pd

from analyze.calc_indicator import EnhancedMultiTimeframeStrategy
from data.td_dao import query_df_from_tdengine


# ========= 时区工具 =========
def ensure_utc(ts) -> pd.Timestamp:
    """把任意 datetime/Timestamp 统一为 UTC 带时区的 pandas.Timestamp。"""
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


# ========= 先触达 TP/SL 判定（主周期）=========
def first_touch_outcome_on_primary(strat, entry_time, side, targets, lookahead_hours=48):
    """
    用主周期数据，在 entry_time 之后、最多 lookahead_hours 小时内，
    判定先触达 TP/SL。返回 dict: outcome, exit_time, exit_price
    规则：SL 优先（保守）。
    """
    try:
        entry_time = ensure_utc(entry_time)
        end_eval = ensure_utc(entry_time + timedelta(hours=lookahead_hours))

        # 主周期表名
        table = strat._table_name(strat.primary_tf) if hasattr(strat, "_table_name") \
            else strat.symbol.replace('/', '_').lower() + "_kline"

        # 查询到回看上限
        df_eval = query_df_from_tdengine(table, strat.primary_tf, end_eval)
        if df_eval is None or len(df_eval) == 0:
            return {'outcome': 'NONE', 'exit_time': None, 'exit_price': None}

        # 规范化索引（UTC 带时区）
        df = pd.DataFrame(df_eval)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if 'timestamp' in df.columns:
            ts0 = pd.to_numeric(df['timestamp'].iloc[0], errors='coerce')
            unit = 'ms' if pd.notna(ts0) and int(ts0) > 10**11 else 's'
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit=unit, utc=True, errors='coerce')
            df = df.set_index('timestamp')
        if not isinstance(df.index, pd.DatetimeIndex):
            return {'outcome': 'NONE', 'exit_time': None, 'exit_price': None}
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df = df.sort_index()

        # 只看 entry_time 之后
        future = df[df.index > entry_time]
        if future.empty:
            return {'outcome': 'NONE', 'exit_time': None, 'exit_price': None}

        sl = float(targets['stop_loss'])
        tps = list(map(float, targets.get('take_profit', [])))

        for ts, row in future.iterrows():
            hi, lo = float(row['high']), float(row['low'])
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
    except Exception as e:
        print(f"⚠️ 先触达判定失败：{e}")

    return {'outcome': 'NONE', 'exit_time': None, 'exit_price': None}


# ========= 主流程 =========
def run_historical_analysis(args):
    print("开始历史数据分析...")

    # 三个月窗口（UTC 带时区）
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=90)

    # 逐小时时间点（皆为 tz-aware UTC）
    time_points = []
    current_time = start_time.replace(minute=0, second=0, microsecond=0)
    end_time = end_time.replace(minute=0, second=0, microsecond=0)
    while current_time <= end_time:
        time_points.append(current_time)
        current_time += timedelta(hours=1)

    print(f"将从 {start_time} 到 {end_time} 分析 {len(time_points)} 个时间点")

    rows = []

    for i, time_point in enumerate(time_points):
        print(f"\n[{i+1}/{len(time_points)}] 分析时间(UTC): {time_point.isoformat()}")

        try:
            strat = EnhancedMultiTimeframeStrategy(
                symbol=args.symbol,
                primary_tf=args.primary,
                secondary_tf=args.secondary,
                tertiary_tf=args.tertiary,
                end_time=time_point,              # ✅ 传入 tz-aware UTC
                table_with_tf=args.table_with_tf,
                confirmation_required=args.confirm,
                rr_tp_index=args.rr_tp,
            )

            signal = strat.multi_timeframe_analysis()

            row = {
                'timestamp': time_point,  # tz-aware
                'signal_type': signal.get('type', '无信号'),
                'long': signal.get('long', False),
                'short': signal.get('short', False),
                'confidence': signal.get('confidence', 0),
                'timeframes': ', '.join(signal.get('timeframes', [])),
            }

            if signal.get('long') or signal.get('short'):
                targets = strat.calculate_price_targets(signal)
                row.update({
                    'entry_price': targets.get('entry', 0.0),
                    'stop_loss': targets.get('stop_loss', 0.0),
                    'take_profit_1': (targets.get('take_profit') or [0.0])[0],
                    'take_profit_2': (targets.get('take_profit') or [0.0, 0.0])[1] if len(targets.get('take_profit') or []) > 1 else 0.0,
                    'take_profit_3': (targets.get('take_profit') or [0.0, 0.0, 0.0])[2] if len(targets.get('take_profit') or []) > 2 else 0.0,
                    'rr_ratio': targets.get('rr_ratio', 0.0),
                })

                # 进场时间：主周期最后一根时间（tz-aware），否则用 time_point
                primary_df = strat.data.get(strat.primary_tf)
                if primary_df is not None and not primary_df.empty:
                    entry_time = ensure_utc(primary_df.index[-1])
                else:
                    entry_time = ensure_utc(time_point)

                side = 'long' if signal.get('long') else 'short'
                outcome = first_touch_outcome_on_primary(strat, entry_time, side, targets, lookahead_hours=48)

                # 计算 R 倍数 和 百分比收益
                r_multiple = 0.0
                ret_pct = 0.0
                entry = float(targets.get('entry', 0.0))
                sl = float(targets.get('stop_loss', 0.0))
                risk = abs(entry - sl) or 1e-8
                exit_p = float(outcome['exit_price']) if outcome['exit_price'] is not None else entry

                if outcome['outcome'] == 'SL':
                    r_multiple = -1.0
                    ret_pct = (exit_p - entry) / entry if side == 'long' else (entry - exit_p) / entry
                elif outcome['outcome'].startswith('TP'):
                    idx = int(outcome['outcome'][-1]) - 1  # TP1/2/3 -> 0/1/2
                    rr_list = targets.get('rr_all', []) or []
                    r_multiple = float(rr_list[idx]) if idx < len(rr_list) else max(0.0, abs(exit_p - entry) / risk)
                    ret_pct = (exit_p - entry) / entry if side == 'long' else (entry - exit_p) / entry
                else:
                    r_multiple = 0.0
                    ret_pct = 0.0

                row.update({
                    'entry_time': entry_time,
                    'outcome': outcome['outcome'],
                    'exit_time': outcome['exit_time'],
                    'exit_price': outcome['exit_price'],
                    'r_multiple': r_multiple,
                    'ret_pct': ret_pct,
                })
                print(f"  ✅ 信号: {signal['type']} | Outcome: {outcome['outcome']} | R: {r_multiple:.2f}")
            else:
                row.update({
                    'entry_price': 0.0, 'stop_loss': 0.0,
                    'take_profit_1': 0.0, 'take_profit_2': 0.0, 'take_profit_3': 0.0,
                    'rr_ratio': 0.0, 'entry_time': None,
                    'outcome': '无信号', 'exit_time': None, 'exit_price': 0.0,
                    'r_multiple': 0.0, 'ret_pct': 0.0,
                })
                print("  ⏳ 无信号")

            rows.append(row)

        except Exception as e:
            print(f"  ❌ 分析失败: {e}")
            rows.append({
                'timestamp': time_point, 'signal_type': '分析失败',
                'long': False, 'short': False, 'confidence': 0, 'timeframes': '',
                'entry_price': 0.0, 'stop_loss': 0.0,
                'take_profit_1': 0.0, 'take_profit_2': 0.0, 'take_profit_3': 0.0,
                'rr_ratio': 0.0, 'entry_time': None,
                'outcome': '失败', 'exit_time': None, 'exit_price': 0.0,
                'r_multiple': 0.0, 'ret_pct': 0.0,
            })

    # 保存 CSV（含 tz 信息）
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\n✅ 历史分析完成，结果保存至：{args.output}")

    # 统计回报
    trades = df[df['entry_price'] > 0].copy()
    total_R = trades['r_multiple'].sum()
    avg_R = trades['r_multiple'].mean() if len(trades) else 0.0
    winrate = (trades['r_multiple'] > 0).mean() if len(trades) else 0.0
    avg_ret_pct = trades['ret_pct'].mean() if len(trades) else 0.0
    sum_ret_pct = trades['ret_pct'].sum() if len(trades) else 0.0

    print("\n=== 回报统计（基于先触 TP/SL）===")
    print(f"样本笔数: {len(trades)}")
    print(f"胜率: {winrate:.2%}")
    print(f"平均R: {avg_R:.3f}, 总R: {total_R:.3f}")
    print(f"平均收益: {avg_ret_pct:.3%}, 累计简单收益: {sum_ret_pct:.3%}")

    return df


# ========= CLI =========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Historical hourly replay for Enhanced Multi-Timeframe Strategy')
    parser.add_argument('--symbol', default='BTC/USDT')
    parser.add_argument('--primary', default='1h')
    parser.add_argument('--secondary', default='4h')
    parser.add_argument('--tertiary', default='1d')
    parser.add_argument('--table-with-tf', action='store_true', help='Table name includes timeframe suffix')
    parser.add_argument('--no-table-with-tf', dest='table_with_tf', action='store_false')
    parser.set_defaults(table_with_tf=True)
    parser.add_argument('--confirm', type=int, default=0, help='确认次数（>N 触发），历史回放建议 0 或 1')
    parser.add_argument('--rr-tp', type=int, default=1, help='RR 展示使用的 TP 索引(0/1/2)')
    parser.add_argument('--output', default='historical_results.csv')

    args = parser.parse_args()
    run_historical_analysis(args)
