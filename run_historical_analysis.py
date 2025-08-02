import argparse
from datetime import datetime, timedelta, timezone
import pandas as pd
import pytz  # ★ 新增

from analyze.calc_indicator import EnhancedMultiTimeframeStrategy
from data.td_dao import query_df_from_tdengine

# ========= 时区设置 =========
LOCAL_TZ = pytz.timezone("Asia/Shanghai")  # ★ 本地时区

def ensure_local(ts) -> pd.Timestamp:
    """把任意 datetime/Timestamp 统一为 Asia/Shanghai 的 pandas.Timestamp（带时区）。"""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize(LOCAL_TZ)  # ★
    return ts.tz_convert(LOCAL_TZ)       # ★

def ensure_utc(ts) -> pd.Timestamp:
    """把任意 datetime/Timestamp 统一为 UTC 带时区的 pandas.Timestamp（用于数据库查询等）。"""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")

# ========= 先触达 TP/SL 判定（主周期，外部时间统一北京时区）=========
def first_touch_outcome_on_primary(strat, entry_time, side, targets, lookahead_hours=48):
    """
    以北京时区进行时间计算与比较；查询 TDengine 时转换为 UTC。
    规则：SL 优先（保守）。
    """
    try:
        entry_time_local = ensure_local(entry_time)               # ★ 统一为本地时区
        end_eval_local = ensure_local(entry_time_local + timedelta(hours=lookahead_hours))  # ★
        end_eval_utc = ensure_utc(end_eval_local)                 # ★ 查询用 UTC

        # 主周期表名
        table = strat._table_name(strat.primary_tf) if hasattr(strat, "_table_name") \
            else strat.symbol.replace('/', '_').lower() + "_kline"

        # 查询到回看上限（UTC）
        df_eval = query_df_from_tdengine(table, strat.primary_tf, end_eval_utc)  # ★
        if df_eval is None or len(df_eval) == 0:
            return {'outcome': 'NONE', 'exit_time': None, 'exit_price': None}

        # 规范化索引 -> 先解析为 UTC，再转北京时区
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
        df = df.sort_index().tz_convert(LOCAL_TZ)  # ★ 转北京时区

        # 只看 entry_time 之后（北京时区）
        future = df[df.index > entry_time_local]  # ★
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
# ……前略（保持你“北京时区”版本的其他代码不变）……


# ========= 主流程（外部时间统一北京时区）=========
def run_historical_analysis(args):
    """
    历史回放入口：
    - 外部（打印/CSV/比较）统一使用北京时区（Asia/Shanghai）
    - TDengine 查询仍使用内部工具在函数内进行 UTC 转换（由 first_touch_* 内部完成）
    - 无信号不写入结果（直接跳过）
    - 收尾输出总体与分类统计（signal_type / side / outcome）
    """
    print("开始历史数据分析...（时区：北京/Asia/Shanghai）")

    # -------- 北京时区的回看窗口 --------
    end_time_local = ensure_local(datetime.now())
    start_time_local = end_time_local - timedelta(days=5)

    # 逐小时时间点（皆为 tz-aware 北京时区）
    time_points = []
    # 对齐到整点
    end_time_local = end_time_local.replace(minute=0, second=0, microsecond=0)
    current_time = start_time_local.replace(minute=0, second=0, microsecond=0)
    while current_time <= end_time_local:
        time_points.append(current_time)
        current_time += timedelta(hours=1)

    print(f"将从 {start_time_local.isoformat()} 到 {end_time_local.isoformat()} 分析 {len(time_points)} 个时间点")

    rows = []

    for i, time_point in enumerate(time_points):
        print(f"\n[{i+1}/{len(time_points)}] 分析时间(北京): {time_point.isoformat()}")

        try:
            strat = EnhancedMultiTimeframeStrategy(
                symbol=args.symbol,
                primary_tf=args.primary,
                secondary_tf=args.secondary,
                tertiary_tf=args.tertiary,
                end_time=time_point,              # 传入 tz-aware 北京时区
                table_with_tf=args.table_with_tf,
                confirmation_required=args.confirm,
                rr_tp_index=args.rr_tp,
            )

            signal = strat.multi_timeframe_analysis()

            # —— 无信号：不保存，直接下一个时间点 ——
            if not (signal.get('long') or signal.get('short')):
                print("  ⏳ 无信号（不写入结果）")
                continue

            # 有信号
            side = 'long' if signal.get('long') else 'short'
            row = {
                'timestamp': time_point,  # 北京时区
                'signal_type': signal.get('type', '未知'),
                'side': side,
                'long': signal.get('long', False),
                'short': signal.get('short', False),
                'confidence': signal.get('confidence', 0),
                'timeframes': ', '.join(signal.get('timeframes', [])),
            }

            # 计算入场/止损/止盈
            targets = strat.calculate_price_targets(signal)
            row.update({
                'entry_price': targets.get('entry', 0.0),
                'stop_loss': targets.get('stop_loss', 0.0),
                'take_profit_1': (targets.get('take_profit') or [0.0])[0],
                'take_profit_2': (targets.get('take_profit') or [0.0, 0.0])[1] if len(targets.get('take_profit') or []) > 1 else 0.0,
                'take_profit_3': (targets.get('take_profit') or [0.0, 0.0, 0.0])[2] if len(targets.get('take_profit') or []) > 2 else 0.0,
                'rr_ratio': targets.get('rr_ratio', 0.0),
            })

            # 进场时间：主周期最后一根K线的时间（若无则用 time_point），统一本地时区
            primary_df = strat.data.get(strat.primary_tf)
            if primary_df is not None and not primary_df.empty:
                entry_time = ensure_local(primary_df.index[-1])
            else:
                entry_time = ensure_local(time_point)

            # 先触达 TP/SL（主周期判定；内部会做时区/查询处理）
            outcome = first_touch_outcome_on_primary(strat, entry_time, side, targets, lookahead_hours=48)

            # R 与收益率
            entry = float(targets.get('entry', 0.0))
            sl = float(targets.get('stop_loss', 0.0))
            risk = abs(entry - sl) or 1e-8
            exit_p = float(outcome['exit_price']) if outcome['exit_price'] is not None else entry

            if outcome['outcome'] == 'SL':
                r_multiple = -1.0
                ret_pct = (exit_p - entry) / entry if side == 'long' else (entry - exit_p) / entry
            elif isinstance(outcome['outcome'], str) and outcome['outcome'].startswith('TP'):
                idx = int(outcome['outcome'][-1]) - 1
                rr_list = targets.get('rr_all', []) or []
                r_multiple = float(rr_list[idx]) if idx < len(rr_list) else max(0.0, abs(exit_p - entry) / risk)
                ret_pct = (exit_p - entry) / entry if side == 'long' else (entry - exit_p) / entry
            else:
                r_multiple = 0.0
                ret_pct = 0.0

            row.update({
                'entry_time': entry_time,            # 北京时区
                'outcome': outcome['outcome'],       # TP1/TP2/TP3/SL/NONE
                'exit_time': outcome['exit_time'],   # 北京时区
                'exit_price': outcome['exit_price'],
                'r_multiple': r_multiple,
                'ret_pct': ret_pct,
            })

            print(f"  ✅ 信号: {signal['type']} | {side.upper()} | Outcome: {outcome['outcome']} | R: {r_multiple:.2f}")
            rows.append(row)

        except Exception as e:
            print(f"  ❌ 分析失败: {e}")
            # 如不希望保留失败行，可改为 continue
            rows.append({
                'timestamp': time_point, 'signal_type': '分析失败',
                'side': None, 'long': False, 'short': False, 'confidence': 0, 'timeframes': '',
                'entry_price': 0.0, 'stop_loss': 0.0,
                'take_profit_1': 0.0, 'take_profit_2': 0.0, 'take_profit_3': 0.0,
                'rr_ratio': 0.0, 'entry_time': None,
                'outcome': '失败', 'exit_time': None, 'exit_price': 0.0,
                'r_multiple': 0.0, 'ret_pct': 0.0,
            })

    # ====== 保存 CSV：只包含“有信号”的记录（以及可选的失败记录） ======
    import pandas as pd
    df = pd.DataFrame(rows)
    # 如不想包含“分析失败”行，取消注释下一行：
    # df = df[df['signal_type'] != '分析失败']
    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\n✅ 历史分析完成，结果保存至：{args.output}")

    # ====== 汇总统计（仅对有效交易记录进行） ======
    trades = df[df['entry_price'] > 0].copy()
    total_R = trades['r_multiple'].sum()
    avg_R = trades['r_multiple'].mean() if len(trades) else 0.0
    winrate = (trades['r_multiple'] > 0).mean() if len(trades) else 0.0
    avg_ret_pct = trades['ret_pct'].mean() if len(trades) else 0.0
    sum_ret_pct = trades['ret_pct'].sum() if len(trades) else 0.0

    print("\n=== 回报统计（仅含有信号的交易，先触 TP/SL，时间：北京时区）===")
    print(f"样本笔数: {len(trades)}")
    print(f"胜率: {winrate:.2%}")
    print(f"平均R: {avg_R:.3f}, 总R: {total_R:.3f}")
    print(f"平均收益: {avg_ret_pct:.3%}, 累计简单收益: {sum_ret_pct:.3%}")

    # ====== 分类统计 ======
    if not trades.empty:
        # 按 signal_type
        print("\n--- 按 signal_type 分类 ---")
        for k, t in trades.groupby('signal_type'):
            n = len(t)
            wr = (t['r_multiple'] > 0).mean()
            avgR = t['r_multiple'].mean()
            sumR = t['r_multiple'].sum()
            avgRet = t['ret_pct'].mean()
            sumRet = t['ret_pct'].sum()
            print(f"{k}: 样本={n}, 胜率={wr:.2%}, 平均R={avgR:.3f}, 总R={sumR:.3f}, 平均收益={avgRet:.3%}, 累计收益={sumRet:.3%}")

        # 按 side（long/short）
        print("\n--- 按 side（多空）分类 ---")
        for k, t in trades.groupby('side'):
            n = len(t)
            wr = (t['r_multiple'] > 0).mean()
            avgR = t['r_multiple'].mean()
            sumR = t['r_multiple'].sum()
            avgRet = t['ret_pct'].mean()
            sumRet = t['ret_pct'].sum()
            print(f"{k}: 样本={n}, 胜率={wr:.2%}, 平均R={avgR:.3f}, 总R={sumR:.3f}, 平均收益={avgRet:.3%}, 累计收益={sumRet:.3%}")

        # 按 outcome（TP1/TP2/TP3/SL）
        print("\n--- 按 outcome（TP/SL）分类 ---")
        for k, t in trades.groupby('outcome'):
            n = len(t)
            avgR = t['r_multiple'].mean()
            sumR = t['r_multiple'].sum()
            avgRet = t['ret_pct'].mean()
            sumRet = t['ret_pct'].sum()
            print(f"{k}: 样本={n}, 平均R={avgR:.3f}, 总R={sumR:.3f}, 平均收益={avgRet:.3%}, 累计收益={sumRet:.3%}")

    return df

# ========= CLI =========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Historical hourly replay for Enhanced Multi-Timeframe Strategy (Asia/Shanghai)')
    parser.add_argument('--symbol', default='BTC/USDT')
    parser.add_argument('--primary', default='15h')
    parser.add_argument('--secondary', default='1h')
    parser.add_argument('--tertiary', default='4h')
    parser.add_argument('--table-with-tf', action='store_true', help='Table name includes timeframe suffix')
    parser.add_argument('--no-table-with-tf', dest='table_with_tf', action='store_false')
    parser.set_defaults(table_with_tf=True)
    parser.add_argument('--confirm', type=int, default=0, help='确认次数（>N 触发），历史回放建议 0 或 1')
    parser.add_argument('--rr-tp', type=int, default=1, help='RR 展示使用的 TP 索引(0/1/2)')
    parser.add_argument('--output', default='historical_results.csv')

    args = parser.parse_args()
    run_historical_analysis(args)
