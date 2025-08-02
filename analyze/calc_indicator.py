"""
Enhanced Multi-Timeframe Strategy (TDengine + TA-Lib + matplotlib)
- Loads OHLCV from TDengine via query_df_from_tdengine(table, timeframe, end_time)
- Computes indicators per timeframe (no forced reindex/resample across TFs)
- Generates per-TF signals and aggregates into multi-timeframe consensus
- Calculates stops/targets and plots a three-panel chart

Requirements:
  pip install pandas numpy TA-Lib ccxt matplotlib mplfinance

Notes:
- Binance K-line boundaries are aligned to UTC. This script schedules next run
  against UTC timeframe boundaries.
- Ensure data.td_dao.query_df_from_tdengine(table, tf, end_time) returns a DataFrame
  with columns: ['timestamp','open','high','low','close','volume'] or an index
  already set to a datetime index. Volume must be numeric.
"""

import os
import sys
import time
import logging
import traceback
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import talib
import ccxt

# Use a headless backend for servers/containers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates

# Make local imports work when run as a script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.td_dao import query_df_from_tdengine

# ------------------------------ Logging ----------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_strategy.log'),
        logging.StreamHandler()
    ]
)


class EnhancedMultiTimeframeStrategy:
    def __init__(
        self,
        symbol: str = 'BTC/USDT',
        primary_tf: str = '1h',
        secondary_tf: str = '4h',
        tertiary_tf: str = '1d',
        end_time = datetime.now(),
        *,
        table_with_tf: bool = True,
        confirmation_required: int = 1,  # relaxed: allow immediate trigger in tests
        rr_tp_index: int = 1,            # index 0/1/2 used to compute displayed RR
    ):
        """Initialize enhanced multi-timeframe strategy.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT').
            primary_tf: Signal-generation timeframe.
            secondary_tf: Trend-confirmation timeframe.
            tertiary_tf: Macro trend timeframe.
            table_with_tf: If True, expect TDengine table name to include timeframe
                           as f"{symbol}_{tf}_kline"; otherwise use f"{symbol}_kline".
            confirmation_required: Number of consecutive scans with the same side
                           required before confirming combined signal.
            rr_tp_index: Which TP (0,1,2) to use when reporting RR.
        """
        self.symbol = symbol
        self.primary_tf = primary_tf
        self.secondary_tf = secondary_tf
        self.tertiary_tf = tertiary_tf
        self.table_with_tf = table_with_tf
        self.confirmation_required = max(0, int(confirmation_required))
        self.rr_tp_index = max(0, min(2, int(rr_tp_index)))

        self.last_signal = None
        self.signal_confirmation_count = 0
        self.end_time = end_time or datetime.utcnow()

        # If you later need exchange price or server time; not used for data fetch now
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}  # perpetual futures
        })

        # Timeframe conversion factors relative to 1h
        self.tf_conversion = {
            '1d': 24,
            '4h': 6,
            '1h': 1,
            '30m': 0.5,
            '15m': 0.25,
            '5m': 0.083
        }

        self.data: dict[str, pd.DataFrame] = {}
        self.load_data()

    # -------------------------- Data Loading ------------------------------
    def _table_name(self, tf: str) -> str:
        base = self.symbol.replace('/', '_').lower()
        return f"{base}_kline"

    def load_data(self) -> None:
        """Load OHLCV for each timeframe from TDengine and minimally clean."""
        self.data = {}
        for tf in [self.tertiary_tf, self.secondary_tf, self.primary_tf]:
            try:
                table = self._table_name(tf)
                logging.info(f"读取 {self.symbol} {tf} => 表: {table}")
                df = query_df_from_tdengine(table, tf, self.end_time)

                # ---- 调试信息 ----
                try:
                    logging.info(f"[DEBUG] raw type={type(df)}, hasattr_len={hasattr(df, '__len__')}")
                    if isinstance(df, pd.DataFrame):
                        logging.info(f"[DEBUG] raw shape={df.shape}, columns={list(df.columns)[:10]}")
                    else:
                        df = pd.DataFrame(df)
                        logging.info(f"[DEBUG] casted to DataFrame, shape={df.shape}, columns={list(df.columns)[:10]}")
                except Exception:
                    pass

                if df is None or len(df) == 0:
                    logging.warning(f"{table} 返回空数据(初始)")
                    continue

                # ---- 列名标准化 ----
                df.columns = [str(c).strip().lower() for c in df.columns]
                alias_map = {'ts':'timestamp', 'time':'timestamp', 'vol':'volume'}
                df.rename(columns={k:v for k,v in alias_map.items() if k in df.columns}, inplace=True)

                # ---- 统一索引 ----
                if 'timestamp' in df.columns:
                    ts0 = pd.to_numeric(df['timestamp'].iloc[0], errors='coerce')
                    if pd.notna(ts0) and np.isfinite(ts0):
                        ts0 = int(ts0)
                        unit = 'ms' if ts0 > 10**11 else 's'
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit=unit, utc=True)
                    else:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
                    df = df.set_index('timestamp')
                else:
                    if not isinstance(df.index, pd.DatetimeIndex):
                        raise ValueError(f"{table} 需包含 datetime 索引或 timestamp 列")

                df = df.sort_index()

                # ---- 必要列检查 + 转数值 ----
                needed = ['open', 'high', 'low', 'close', 'volume']
                missing = [c for c in needed if c not in df.columns]
                if missing:
                    raise ValueError(f"{table} 缺少必要列: {missing}")

                for col in needed:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

                before_clean = len(df)
                df = df[~df.index.duplicated(keep='last')]
                df = df.dropna(subset=['open','high','low','close','volume'])
                df = df[df['volume'] > 0]
                logging.info(f"[DEBUG] 清洗前行数={before_clean}，去重/NaN/成交量>0 后={len(df)}")

                if len(df) == 0:
                    logging.warning(f"{self.symbol} {tf} 清洗后无数据")
                    continue

                self.data[tf] = df
            except Exception as e:
                logging.error(f"加载数据失败: {tf}, 错误: {str(e)}")
                logging.error(traceback.format_exc())

    # ------------------------ Indicator Computation -----------------------
    def _p(self, x: float) -> int:
        """Period helper: ensure >=1 integer."""
        return max(1, int(round(x)))

    def calculate_indicators(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """计算指标，自动适配样本量，避免因 warmup 过长导致全 NaN。"""
        try:
            if df is None or len(df) == 0:
                return df

            # 确保数值类型，talib 需要 float64
            for col in ['open','high','low','close','volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

            mult = self.tf_conversion.get(tf, 1)
            # 基础周期
            base = {
                'boll': max(1, int(round(20 * mult))),
                'macd_fast': max(1, int(round(12 * mult))),
                'macd_slow': max(1, int(round(26 * mult))),
                'macd_sig':  max(1, int(round(9  * mult))),
                'rsi':       max(1, int(round(14 * mult))),
                'vol_ma':    max(1, int(round(20 * mult))),
                'ema50':     max(1, int(round(50 * mult))),
                'ema200':    max(1, int(round(200 * mult))),
                'atr':       max(1, int(round(14 * mult))),
            }

            # 估算 warmup 需求
            warmup_needed = max(base['ema200'], base['boll'], base['atr'], base['macd_slow'] + base['macd_sig'])

            n = len(df)
            # 如果样本不足，动态下调周期到可用范围的约 1/3
            if n <= warmup_needed:
                scale = max(1, int(n // 3))
                for k in base:
                    base[k] = max(1, min(base[k], scale))
                logging.warning(f"[{tf}] 样本 {n} 小于 warmup {warmup_needed}，已自适应缩短周期: {base}")
                warmup_needed = max(base.values())

            out = df.copy()
            out['upper'], out['middle'], out['lower'] = talib.BBANDS(out['close'], timeperiod=base['boll'], nbdevup=2, nbdevdn=2)
            out['macd'], out['signal'], out['hist']  = talib.MACD(out['close'], fastperiod=base['macd_fast'], slowperiod=base['macd_slow'], signalperiod=base['macd_sig'])
            out['rsi']      = talib.RSI(out['close'], timeperiod=base['rsi'])
            out['vol_ma']   = talib.MA(out['volume'], timeperiod=base['vol_ma'])
            out['ema50']    = talib.EMA(out['close'], timeperiod=base['ema50'])
            out['ema200']   = talib.EMA(out['close'], timeperiod=base['ema200'])
            out['trend']    = np.where(out['ema50'] > out['ema200'], 'Up', 'Down')
            out['atr']      = talib.ATR(out['high'], out['low'], out['close'], timeperiod=base['atr'])
            out['obv']      = talib.OBV(out['close'], out['volume'])
            out['obv_ema']  = talib.EMA(out['obv'], timeperiod=max(1, min(20, n//4)))

            # RSI 反转
            out['rsi_reversal'] = 0
            rsi = out['rsi'].values
            for i in range(1, len(out)):
                if rsi[i-1] < 30 <= rsi[i]:
                    out.iloc[i, out.columns.get_loc('rsi_reversal')] = 1
                elif rsi[i-1] > 70 >= rsi[i]:
                    out.iloc[i, out.columns.get_loc('rsi_reversal')] = -1

            # 只截取 warmup 之后的有效段，收敛 dropna 范围，避免误删
            valid = out.iloc[max(0, warmup_needed-1):]
            valid = valid.dropna(subset=['upper','middle','lower','ema50','ema200','macd','signal','hist','rsi','vol_ma','atr']).tail(1000)
            if len(valid) == 0:
                logging.warning(f"[{tf}] 指标计算后全部为 NaN，返回原数据以便排查。")
                return out
            return valid
        except Exception as e:
            logging.error(f"指标计算失败: {tf}, 错误: {str(e)}")
            logging.error(traceback.format_exc())
            return df

    def detect_obv_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            out = df.copy()
            out['obv_divergence'] = 0
            hh = out['high'].rolling(5, center=True).max()
            ll = out['low'].rolling(5, center=True).min()
            obv_ma = talib.EMA(out['obv'], timeperiod=max(3, min(14, len(out)//6)))

            for i in range(5, len(out)):
                if ll.iloc[i] < ll.iloc[i-5] and obv_ma.iloc[i] >= obv_ma.iloc[i-5]:
                    out.at[out.index[i], 'obv_divergence'] = 1
                elif hh.iloc[i] > hh.iloc[i-5] and obv_ma.iloc[i] <= obv_ma.iloc[i-5]:
                    out.at[out.index[i], 'obv_divergence'] = -1
            return out
        except Exception as e:
            logging.error(f"OBV背离检测失败: {str(e)}")
            return df


    # ---------------------------- Signals ---------------------------------
    def generate_signal(self, df: pd.DataFrame) -> dict:
        try:
            if df is None or len(df) < 3:
                return {'strength': 0, 'long': False, 'short': False, 'conditions': []}

            last = df.iloc[-1]
            prev = df.iloc[-2]

            signal = {
                'strength': 0,
                'long': False,
                'short': False,
                'conditions': [],
                'obv_divergence': 0,
                'rsi_reversal': 0
            }
            vol_ok = last['volume'] > last['vol_ma']

            # Long-side conditions (relaxed)
            trend_conditions = []
            if last['close'] > last['middle'] and last['middle'] > prev['middle']:
                trend_conditions.append('Boll_Up')
            if last['macd'] > last['signal'] and last['hist'] > 0:
                trend_conditions.append('MACD_Up')
            if last['rsi'] > 52:  # relaxed from 50~70
                trend_conditions.append('RSI_Ok')
            if last['trend'] == 'Up' and last['close'] > last['ema50']:
                trend_conditions.append('Trend_Up')

            # Volume as bonus (not a hard gate)
            if last['volume'] > last['vol_ma']:
                signal['strength'] += 1
                signal['conditions'].append('Volume_Up')

            if len(trend_conditions) >= 3:
                signal['long'] = True
                signal['strength'] += len(trend_conditions) + 1  # +1 来自 vol_ok
                signal['conditions'].extend(trend_conditions + ['Volume_Up'])

            # Short-side conditions (relaxed)
            short_conditions = []
            if last['close'] < last['middle'] and last['middle'] < prev['middle']:
                short_conditions.append('Boll_Down')
            if last['macd'] < last['signal'] and last['hist'] < 0:
                short_conditions.append('MACD_Down')
            if last['rsi'] < 48:  # relaxed from 30~50
                short_conditions.append('RSI_Ok')
            if last['trend'] == 'Down' and last['close'] < last['ema50']:
                short_conditions.append('Trend_Down')

            if vol_ok and len(short_conditions) >= 3:
                signal['short'] = True
                signal['strength'] += len(short_conditions) + 1
                signal['conditions'].extend(short_conditions + ['Volume_Up'])

            # Divergences
            if last.get('obv_divergence', 0) == 1:
                signal['obv_divergence'] = 1
                signal['strength'] += 1
                signal['conditions'].append('OBV_Bull_Div')
            elif last.get('obv_divergence', 0) == -1:
                signal['obv_divergence'] = -1
                signal['strength'] += 1
                signal['conditions'].append('OBV_Bear_Div')

            # RSI reversals
            if last.get('rsi_reversal', 0) == 1:
                signal['rsi_reversal'] = 1
                signal['strength'] += 1
                signal['conditions'].append('RSI_Bull_Rev')
            elif last.get('rsi_reversal', 0) == -1:
                signal['rsi_reversal'] = -1
                signal['strength'] += 1
                signal['conditions'].append('RSI_Bear_Rev')

            return signal
        except Exception as e:
            logging.error(f"信号生成失败: {str(e)}")
            return {'strength': 0, 'long': False, 'short': False, 'conditions': []}

    def multi_timeframe_analysis(self) -> dict:
        try:
            # Compute indicators/divergence per TF
            for tf in list(self.data.keys()):
                df = self.data.get(tf)
                if df is None or len(df) == 0:
                    continue
                df = self.calculate_indicators(df, tf)
                df = self.detect_obv_divergence(df)
                if len(df) < 3:
                    logging.warning(f"{self.symbol} {tf} 指标有效样本不足")
                    continue
                self.data[tf] = df

            # Per-TF signals
            signals: dict[str, dict] = {}
            for tf in [self.tertiary_tf, self.secondary_tf, self.primary_tf]:
                if tf in self.data:
                    s = self.generate_signal(self.data[tf])
                    logging.info(f"[{tf}] signal={s}")
                    signals[tf] = s

            combined = {
                'long': False,
                'short': False,
                'confidence': 0,
                'timeframes': [],
                'type': '无信号',
                'details': signals
            }

            long_tfs = [tf for tf in [self.tertiary_tf, self.secondary_tf, self.primary_tf]
                        if tf in signals and signals[tf].get('long')]
            short_tfs = [tf for tf in [self.tertiary_tf, self.secondary_tf, self.primary_tf]
                         if tf in signals and signals[tf].get('short')]

            # Confirmation mechanism (relaxed: >=)
            if len(long_tfs) >= 2:
                if self.last_signal == 'long':
                    self.signal_confirmation_count += 1
                else:
                    self.last_signal = 'long'
                    self.signal_confirmation_count = 1
                if self.signal_confirmation_count >= self.confirmation_required:
                    combined['long'] = True
                    combined['short'] = False
                    combined['type'] = '多头共振'
                    combined['timeframes'] = long_tfs
                    combined['confidence'] = int(50 + 25 * (len(long_tfs) - 1))
                    self.signal_confirmation_count = 0
                    self.last_signal = None
            elif len(short_tfs) >= 2:
                if self.last_signal == 'short':
                    self.signal_confirmation_count += 1
                else:
                    self.last_signal = 'short'
                    self.signal_confirmation_count = 1
                if self.signal_confirmation_count >= self.confirmation_required:
                    combined['short'] = True
                    combined['long'] = False
                    combined['type'] = '空头共振'
                    combined['timeframes'] = short_tfs
                    combined['confidence'] = int(50 + 25 * (len(short_tfs) - 1))
                    self.signal_confirmation_count = 0
                    self.last_signal = None
            else:
                # 单TF强信号后备触发：primary_tf 强度 >= 4
                primary = signals.get(self.primary_tf, {})
                if primary.get('long') and primary.get('strength', 0) >= 4:
                    combined['long'] = True
                    combined['type'] = '单TF强多'
                    combined['timeframes'] = [self.primary_tf]
                    combined['confidence'] = 55
                    self.signal_confirmation_count = 0
                    self.last_signal = None
                elif primary.get('short') and primary.get('strength', 0) >= 4:
                    combined['short'] = True
                    combined['type'] = '单TF强空'
                    combined['timeframes'] = [self.primary_tf]
                    combined['confidence'] = 55
                    self.signal_confirmation_count = 0
                    self.last_signal = None
                else:
                    self.signal_confirmation_count = 0
                    self.last_signal = None

            return combined
        except Exception as e:
            logging.error(f"多时间框架分析失败: {str(e)}")
            logging.error(traceback.format_exc())
            return {'long': False, 'short': False, 'confidence': 0, 'timeframes': [], 'type': '分析错误'}

    # ------------------------- Targets & Risk ------------------------------
    def calculate_price_targets(self, signal: dict) -> dict:
        try:
            if self.primary_tf not in self.data:
                return {'entry': 0, 'stop_loss': 0, 'take_profit': [], 'rr_ratio': 0, 'rr_all': []}

            df = self.data[self.primary_tf]
            last = df.iloc[-1]
            targets = {'entry': float(last['close']), 'stop_loss': None, 'take_profit': [], 'rr_ratio': 0.0, 'rr_all': []}

            if signal.get('long'):
                candidates = [last['lower'], df['low'].iloc[-5:].min(), last['close'] - 2 * last['atr']]
                stop = float(min(candidates))
                tps = [float(last['upper']), float(last['upper'] + last['atr']), float(last['upper'] + 2 * last['atr'])]
                risk = max(1e-8, float(last['close']) - stop)
                rrs = [max(0.0, tp - float(last['close'])) / risk for tp in tps]
            elif signal.get('short'):
                candidates = [last['upper'], df['high'].iloc[-5:].max(), last['close'] + 2 * last['atr']]
                stop = float(max(candidates))
                tps = [float(last['lower']), float(last['lower'] - last['atr']), float(last['lower'] - 2 * last['atr'])]
                risk = max(1e-8, stop - float(last['close']))
                rrs = [max(0.0, float(last['close']) - tp) / risk for tp in tps]
            else:
                return targets

            targets['stop_loss'] = stop
            targets['take_profit'] = tps
            targets['rr_all'] = rrs
            targets['rr_ratio'] = rrs[self.rr_tp_index] if rrs else 0.0
            return targets
        except Exception as e:
            logging.error(f"价格目标计算失败: {str(e)}")
            logging.error(traceback.format_exc())
            return {'entry': 0, 'stop_loss': 0, 'take_profit': [], 'rr_ratio': 0, 'rr_all': []}

    # ---------------------------- Plotting --------------------------------
    def plot_multi_timeframe_chart(self, signal: dict) -> bool:
        try:
            fig, axes = plt.subplots(3, 1, figsize=(16, 18), sharex=False)
            timeframes = [self.tertiary_tf, self.secondary_tf, self.primary_tf]
            color = 'green' if signal.get('long') else ('red' if signal.get('short') else 'gray')

            for i, tf in enumerate(timeframes):
                if tf not in self.data:
                    continue
                df = self.data[tf]
                ax = axes[i]

                # Prepare OHLC for mplfinance
                date_num = mdates.date2num(df.index.to_pydatetime())
                ohlc = np.column_stack([date_num, df[['open','high','low','close']].values])

                candlestick_ohlc(
                    ax,
                    ohlc,
                    width=0.8/len(timeframes),
                    colorup='g',
                    colordown='r'
                )

                ax.plot(df.index, df['upper'],  linestyle='--', label='Upper Band', alpha=0.7)
                ax.plot(df.index, df['middle'],               label='Middle Band', alpha=0.7)
                ax.plot(df.index, df['lower'],  linestyle='--', label='Lower Band', alpha=0.7)
                ax.plot(df.index, df['ema50'],                label='EMA50',       alpha=0.7)
                ax.plot(df.index, df['ema200'],               label='EMA200',      alpha=0.7)

                if tf in signal.get('timeframes', []):
                    ax.plot(df.index[-1], df['close'].iloc[-1], 'o', markersize=10, color=color, label='Signal')

                obv_div_points = df[df['obv_divergence'] != 0].tail(10)
                for idx, row in obv_div_points.iterrows():
                    c = 'green' if row['obv_divergence'] > 0 else 'red'
                    ax.plot(idx, row['close'], 's', markersize=8, color=c, alpha=0.7)

                ax.set_title(f"{self.symbol} - {tf} Timeframe")
                ax.legend(loc='best')
                ax.grid(True)
                ax.xaxis_date()
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax.get_xticklabels(), rotation=45)

            plt.tight_layout()
            fig.suptitle(
                f"{signal.get('type','无信号')} Signal - Confidence: {signal.get('confidence',0)}%",
                fontsize=16, fontweight='bold', color=color
            )

            out_path = f"{self.symbol.replace('/', '_')}_multi_timeframe.png"
            plt.savefig(out_path, bbox_inches='tight')
            plt.close()
            logging.info(f"图表已保存: {out_path}")
            return True
        except Exception as e:
            logging.error(f"图表绘制失败: {str(e)}")
            logging.error(traceback.format_exc())
            return False

 
    # -------------------------- Execution ---------------------------------
    def execute_trade(self, signal: dict, targets: dict) -> None:
        try:
            logging.info("\n执行交易指令...")
            side = '买入' if signal.get('long') else ('卖出' if signal.get('short') else None)
            if not side:
                return
            logging.info(f"👉 {side} {self.symbol} @ {targets['entry']:.4f}")
            logging.info(f"⛔ 止损: {targets['stop_loss']:.4f}")
            for i, tp in enumerate(targets['take_profit']):
                logging.info(f"🎯 目标{i+1}: {tp:.4f}  (RR≈{targets['rr_all'][i]:.2f}:1)")
        except Exception as e:
            logging.error(f"交易执行失败: {str(e)}")

    def run_strategy(self, live_trading: bool = False, *, run_once: bool = False) -> None:
        logging.info("启动多时间框架交易策略")
        while True:
            try:
                start_time = time.time()

                self.load_data()
                signal = self.multi_timeframe_analysis()
                print(signal)
                if signal.get('long') or signal.get('short'):
                    targets = self.calculate_price_targets(signal)

                    logging.info("\n" + "=" * 50)
                    logging.info(f"发现 {signal['type']} 信号!")
                    logging.info(f"时间框架共振: {', '.join(signal['timeframes']) if signal['timeframes'] else '无'}")
                    logging.info(f"置信度: {signal['confidence']}%")
                    logging.info(f"当前价格: {targets['entry']:.4f}")
                    logging.info(f"止损位: {targets['stop_loss']:.4f}")
                    logging.info("止盈位:")
                    for i, tp in enumerate(targets['take_profit']):
                        logging.info(f"  TP{i + 1}: {tp:.4f}  (RR≈{targets['rr_all'][i]:.2f}:1)")
                    logging.info(f"主展示 RR (TP{self.rr_tp_index+1}): {targets['rr_ratio']:.2f}:1")

                    if self.plot_multi_timeframe_chart(signal):
                        logging.info("图表已保存!")

                    if live_trading:
                        self.execute_trade(signal, targets)

                processing_time = time.time() - start_time
                sleep_minutes = 5
                next_run_utc = datetime.utcnow() + timedelta(minutes=sleep_minutes)
                adjusted_sleep = max(1, sleep_minutes * 60 - processing_time)

                logging.info(f"处理时间: {processing_time:.2f}秒")
                logging.info(f"下一次扫描(UTC): {next_run_utc} (等待 {adjusted_sleep:.2f}秒)")

                if run_once:
                    break
                time.sleep(adjusted_sleep)

            except KeyboardInterrupt:
                logging.info("用户中断策略执行")
                break
            except Exception as e:
                logging.error(f"策略运行错误: {str(e)}")
                logging.error(traceback.format_exc())
                time.sleep(300)  # backoff 5 minutes


if __name__ == '__main__':
    # 简单自测：只跑一次，便于在日志中观察信号
    strat = EnhancedMultiTimeframeStrategy(
        symbol='BTC/USDT',
        primary_tf='1h',
        secondary_tf='4h',
        tertiary_tf='1d',
        end_time=datetime.now(),
        table_with_tf=True,
        confirmation_required=0,  # 测试阶段建议 0
        rr_tp_index=1,
    )
    strat.run_strategy(run_once=True)
