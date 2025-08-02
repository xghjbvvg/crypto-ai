# -*- coding: utf-8 -*-
"""
Enhanced Multi-Timeframe Strategy with Unified Global Lock
- TDengine + TA-Lib + matplotlib
- Multi-timeframe signals + BEST ENTRY (Setup vs Trigger) with anti-churn
- Unified GLOBAL LOCK across all signal types (resonance/singleTF/best) to avoid consecutive signals
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

# Headless backend for server
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates

# Local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.td_dao import query_df_from_tdengine

# ------------------------------ Logging ----------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('trading_strategy.log'), logging.StreamHandler()]
)


class EnhancedMultiTimeframeStrategy:
    def __init__(
        self,
        symbol: str = 'BTC/USDT',
        primary_tf: str = '1h',
        secondary_tf: str = '4h',
        tertiary_tf: str = '1d',
        end_time = datetime.utcnow(),
        *,
        table_with_tf: bool = True,
        confirmation_required: int = 1,
        rr_tp_index: int = 1,

        # ====== De-churn & Quality knobs ======
        require_two_tf: bool = False,             # 至少两TF共振才触发（禁用单TF兜底）
        min_rr: float = 2.5,                      # 最低RR门槛
        cooldown_bars: int = 16,                  # 同侧冷却条数（主TF）
        lock_after_trade_bars: int = 24,          # 【全局锁定】条数（主TF）
        breakout_buffer_atr: float = 0.30,        # 突破需超过摆动位 ±0.30*ATR
        min_distance_from_prev_entry_atr: float = 0.60,  # 新入场需与上次价相距 ≥0.6*ATR
        max_candle_atr: float = 1.2,              # 大实体K过滤
        session_filter: bool = True,              # 低流动时段过滤
        reentry_min_bars: int = 8,                # 再入最少间隔条数
        reentry_reset_k: float = 0.25,            # 需回撤至中轨/EMA50 ±k*ATR 才允许再入
        require_new_pivot: bool = True,           # 再入需出现新摆动点
    ):
        self.symbol = symbol
        self.primary_tf = primary_tf
        self.secondary_tf = secondary_tf
        self.tertiary_tf = tertiary_tf
        self.table_with_tf = table_with_tf
        self.confirmation_required = max(0, int(confirmation_required))
        self.rr_tp_index = max(0, min(2, int(rr_tp_index)))

        # 去噪参数
        self.require_two_tf = bool(require_two_tf)
        self.min_rr = float(min_rr)
        self.cooldown_bars = int(cooldown_bars)
        self.lock_after_trade_bars = int(lock_after_trade_bars)
        self.breakout_buffer_atr = float(breakout_buffer_atr)
        self.min_distance_from_prev_entry_atr = float(min_distance_from_prev_entry_atr)
        self.max_candle_atr = float(max_candle_atr)
        self.session_filter = bool(session_filter)
        self.reentry_min_bars = int(reentry_min_bars)
        self.reentry_reset_k = float(reentry_reset_k)
        self.require_new_pivot = bool(require_new_pivot)

        self.last_signal = None
        self.signal_confirmation_count = 0
        self.end_time = end_time or datetime.utcnow()

        # BEST 状态 & 全局锁定
        self._last_entry_side = None
        self._last_entry_bar_time = None
        self._last_stats_bar_time = None
        self._last_break_sw_high = None
        self._last_break_sw_low = None

        # 统一锁定的数据：按主TF“条数” + 时间双重判定
        self._lock_until_time = None
        self._lock_until_bar_index = None  # 主TF索引（整数位置）

        # Exchange（可选）
        self.exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})

        # TF缩放（相对1h）
        self.tf_conversion = {'1d': 24, '4h': 6, '1h': 1, '30m': 0.5, '15m': 0.25, '5m': 0.083}

        self.data: dict[str, pd.DataFrame] = {}
        self.load_data()

    # -------------------------- Utils -------------------------------------
    def _table_name(self, tf: str) -> str:
        base = self.symbol.replace('/', '_').lower()
        return f"{base}_kline"  # 如需分表可改：f"{base}_{tf}_kline"

    def _tf_to_seconds(self, tf: str) -> int:
        unit = ''.join([c for c in tf if c.isalpha()])
        num = int(''.join([c for c in tf if c.isdigit()]) or 1)
        mult = {'m':60, 'h':3600, 'd':86400}.get(unit, 3600)
        return num * mult

    def _is_time_ok(self, ts_utc: pd.Timestamp) -> bool:
        if not self.session_filter:
            return True
        return ts_utc.hour not in (23, 0, 1)

    # ---------- GLOBAL LOCK（统一锁定） ----------
    def _current_primary_bar_info(self):
        """返回主TF最后一根的 (time, pos)；若无则(None, None)"""
        df = self.data.get(self.primary_tf)
        if df is None or df.empty:
            return None, None
        ts = df.index[-1]
        try:
            pos = df.index.get_loc(ts)
            if isinstance(pos, slice):
                # 极少数情况，退化到长度-1
                pos = len(df) - 1
        except Exception:
            pos = None
        return ts, int(pos) if pos is not None else None

    def _enter_global_lock(self, from_bar_time: pd.Timestamp):
        """进入全局锁定：以主TF条数为单位 + 时间兜底。"""
        df = self.data.get(self.primary_tf)
        if df is None or df.empty:
            # 仅按时间兜底
            step_sec = self._tf_to_seconds(self.primary_tf)
            self._lock_until_time = from_bar_time + pd.Timedelta(seconds=step_sec * self.lock_after_trade_bars)
            self._lock_until_bar_index = None
            logging.info(f"[LOCK] 仅时间锁定至 {self._lock_until_time}")
            return

        # 计算目标结束“条”位置
        try:
            start_idx = df.index.get_loc(from_bar_time)
            if isinstance(start_idx, slice):
                start_idx = len(df) - 1
        except Exception:
            start_idx = len(df) - 1

        lock_end_idx = start_idx + self.lock_after_trade_bars
        self._lock_until_bar_index = lock_end_idx

        # 时间兜底（以免后续数据重载、缺K等造成偏差）
        step_sec = self._tf_to_seconds(self.primary_tf)
        self._lock_until_time = from_bar_time + pd.Timedelta(seconds=step_sec * self.lock_after_trade_bars)

        logging.info(f"[LOCK] 条数锁定至 idx>={self._lock_until_bar_index}；时间锁定至 {self._lock_until_time}")

    def _is_locked(self) -> bool:
        """主TF条数优先；时间为兜底。"""
        ts, pos = self._current_primary_bar_info()
        # 条数判断
        if self._lock_until_bar_index is not None and pos is not None:
            if pos <= self._lock_until_bar_index:
                return True
        # 时间兜底
        if self._lock_until_time is not None and ts is not None:
            if ts <= self._lock_until_time:
                return True
        return False

    # -------------------------- Data Loading ------------------------------
    def load_data(self) -> None:
        self.data = {}
        for tf in [self.tertiary_tf, self.secondary_tf, self.primary_tf]:
            try:
                table = self._table_name(tf)
                logging.info(f"读取 {self.symbol} {tf} => 表: {table}")
                df = query_df_from_tdengine(table, tf, self.end_time)
                df = pd.DataFrame(df)
                if df is None or len(df) == 0:
                    logging.warning(f"{table} 返回空数据")
                    continue

                # 标准化
                df.columns = [str(c).strip().lower() for c in df.columns]
                alias_map = {'ts':'timestamp', 'time':'timestamp', 'vol':'volume'}
                df.rename(columns={k:v for k,v in alias_map.items() if k in df.columns}, inplace=True)

                if 'timestamp' in df.columns:
                    ts0 = pd.to_numeric(df['timestamp'].iloc[0], errors='coerce')
                    if pd.notna(ts0) and np.isfinite(ts0):
                        unit = 'ms' if int(ts0) > 10**11 else 's'
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit=unit, utc=True)
                    else:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
                    df = df.set_index('timestamp')
                else:
                    if not isinstance(df.index, pd.DatetimeIndex):
                        raise ValueError(f"{table} 需包含 datetime 索引或 timestamp 列")

                df = df.sort_index()

                # 数值列
                for col in ['open','high','low','close','volume']:
                    if col not in df.columns:
                        raise ValueError(f"{table} 缺少列: {col}")
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

                before = len(df)
                df = df[~df.index.duplicated(keep='last')]
                df = df.dropna(subset=['open','high','low','close','volume'])
                df = df[df['volume'] > 0]
                logging.info(f"[DEBUG] 清洗 {tf}: {before} -> {len(df)}")

                if not df.empty:
                    self.data[tf] = df
            except Exception as e:
                logging.error(f"加载数据失败: {tf}, 错误: {e}")
                logging.error(traceback.format_exc())

    # ------------------------ Indicator Computation -----------------------
    def _p(self, x: float) -> int:
        return max(1, int(round(x)))

    def calculate_indicators(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        try:
            if df is None or len(df) == 0:
                return df
            for col in ['open','high','low','close','volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

            mult = self.tf_conversion.get(tf, 1)
            base = {
                'boll': self._p(20 * mult),
                'macd_fast': self._p(12 * mult),
                'macd_slow': self._p(26 * mult),
                'macd_sig':  self._p(9  * mult),
                'rsi':       self._p(14 * mult),
                'vol_ma':    self._p(20 * mult),
                'ema50':     self._p(50 * mult),
                'ema200':    self._p(200 * mult),
                'atr':       self._p(14 * mult),
            }

            warm = max(base['ema200'], base['boll'], base['atr'], base['macd_slow'] + base['macd_sig'])
            n = len(df)
            if n <= warm:
                scale = max(10, int(n // 3))
                for k in base:
                    base[k] = max(10, min(base[k], scale))
                logging.warning(f"[{tf}] 样本{n}<warmup{warm}，自适应周期: {base}")
                warm = max(base.values())

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

            valid = out.iloc[max(0, warm-1):]
            valid = valid.dropna(subset=['upper','middle','lower','ema50','ema200','macd','signal','hist','rsi','vol_ma','atr'])
            return valid if len(valid) else out
        except Exception as e:
            logging.error(f"指标计算失败: {tf}, 错误: {e}")
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
            logging.error(f"OBV背离检测失败: {e}")
            return df

    # ---------------------------- Signals ---------------------------------
    def _body_size(self, row) -> float:
        return abs(row['close'] - row['open'])

    def generate_signal(self, df: pd.DataFrame) -> dict:
        try:
            if df is None or len(df) < 3:
                return {'strength': 0, 'long': False, 'short': False, 'conditions': []}

            last, prev = df.iloc[-1], df.iloc[-2]
            signal = {'strength': 0, 'long': False, 'short': False, 'conditions': [], 'obv_divergence': 0, 'rsi_reversal': 0}
            vol_ok = last['volume'] > last['vol_ma']

            # 多头
            bull = []
            if last['close'] > last['middle'] and last['middle'] > prev['middle']: bull.append('Boll_Up')
            if last['macd'] > last['signal'] and last['hist'] > 0:                bull.append('MACD_Up')
            if last['rsi'] > 52:                                                  bull.append('RSI_Ok')
            if last['trend'] == 'Up' and last['close'] > last['ema50']:           bull.append('Trend_Up')
            if len(bull) >= 3:
                signal['long'] = True
                signal['strength'] += len(bull) + (1 if vol_ok else 0)
                signal['conditions'].extend(bull + (['Volume_Up'] if vol_ok else []))

            # 空头
            bear = []
            if last['close'] < last['middle'] and last['middle'] < prev['middle']: bear.append('Boll_Down')
            if last['macd'] < last['signal'] and last['hist'] < 0:                 bear.append('MACD_Down')
            if last['rsi'] < 48:                                                   bear.append('RSI_Ok')
            if last['trend'] == 'Down' and last['close'] < last['ema50']:          bear.append('Trend_Down')
            if len(bear) >= 3 and vol_ok:
                signal['short'] = True
                signal['strength'] += len(bear) + 1
                signal['conditions'].extend(bear + ['Volume_Up'])

            # 背离/反转
            if last.get('obv_divergence', 0) == 1:  signal.update(obv_divergence=1);  signal['strength'] += 1; signal['conditions'].append('OBV_Bull_Div')
            if last.get('obv_divergence', 0) == -1: signal.update(obv_divergence=-1); signal['strength'] += 1; signal['conditions'].append('OBV_Bear_Div')
            if last.get('rsi_reversal', 0) == 1:    signal.update(rsi_reversal=1);    signal['strength'] += 1; signal['conditions'].append('RSI_Bull_Rev')
            if last.get('rsi_reversal', 0) == -1:   signal.update(rsi_reversal=-1);   signal['strength'] += 1; signal['conditions'].append('RSI_Bear_Rev')

            return signal
        except Exception as e:
            logging.error(f"信号生成失败: {e}")
            return {'strength': 0, 'long': False, 'short': False, 'conditions': []}

    def multi_timeframe_analysis(self) -> dict:
        try:
            # 计算指标
            for tf in list(self.data.keys()):
                df = self.data.get(tf)
                if df is None or df.empty:
                    continue
                df = self.calculate_indicators(df, tf)
                df = self.detect_obv_divergence(df)
                if len(df) < 3:
                    logging.warning(f"{self.symbol} {tf} 指标有效样本不足")
                    continue
                self.data[tf] = df

            # 统一锁定：直接返回“锁定中”
            if self._is_locked():
                return {'long': False, 'short': False, 'confidence': 0, 'timeframes': [], 'type': '锁定中', 'details': {}}

            # TF信号
            signals = {}
            for tf in [self.tertiary_tf, self.secondary_tf, self.primary_tf]:
                if tf in self.data:
                    signals[tf] = self.generate_signal(self.data[tf])

            combined = {'long': False, 'short': False, 'confidence': 0, 'timeframes': [], 'type': '无信号', 'details': signals}

            long_tfs = [tf for tf in [self.tertiary_tf, self.secondary_tf, self.primary_tf] if tf in signals and signals[tf].get('long')]
            short_tfs = [tf for tf in [self.tertiary_tf, self.secondary_tf, self.primary_tf] if tf in signals and signals[tf].get('short')]

            # 至少两TF共振
            if len(long_tfs) >= 2:
                if self.last_signal == 'long': self.signal_confirmation_count += 1
                else: self.last_signal, self.signal_confirmation_count = 'long', 1
                if self.signal_confirmation_count >= self.confirmation_required:
                    combined.update(long=True, short=False, type='多头共振', timeframes=long_tfs,
                                    confidence=int(50 + 25 * (len(long_tfs)-1)))
                    self.signal_confirmation_count = 0; self.last_signal = None
            elif len(short_tfs) >= 2:
                if self.last_signal == 'short': self.signal_confirmation_count += 1
                else: self.last_signal, self.signal_confirmation_count = 'short', 1
                if self.signal_confirmation_count >= self.confirmation_required:
                    combined.update(short=True, long=False, type='空头共振', timeframes=short_tfs,
                                    confidence=int(50 + 25 * (len(short_tfs)-1)))
                    self.signal_confirmation_count = 0; self.last_signal = None
            else:
                if not self.require_two_tf:
                    primary = signals.get(self.primary_tf, {})
                    if primary.get('long') and primary.get('strength', 0) >= 4:
                        combined.update(long=True, type='单TF强多', timeframes=[self.primary_tf], confidence=55)
                    elif primary.get('short') and primary.get('strength', 0) >= 4:
                        combined.update(short=True, type='单TF强空', timeframes=[self.primary_tf], confidence=55)
                    else:
                        self.signal_confirmation_count = 0; self.last_signal = None
                else:
                    self.signal_confirmation_count = 0; self.last_signal = None

            return combined
        except Exception as e:
            logging.error(f"多时间框架分析失败: {e}")
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
                stop = float(min(last['lower'], df['low'].iloc[-5:].min(), last['close'] - 2*last['atr']))
                tps  = [float(last['upper']), float(last['upper'] + last['atr']), float(last['upper'] + 2*last['atr'])]
                risk = max(1e-8, float(last['close']) - stop)
                rrs  = [max(0.0, tp - float(last['close'])) / risk for tp in tps]
            elif signal.get('short'):
                stop = float(max(last['upper'], df['high'].iloc[-5:].max(), last['close'] + 2*last['atr']))
                tps  = [float(last['lower']), float(last['lower'] - last['atr']), float(last['lower'] - 2*last['atr'])]
                risk = max(1e-8, stop - float(last['close']))
                rrs  = [max(0.0, float(last['close']) - tp) / risk for tp in tps]
            else:
                return targets

            targets.update(stop_loss=stop, take_profit=tps, rr_all=rrs, rr_ratio=(rrs[self.rr_tp_index] if rrs else 0.0))
            return targets
        except Exception as e:
            logging.error(f"价格目标计算失败: {e}")
            logging.error(traceback.format_exc())
            return {'entry': 0, 'stop_loss': 0, 'take_profit': [], 'rr_ratio': 0, 'rr_all': []}

    # ------------------------- Pivots & Helpers ---------------------------
    def _pivot_high_low(self, series: pd.Series, left=2, right=2):
        n = len(series)
        ph = pd.Series(False, index=series.index)
        pl = pd.Series(False, index=series.index)
        vals = series.values
        for i in range(left, n - right):
            win = vals[i-left:i+right+1]
            if vals[i] == np.max(win): ph.iloc[i] = True
            if vals[i] == np.min(win): pl.iloc[i] = True
        return ph, pl

    def _recent_swing(self, df: pd.DataFrame):
        phh, pll = self._pivot_high_low(df['high'])
        last_sw_high = df['high'][phh].tail(3).max() if phh.any() else df['high'].iloc[-5]
        last_sw_low  = df['low'][pll].tail(3).min()  if pll.any() else df['low'].iloc[-5]
        return float(last_sw_high), float(last_sw_low)

    def _has_bull_bias(self, signals: dict) -> bool:
        vote = 0
        for tf in [self.tertiary_tf, self.secondary_tf]:
            s = signals.get(tf, {})
            vote += 1 if s.get('long') else (-1 if s.get('short') else 0)
        return vote >= 1

    def _has_bear_bias(self, signals: dict) -> bool:
        vote = 0
        for tf in [self.tertiary_tf, self.secondary_tf]:
            s = signals.get(tf, {})
            vote += -1 if s.get('short') else (1 if s.get('long') else 0)
        return vote <= -1

    def _is_mean_reversion_zone(self, row) -> bool:
        in_boll_mid = abs(row['close'] - row['middle']) <= 0.5 * row['atr']
        near_ema50  = abs(row['close'] - row['ema50'])  <= 0.5 * row['atr']
        return in_boll_mid or near_ema50

    def _macd_hist_rising(self, df: pd.DataFrame) -> bool:
        if len(df) < 3: return False
        h = df['hist'].iloc[-3:]
        return (h.iloc[-1] > h.iloc[-2] > h.iloc[-3])

    def _macd_hist_falling(self, df: pd.DataFrame) -> bool:
        if len(df) < 3: return False
        h = df['hist'].iloc[-3:]
        return (h.iloc[-1] < h.iloc[-2] < h.iloc[-3])

    def _cooldown_ok(self, side: str, current_bar_time) -> bool:
        # 方向性的冷却（锁定是全局的；冷却保留方向性）
        if self._last_entry_side != side or self._last_entry_bar_time is None:
            return True
        df = self.data[self.primary_tf]
        try:
            idx_now = df.index.get_loc(current_bar_time)
            idx_last = df.index.get_loc(self._last_entry_bar_time)
            if isinstance(idx_now, (int, np.integer)) and isinstance(idx_last, (int, np.integer)):
                bars_since = idx_now - idx_last
            else:
                bars_since = 0
        except Exception:
            return True
        return bars_since >= max(self.cooldown_bars, self.reentry_min_bars)

    def _reset_done_since_last_entry(self, side: str, df: pd.DataFrame, last_idx, *, k: float = None) -> bool:
        if k is None: k = self.reentry_reset_k
        if self._last_entry_bar_time is None or self._last_entry_side != side:
            return True
        try:
            start = df.index.get_loc(self._last_entry_bar_time)
            end = df.index.get_loc(last_idx)
        except Exception:
            return True
        if isinstance(start, slice) or isinstance(end, slice): return True
        segment = df.iloc[start+1:end+1]
        if segment.empty: return False
        if side == 'long':
            reset_mask = (segment['close'] <= segment['middle'] - k*segment['atr']) | (segment['close'] <= segment['ema50'] - k*segment['atr'])
        else:
            reset_mask = (segment['close'] >= segment['middle'] + k*segment['atr']) | (segment['close'] >= segment['ema50'] + k*segment['atr'])
        if not bool(reset_mask.any()):
            return False
        if not self.require_new_pivot:
            return True
        phh, pll = self._pivot_high_low(df['high'])
        return pll.iloc[start+1:end+1].any() if side == 'long' else phh.iloc[start+1:end+1].any()

    # ------------------------- BEST ENTRY (Unified Lock) ------------------
    def best_entry_trigger(self, signals: dict) -> dict:
        # 全局锁定：直接禁止
        if self._is_locked():
            return {'side': None}

        if self.primary_tf not in self.data:
            return {'side': None}

        df = self.data[self.primary_tf]
        last = df.iloc[-1]
        last_idx = df.index[-1]

        if not self._is_time_ok(last_idx):
            return {'side': None}
        if self._body_size(last) > self.max_candle_atr * last['atr']:
            return {'side': None}

        # 摆动位、方向偏好、突破缓冲
        sw_high, sw_low = self._recent_swing(df)
        bull_ok = self._has_bull_bias(signals) and self._is_mean_reversion_zone(last)
        bear_ok = self._has_bear_bias(signals) and self._is_mean_reversion_zone(last)
        long_break  = last['close'] > (sw_high + self.breakout_buffer_atr * last['atr'])
        short_break = last['close'] < (sw_low  - self.breakout_buffer_atr * last['atr'])

        # MACD 复位（避免连打）
        def _macd_reset_ok(_df, side):
            if len(_df) < 5: return False
            h = _df['hist'].iloc[-5:]
            near_zero = h.abs().min() <= _df['atr'].iloc[-1] * 0.05
            trend_ok  = (h.iloc[-1] > h.iloc[-2] > h.iloc[-3]) if side=='long' else (h.iloc[-1] < h.iloc[-2] < h.iloc[-3])
            return near_zero or trend_ok

        # 与上次入场的最小价格距离
        def _min_dist_ok(side):
            if self._last_entry_bar_time is None or self._last_entry_side != side:
                return True
            prev_price = float(df.loc[self._last_entry_bar_time]['close'])
            return abs(float(last['close']) - prev_price) >= self.min_distance_from_prev_entry_atr * float(last['atr'])

        # LONG
        if bull_ok and long_break and (last['volume'] > last['vol_ma']) and self._macd_hist_rising(df):
            if _macd_reset_ok(df, 'long') and _min_dist_ok('long') and self._cooldown_ok('long', last_idx) and self._reset_done_since_last_entry('long', df, last_idx):
                rr = self.calculate_price_targets({'long': True, 'short': False})
                if rr['rr_ratio'] >= self.min_rr:
                    return {'side':'long','entry':float(last['close']),'stop':float(rr['stop_loss']),
                            'tp': float(rr['take_profit'][self.rr_tp_index]), 'rr': float(rr['rr_ratio']),
                            'bar_time': last_idx}

        # SHORT
        if bear_ok and short_break and (last['volume'] > last['vol_ma']) and self._macd_hist_falling(df):
            if _macd_reset_ok(df, 'short') and _min_dist_ok('short') and self._cooldown_ok('short', last_idx) and self._reset_done_since_last_entry('short', df, last_idx):
                rr = self.calculate_price_targets({'long': False, 'short': True})
                if rr['rr_ratio'] >= self.min_rr:
                    return {'side':'short','entry':float(last['close']),'stop':float(rr['stop_loss']),
                            'tp': float(rr['take_profit'][self.rr_tp_index]), 'rr': float(rr['rr_ratio']),
                            'bar_time': last_idx}

        return {'side': None}

    # ---------------------------- Plotting --------------------------------
    def plot_multi_timeframe_chart(self, signal: dict) -> bool:
        try:
            fig, axes = plt.subplots(3, 1, figsize=(16, 18), sharex=False)
            tfs = [self.tertiary_tf, self.secondary_tf, self.primary_tf]
            color = 'green' if signal.get('long') else ('red' if signal.get('short') else 'gray')

            for i, tf in enumerate(tfs):
                if tf not in self.data: continue
                df = self.data[tf]; ax = axes[i]
                date_num = mdates.date2num(df.index.to_pydatetime())
                ohlc = np.column_stack([date_num, df[['open','high','low','close']].values])
                candlestick_ohlc(ax, ohlc, width=0.8/len(tfs), colorup='g', colordown='r')

                for col, ls in [('upper','--'), ('middle','-'), ('lower','--')]:
                    ax.plot(df.index, df[col], linestyle=ls, label=col.title(), alpha=0.7)
                ax.plot(df.index, df['ema50'], label='EMA50', alpha=0.7)
                ax.plot(df.index, df['ema200'], label='EMA200', alpha=0.7)

                if tf in signal.get('timeframes', []):
                    ax.plot(df.index[-1], df['close'].iloc[-1], 'o', ms=10, color=color, label='Signal')

                obv_div = df[df['obv_divergence'] != 0].tail(10)
                for idx, row in obv_div.iterrows():
                    c = 'green' if row['obv_divergence'] > 0 else 'red'
                    ax.plot(idx, row['close'], 's', ms=8, color=c, alpha=0.7)

                ax.set_title(f"{self.symbol} - {tf}")
                ax.legend(loc='best'); ax.grid(True)
                ax.xaxis_date(); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax.get_xticklabels(), rotation=45)

            plt.tight_layout()
            fig.suptitle(f"{signal.get('type','无信号')} - Confidence: {signal.get('confidence',0)}%", fontsize=16, fontweight='bold', color=color)
            out_path = f"{self.symbol.replace('/', '_')}_multi_timeframe.png"
            plt.savefig(out_path, bbox_inches='tight'); plt.close()
            logging.info(f"图表已保存: {out_path}")
            return True
        except Exception as e:
            logging.error(f"图表绘制失败: {e}"); logging.error(traceback.format_exc()); return False

    # -------------------------- Execution ---------------------------------
    def execute_trade(self, signal: dict, targets: dict) -> None:
        try:
            side = '买入' if signal.get('long') else ('卖出' if signal.get('short') else None)
            if not side: return
            logging.info(f"👉 {side} {self.symbol} @ {targets['entry']:.4f}")
            logging.info(f"⛔ 止损: {targets['stop_loss']:.4f}")
            for i, tp in enumerate(targets['take_profit']):
                logging.info(f"🎯 目标{i+1}: {tp:.4f}  (RR≈{targets['rr_all'][i]:.2f}:1)")
        except Exception as e:
            logging.error(f"交易执行失败: {e}")

    # ----------------------------- Runner ---------------------------------
    def run_strategy(self, live_trading: bool = False, *, run_once: bool = False) -> None:
        logging.info("启动多时间框架交易策略（统一锁定机制）")
        while True:
            try:
                start_time = time.time()

                self.load_data()
                mkt = self.multi_timeframe_analysis()

                # 统一锁定：直接跳过（日志提示）
                if mkt.get('type') == '锁定中':
                    logging.info("【锁定中】跳过本轮。")
                else:
                    # BEST ENTRY
                    best = self.best_entry_trigger(mkt.get('details', {}))
                    if best.get('side'):
                        # 避免同一根重复输出
                        if self._last_stats_bar_time != best['bar_time']:
                            self._last_stats_bar_time = best['bar_time']

                            final_sig = {
                                'long': best['side']=='long',
                                'short': best['side']=='short',
                                'type': '最佳入场',
                                'timeframes': [self.primary_tf],
                                'confidence': 65 if best['rr'] >= (self.min_rr + 0.5) else 55,
                                'details': mkt.get('details', {})
                            }
                            targets = self.calculate_price_targets(final_sig)
                            # 与BEST快照对齐
                            targets['entry'] = best['entry']
                            targets['stop_loss'] = best['stop']
                            if targets['take_profit']:
                                targets['take_profit'][self.rr_tp_index] = best['tp']
                            targets['rr_ratio'] = best['rr']

                            logging.info("="*64)
                            logging.info(f"【最佳入场】{best['side'].upper()} @ {best['entry']:.2f} | SL={best['stop']:.2f} | TP={best['tp']:.2f} | RR≈{best['rr']:.2f}:1")
                            logging.info("="*64)

                            # 标记最近入场
                            self._last_entry_side = best['side']
                            self._last_entry_bar_time = best['bar_time']

                            # **统一锁定**：从本根开始，锁定 lock_after_trade_bars 条
                            self._enter_global_lock(best['bar_time'])

                            if self.plot_multi_timeframe_chart(final_sig):
                                logging.info("图表已保存!")
                            if live_trading:
                                self.execute_trade(final_sig, targets)
                        else:
                            logging.info("跳过重复统计输出（同一K线）")
                    else:
                        logging.info("无【最佳入场】触发；等待下一根K线收盘。")

                # 调度
                processing_time = time.time() - start_time
                sleep_minutes = 5
                next_run_utc = datetime.utcnow() + timedelta(minutes=sleep_minutes)
                adjusted_sleep = max(1, sleep_minutes * 60 - processing_time)

                logging.info(f"处理时间: {processing_time:.2f}秒")
                logging.info(f"下一次扫描(UTC): {next_run_utc} (等待 {adjusted_sleep:.2f}秒)")

                if run_once: break
                time.sleep(adjusted_sleep)

            except KeyboardInterrupt:
                logging.info("用户中断策略执行"); break
            except Exception as e:
                logging.error(f"策略运行错误: {e}")
                logging.error(traceback.format_exc())
                time.sleep(300)  # backoff


if __name__ == '__main__':
    # 示例：仅跑一轮，使用统一锁定的参数
    strat = EnhancedMultiTimeframeStrategy(
        symbol='BTC/USDT',
        primary_tf='1h',
        secondary_tf='4h',
        tertiary_tf='1d',
        end_time=datetime.utcnow(),
        table_with_tf=True,
        confirmation_required=1,
        rr_tp_index=1,

        # 统一锁定 & 去噪参数（可调）
        require_two_tf=True,
        min_rr=2.5,
        cooldown_bars=16,
        lock_after_trade_bars=24,
        breakout_buffer_atr=0.30,
        min_distance_from_prev_entry_atr=0.60,
        max_candle_atr=1.2,
        session_filter=True,
        reentry_min_bars=8,
        reentry_reset_k=0.25,
        require_new_pivot=True,
    )
    strat.run_strategy(run_once=True)
