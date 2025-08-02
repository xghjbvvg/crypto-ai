# -*- coding: utf-8 -*-
"""
趋势反转策略 - 顶部做空，底部做多
基于15分钟时间框架
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import talib
import ccxt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.td_dao import query_df_from_tdengine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TrendReversalStrategy:
    def __init__(self, symbol='BTC/USDT', timeframe='15m', end_time=datetime.utcnow()):
        self.symbol = symbol
        self.timeframe = timeframe
        self.end_time = end_time
        self.data = None
        self.load_data()
    
    def _table_name(self):
        return f"{self.symbol.replace('/', '_').lower()}_kline"
    
    def load_data(self):
        try:
            table = self._table_name()
            df = query_df_from_tdengine(table, self.timeframe, self.end_time)
            df = pd.DataFrame(df)
            
            if df is None or len(df) == 0:
                self.data = None
                return
            
            # 标准化
            df.columns = [str(c).strip().lower() for c in df.columns]
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                df = df.set_index('timestamp')
            
            df = df.sort_index()
            for col in ['open','high','low','close','volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            self.data = df
            self.calculate_indicators()
            
        except Exception as e:
            logging.error(f"数据加载失败: {e}")
            self.data = None
    
    def calculate_indicators(self):
        if self.data is None or len(self.data) < 50:
            return
        
        df = self.data.copy()
        
        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        
        # 布林带
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
        
        # ATR
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'])
        
        # 均线
        df['ema20'] = talib.EMA(df['close'], timeperiod=20)
        df['ema50'] = talib.EMA(df['close'], timeperiod=50)
        
        # 成交量
        df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # 价格变化
        df['price_change_pct'] = df['close'].pct_change() * 100
        
        self.data = df
    
    def detect_reversal_signals(self):
        if self.data is None or len(self.data) < 10:
            return {'signal': 'no_data', 'long': False, 'short': False}
        
        df = self.data
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 底部反转信号（做多）
        bottom_conditions = [
            last['rsi'] < 30,  # RSI超卖
            last['close'] < last['bb_lower'],  # 价格低于布林带下轨
            last['volume_ratio'] > 1.5,  # 成交量放大
            last['price_change_pct'] > 1.0,  # 价格反弹
            last['macd_hist'] > prev['macd_hist'],  # MACD柱状图上升
        ]
        
        # 顶部反转信号（做空）
        top_conditions = [
            last['rsi'] > 70,  # RSI超买
            last['close'] > last['bb_upper'],  # 价格高于布林带上轨
            last['volume_ratio'] > 1.5,  # 成交量放大
            last['price_change_pct'] < -1.0,  # 价格回落
            last['macd_hist'] < prev['macd_hist'],  # MACD柱状图下降
        ]
        
        if sum(bottom_conditions) >= 3:
            return {
                'signal': 'bottom_reversal',
                'long': True,
                'short': False,
                'confidence': 70,
                'entry_price': float(last['close']),
                'stop_loss': float(last['close'] - 2 * last['atr']),
                'take_profit': float(last['close'] + 4 * last['atr']),
                'details': {
                    'rsi': float(last['rsi']),
                    'volume_ratio': float(last['volume_ratio']),
                    'price_change_pct': float(last['price_change_pct'])
                }
            }
        
        elif sum(top_conditions) >= 3:
            return {
                'signal': 'top_reversal',
                'long': False,
                'short': True,
                'confidence': 70,
                'entry_price': float(last['close']),
                'stop_loss': float(last['close'] + 2 * last['atr']),
                'take_profit': float(last['close'] - 4 * last['atr']),
                'details': {
                    'rsi': float(last['rsi']),
                    'volume_ratio': float(last['volume_ratio']),
                    'price_change_pct': float(last['price_change_pct'])
                }
            }
        
        return {
            'signal': 'no_signal',
            'long': False,
            'short': False,
            'confidence': 0,
            'entry_price': float(last['close']),
            'stop_loss': 0,
            'take_profit': 0,
            'details': {
                'rsi': float(last['rsi']),
                'volume_ratio': float(last['volume_ratio']),
                'price_change_pct': float(last['price_change_pct'])
            }
        }
    
    def run_strategy(self, run_once=False):
        logging.info(f"启动趋势反转策略 - {self.symbol} {self.timeframe}")
        
        while True:
            try:
                self.load_data()
                signal = self.detect_reversal_signals()
                
                if signal['signal'] in ['bottom_reversal', 'top_reversal']:
                    side = "做多" if signal['long'] else "做空"
                    logging.info("=" * 50)
                    logging.info(f"🎯 发现{side}信号!")
                    logging.info(f"信号类型: {signal['signal']}")
                    logging.info(f"置信度: {signal['confidence']}%")
                    logging.info(f"入场价格: {signal['entry_price']:.4f}")
                    logging.info(f"止损价格: {signal['stop_loss']:.4f}")
                    logging.info(f"止盈价格: {signal['take_profit']:.4f}")
                    logging.info("=" * 50)
                else:
                    logging.info("⏳ 等待反转信号...")
                
                if run_once:
                    break
                
                time.sleep(300)  # 5分钟扫描一次
                
            except KeyboardInterrupt:
                logging.info("用户中断")
                break
            except Exception as e:
                logging.error(f"策略运行错误: {e}")
                time.sleep(300)

if __name__ == '__main__':
    strat = TrendReversalStrategy('BTC/USDT', '15m', datetime.utcnow())
    strat.run_strategy(run_once=True) 