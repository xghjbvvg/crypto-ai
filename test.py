    # 示例运行
from datetime import datetime
from analyze.trend_reversal_strategy import TrendReversalStrategy


strat = TrendReversalStrategy(
    symbol='BTC/USDT',
    timeframe='15m',
    end_time=datetime.now(),

    # 反转参数
    rsi_period=14,
    rsi_overbought=70,
    rsi_oversold=30,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    bollinger_period=20,
    bollinger_std=2.0,
    atr_period=14,

    # 确认参数
    min_volume_multiplier=1.5,
    min_price_change_percent=1.0,
    confirmation_bars=3,
    divergence_lookback=10,

    # 风险管理
    stop_loss_atr_multiplier=2.0,
    take_profit_ratio=2.0,
    max_risk_percent=2.0,
)

strat.run_strategy(run_once=True) 