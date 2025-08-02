# 简单自测：只跑一次，便于在日志中观察信号
from datetime import datetime
from analyze.calc_indicator import EnhancedMultiTimeframeStrategy


strat = EnhancedMultiTimeframeStrategy(
    symbol='BTC/USDT',
    primary_tf='1h',
    secondary_tf='4h',
    tertiary_tf='1d',
    end_time=datetime.now(),
    table_with_tf=True,
    confirmation_required=1,  # 测试阶段建议 0
    rr_tp_index=1,
)
strat.run_strategy(run_once=True)