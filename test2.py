import argparse
from datetime import datetime, timedelta
import time
import pandas as pd

from analyze.calc_indicator import EnhancedMultiTimeframeStrategy

parser = argparse.ArgumentParser(description='Enhanced Multi-Timeframe Strategy')
parser.add_argument('--symbol', default='BTC/USDT')
parser.add_argument('--primary', default='1h')
parser.add_argument('--secondary', default='4h')
parser.add_argument('--tertiary', default='1d')
parser.add_argument('--table-with-tf', action='store_true', help='Table name includes timeframe suffix')
parser.add_argument('--no-table-with-tf', dest='table_with_tf', action='store_false')
parser.set_defaults(table_with_tf=True)
parser.add_argument('--confirm', type=int, default=1, help='Consecutive scans required to confirm signal')
parser.add_argument('--rr-tp', type=int, default=1, help='Which TP (0/1/2) to base RR display on')
parser.add_argument('--live', action='store_true', help='Execute mock trade logs (no orders placed)')
parser.add_argument('--once', action='store_true', help='Run one scan and exit')
parser.add_argument('--historical', action='store_true', help='Run historical analysis from 3 months ago')
parser.add_argument('--output', default='historical_results.csv', help='Output file for historical results')

args = parser.parse_args()

def run_historical_analysis():
    """从三个月前开始，每小时运行一次策略分析"""
    print("开始历史数据分析...")
    
    # 计算三个月前的开始时间
    end_time = datetime.now()
    start_time = end_time - timedelta(days=90)
    
    # 生成每小时的时间点
    time_points = []
    current_time = start_time
    while current_time <= end_time:
        time_points.append(current_time)
        current_time += timedelta(hours=1)
    
    print(f"将从 {start_time} 到 {end_time} 分析 {len(time_points)} 个时间点")
    
    results = []
    
    for i, time_point in enumerate(time_points):
        print(f"分析进度: {i+1}/{len(time_points)} - {time_point}")
        
        try:
            # 为每个时间点创建策略实例
            strat = EnhancedMultiTimeframeStrategy(
                symbol='BTC/USDT',
                primary_tf='1h',
                secondary_tf='4h',
                tertiary_tf='1d',
                end_time=time_point,
                table_with_tf=True,
                confirmation_required=0,  # 测试阶段建议 0
                rr_tp_index=1,
            )
            
           
            signal = strat.multi_timeframe_analysis()
            
            # 记录结果
            result = {
                'timestamp': time_point,
                'signal_type': signal.get('type', '无信号'),
                'long': signal.get('long', False),
                'short': signal.get('short', False),
                'confidence': signal.get('confidence', 0),
                'timeframes': ', '.join(signal.get('timeframes', [])),
                'current_price': signal.get('current_price', 0),
                'rsi_primary': signal.get('rsi_primary', 0),
                'rsi_secondary': signal.get('rsi_secondary', 0),
                'rsi_tertiary': signal.get('rsi_tertiary', 0),
            }
            
            # 如果有信号，计算目标价格
            if signal.get('long') or signal.get('short'):
                targets = strat.calculate_price_targets(signal)
                result.update({
                    'entry_price': targets.get('entry', 0),
                    'stop_loss': targets.get('stop_loss', 0),
                    'take_profit_1': targets.get('take_profit', [0])[0] if targets.get('take_profit') else 0,
                    'take_profit_2': targets.get('take_profit', [0, 0])[1] if len(targets.get('take_profit', [])) > 1 else 0,
                    'take_profit_3': targets.get('take_profit', [0, 0, 0])[2] if len(targets.get('take_profit', [])) > 2 else 0,
                    'rr_ratio': targets.get('rr_ratio', 0),
                })
            else:
                result.update({
                    'entry_price': 0,
                    'stop_loss': 0,
                    'take_profit_1': 0,
                    'take_profit_2': 0,
                    'take_profit_3': 0,
                    'rr_ratio': 0,
                })
            
            results.append(result)
            
            # 打印信号信息
            if signal.get('long') or signal.get('short'):
                print(f"  ✅ 发现信号: {signal['type']} - 置信度: {signal['confidence']}%")
            else:
                print(f"  ⏳ 无信号")
            # time.sleep(3)
        except Exception as e:
            print(f"  ❌ 分析失败: {str(e)}")
            results.append({
                'timestamp': time_point,
                'signal_type': '分析失败',
                'long': False,
                'short': False,
                'confidence': 0,
                'timeframes': '',
                'current_price': 0,
                'rsi_primary': 0,
                'rsi_secondary': 0,
                'rsi_tertiary': 0,
                'entry_price': 0,
                'stop_loss': 0,
                'take_profit_1': 0,
                'take_profit_2': 0,
                'take_profit_3': 0,
                'rr_ratio': 0,
            })
    
    # 保存结果到CSV文件
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    
    # 统计结果
    total_signals = len(df[df['signal_type'] != '无信号'])
    long_signals = len(df[df['long'] == True])
    short_signals = len(df[df['short'] == True])
    
    print(f"\n=== 分析完成 ===")
    print(f"总时间点: {len(time_points)}")
    print(f"总信号数: {total_signals}")
    print(f"做多信号: {long_signals}")
    print(f"做空信号: {short_signals}")
    print(f"结果已保存到: {args.output}")
    
    return df

# 运行历史分析
run_historical_analysis()