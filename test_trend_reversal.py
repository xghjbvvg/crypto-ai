import argparse
from datetime import datetime, timedelta
import pandas as pd

from analyze.trend_reversal_strategy import TrendReversalStrategy

parser = argparse.ArgumentParser(description='趋势反转策略测试 - 15分钟级别')
parser.add_argument('--symbol', default='BTC/USDT')
parser.add_argument('--timeframe', default='15m')
parser.add_argument('--historical', default=True, action='store_true', help='运行历史分析')
parser.add_argument('--output', default='trend_reversal_15m_results.csv', help='输出文件名')
parser.add_argument('--days', type=int, default=30, help='历史分析天数')

args = parser.parse_args()

def calculate_returns(df):
    """计算每个信号的回报率"""
    returns = []
    
    for idx, row in df.iterrows():
        entry_price = row['entry_price']
        stop_loss = row['stop_loss']
        take_profit = row['take_profit']
        
        # 计算风险回报比
        if row['long']:
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:  # short
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
        
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # 计算潜在收益率（基于止盈）
        potential_return = (reward / entry_price) * 100
        
        # 计算风险收益率（基于止损）
        risk_return = -(risk / entry_price) * 100
        
        returns.append({
            'risk_reward_ratio': risk_reward_ratio,
            'potential_return_pct': potential_return,
            'risk_return_pct': risk_return,
            'max_profit': reward,
            'max_loss': risk
        })
    
    return returns

def run_historical_analysis():
    """从指定天数前开始，每15分钟运行一次策略分析"""
    print("开始15分钟级别趋势反转历史数据分析...")
    
    # 计算开始时间
    end_time = datetime.now()
    start_time = end_time - timedelta(days=args.days)
    
    # 生成每15分钟的时间点
    time_points = []
    current_time = start_time
    while current_time <= end_time:
        time_points.append(current_time)
        current_time += timedelta(minutes=15)  # 15分钟间隔
    
    print(f"将从 {start_time} 到 {end_time} 分析 {len(time_points)} 个15分钟时间点")
    print(f"分析范围: {args.days} 天")
    
    results = []
    signal_count = 0
    analyzed_count = 0
    
    for i, time_point in enumerate(time_points):
        if i % 4 == 0:  # 每小时显示一次进度
            print(f"分析进度: {i+1}/{len(time_points)} - {time_point}")
        
        try:
            analyzed_count += 1
            # 为每个时间点创建策略实例
            strat = TrendReversalStrategy(
                symbol=args.symbol,
                timeframe=args.timeframe,
                end_time=time_point
            )
            
            signal = strat.detect_reversal_signals()
            
            # 只保存有信号的结果
            if signal.get('signal') in ['bottom_reversal', 'top_reversal']:
                signal_count += 1
                
                # 记录结果
                result = {
                    'timestamp': time_point,
                    'signal_type': signal.get('signal', '无信号'),
                    'long': signal.get('long', False),
                    'short': signal.get('short', False),
                    'confidence': signal.get('confidence', 0),
                    'entry_price': signal.get('entry_price', 0),
                    'stop_loss': signal.get('stop_loss', 0),
                    'take_profit': signal.get('take_profit', 0),
                }
                
                # 添加详细信息
                details = signal.get('details', {})
                result.update({
                    'rsi': details.get('rsi', 0),
                    'volume_ratio': details.get('volume_ratio', 0),
                    'price_change_pct': details.get('price_change_pct', 0),
                })
                
                results.append(result)
                
                # 打印信号信息
                side = "做多" if signal.get('long') else "做空"
                print(f"  ✅ [{signal_count}] 发现{side}信号: {signal['signal']} - 置信度: {signal['confidence']}%")
                print(f"      入场: {signal['entry_price']:.4f} | 止损: {signal['stop_loss']:.4f} | 止盈: {signal['take_profit']:.4f}")
                print(f"      RSI: {details.get('rsi', 0):.1f} | 成交量比: {details.get('volume_ratio', 0):.2f} | 价格变化: {details.get('price_change_pct', 0):.2f}%")
                
        except Exception as e:
            print(f"  ❌ 分析失败: {str(e)}")
    
    # 保存结果到CSV文件（只保存有信号的结果）
    if results:
        df = pd.DataFrame(results)
        
        # 计算回报率
        returns_data = calculate_returns(df)
        returns_df = pd.DataFrame(returns_data)
        
        # 合并原始数据和回报率数据
        final_df = pd.concat([df, returns_df], axis=1)
        
        # 保存到CSV
        final_df.to_csv(args.output, index=False, encoding='utf-8-sig')
        
        # 统计结果
        total_signals = len(results)
        long_signals = len(df[df['long'] == True])
        short_signals = len(df[df['short'] == True])
        
        # 计算信号频率
        total_hours = len(time_points) / 4  # 每4个15分钟为1小时
        signals_per_hour = total_signals / total_hours if total_hours > 0 else 0
        
        # 计算回报率统计
        avg_risk_reward = returns_df['risk_reward_ratio'].mean()
        avg_potential_return = returns_df['potential_return_pct'].mean()
        avg_risk_return = returns_df['risk_return_pct'].mean()
        
        # 按方向统计
        long_df = final_df[final_df['long'] == True]
        short_df = final_df[final_df['short'] == True]
        
        long_avg_return = long_df['potential_return_pct'].mean() if len(long_df) > 0 else 0
        short_avg_return = short_df['potential_return_pct'].mean() if len(short_df) > 0 else 0
        
        print(f"\n=== 15分钟级别分析完成 ===")
        print(f"分析时间点: {analyzed_count} (15分钟间隔)")
        print(f"分析时长: {args.days} 天")
        print(f"发现信号数: {total_signals}")
        print(f"做多信号: {long_signals}")
        print(f"做空信号: {short_signals}")
        print(f"信号频率: {signals_per_hour:.2f} 信号/小时")
        print(f"信号密度: {total_signals/analyzed_count*100:.2f}%")
        
        print(f"\n=== 回报率统计 ===")
        print(f"平均风险回报比: {avg_risk_reward:.2f}")
        print(f"平均潜在收益率: {avg_potential_return:.2f}%")
        print(f"平均风险收益率: {avg_risk_return:.2f}%")
        print(f"做多平均收益率: {long_avg_return:.2f}%")
        print(f"做空平均收益率: {short_avg_return:.2f}%")
        
        # 计算总体回报率（假设每个信号投入相同资金）
        total_potential_return = returns_df['potential_return_pct'].sum()
        total_risk_return = returns_df['risk_return_pct'].sum()
        
        print(f"\n=== 总体回报率 ===")
        print(f"总潜在收益: {total_potential_return:.2f}%")
        print(f"总风险损失: {total_risk_return:.2f}%")
        print(f"净收益潜力: {total_potential_return + total_risk_return:.2f}%")
        
        print(f"\n结果已保存到: {args.output}")
        
        # 显示信号详情
        if total_signals > 0:
            print(f"\n=== 信号详情 ===")
            for idx, row in final_df.iterrows():
                side = "做多" if row['long'] else "做空"
                print(f"[{row['timestamp']}] {side} - {row['signal_type']} - 置信度: {row['confidence']}% - 潜在收益: {row['potential_return_pct']:.2f}%")
    else:
        print(f"\n=== 分析完成 ===")
        print(f"分析时间点: {analyzed_count} (15分钟间隔)")
        print(f"分析时长: {args.days} 天")
        print(f"未发现任何信号")
        print(f"结果文件未创建（无信号）")
    
    return results

def run_realtime_strategy():
    """运行实时15分钟策略"""
    print("启动15分钟级别实时趋势反转策略...")
    
    strat = TrendReversalStrategy(
        symbol=args.symbol,
        timeframe=args.timeframe,
        end_time=datetime.now()
    )
    
    strat.run_strategy(run_once=True)

if __name__ == '__main__':
    if args.historical:
        # 运行历史分析
        results_df = run_historical_analysis()
    else:
        # 运行实时策略
        run_realtime_strategy() 