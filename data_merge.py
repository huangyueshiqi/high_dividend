import pandas as pd
import numpy as np

def generate_mock_financial_and_price_data():
    """
    生成模拟的财务数据 (fin_data) 和量价数据 (price_data)
    时间范围: 2025-01-01 到 2025-12-31
    股票: Stock_A, Stock_B
    """
    print("正在生成模拟数据...")
    np.random.seed(42)
    
    instruments = ['Stock_A', 'Stock_B']
    
    # 1. 生成交易日历 (2025年全年的工作日，约 242 个交易日)
    # 假设简单去掉周末作为交易日
    trade_dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='B')
    
    # 2. 找出每月的第一个交易日作为财务数据的发布日
    # pd.Series(trade_dates).groupby([trade_dates.year, trade_dates.month]).first()
    first_trade_days = pd.Series(trade_dates).groupby(trade_dates.to_period('M')).first().values
    first_trade_days = pd.to_datetime(first_trade_days)
    
    # 3. 构造财务数据 fin_data: (24, 2194)
    # 2只股票 * 12个月 = 24 行
    # instrument, datetime, + 2192 个财务因子
    fin_records = []
    for stock in instruments:
        for date in first_trade_days:
            fin_records.append({'instrument': stock, 'datetime': date})
    
    fin_data = pd.DataFrame(fin_records)
    # 添加 2192 个财务特征列
    fin_feature_cols = [f'fin_factor_{i}' for i in range(1, 2193)]
    fin_features = np.random.randn(24, 2192)
    fin_features_df = pd.DataFrame(fin_features, columns=fin_feature_cols)
    fin_data = pd.concat([fin_data, fin_features_df], axis=1)
    
    # 4. 构造量价数据 price_data: (484, 669)
    # 2只股票 * 242个交易日 = 484 行
    # instrument, datetime, + 667 个量价因子
    price_records = []
    for stock in instruments:
        for date in trade_dates:
            price_records.append({'instrument': stock, 'datetime': date})
            
    price_data = pd.DataFrame(price_records)
    # 添加 667 个量价特征列
    price_feature_cols = [f'price_factor_{i}' for i in range(1, 668)]
    price_features = np.random.randn(len(price_data), 667)
    price_features_df = pd.DataFrame(price_features, columns=price_feature_cols)
    price_data = pd.concat([price_data, price_features_df], axis=1)
    
    return fin_data, price_data

def merge_financial_and_price_data(fin_data, price_data):
    """
    将低频的财务数据 (每月初) 与高频的量价数据 (每日) 进行合并。
    采用向前填充 (forward fill) 的方式：
    即某个月初发布的财务数据，在下个月初的新数据发布前，该月内的每一天都使用这个数据。
    """
    print("\n开始合并数据...")
    print(f"财务数据原始形状: {fin_data.shape}")
    print(f"量价数据原始形状: {price_data.shape}")
    
    # 1. 确保 datetime 列是标准的 pandas 周期时间类型，并按时间排序
    fin_data['datetime'] = pd.to_datetime(fin_data['datetime'])
    price_data['datetime'] = pd.to_datetime(price_data['datetime'])
    
    fin_data = fin_data.sort_values(by=['instrument', 'datetime']).reset_index(drop=True)
    price_data = price_data.sort_values(by=['instrument', 'datetime']).reset_index(drop=True)
    
    # 2. 使用 merge_asof 进行对齐合并 (金融数据对齐的神器)
    # pd.merge_asof 会为左表 (高频量价表) 的每一行，去右表 (低频财务表) 中寻找最近的一个时间点。
    # 参数 direction='backward' 表示：寻找右表中 datetime <= 左表 datetime 的最新一条记录。
    # 这正好符合我们的业务逻辑：在今天，我们只能使用过去(包含今天)最新公布的财务数据。
    
    merged_data = pd.merge_asof(
        left=price_data,           # 左表：高频日频数据 (484行)
        right=fin_data,            # 右表：低频月频数据 (24行)
        on='datetime',             # 用于对齐的时间列 (必须是有序的)
        by='instrument',           # 分组键 (按股票对齐)
        direction='backward'       # 向后寻找 (即右表时间 <= 左表时间)
    )
    
    print(f"\n合并后的数据形状: {merged_data.shape}")
    # 合并后的列数应该是： 2(基础列) + 667(量价) + 2192(财务) = 2861 列
    
    # 3. 检查是否有缺失值 (比如第一天量价数据之前没有财务数据发布)
    # 如果 2025-01-01 既有量价又有财务，那就不会有缺失。如果缺失，说明某些交易日在第一笔财务数据之前。
    missing_fin = merged_data['fin_factor_1'].isna().sum()
    if missing_fin > 0:
        print(f"警告: 发现 {missing_fin} 行数据没有匹配到历史财务数据，建议剔除或填充历史数据。")
        # merged_data = merged_data.dropna(subset=['fin_factor_1']) # 剔除方案
        
    return merged_data

if __name__ == "__main__":
    # 1. 生成数据
    fin_df, price_df = generate_mock_financial_and_price_data()
    
    # 2. 打印头部信息验证数据构造
    print("\n--- 财务数据 (fin_data) ---")
    print(fin_df[['instrument', 'datetime', 'fin_factor_1']].head(3))
    
    print("\n--- 量价数据 (price_data) ---")
    print(price_df[['instrument', 'datetime', 'price_factor_1']].head(3))
    
    # 3. 合并数据
    final_df = merge_financial_and_price_data(fin_df, price_df)
    
    # 4. 验证合并结果
    print("\n--- 合并后的数据 (Stock_A 在 1月 的前3个交易日) ---")
    stock_a_jan = final_df[final_df['instrument'] == 'Stock_A'].head(3)
    # 应该看到连续几天的财务数据都是相同的 (来自 1 月初的那一条)
    print(stock_a_jan[['instrument', 'datetime', 'price_factor_1', 'fin_factor_1']])
