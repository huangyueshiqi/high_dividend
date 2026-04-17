import pandas as pd
import cx_Oracle
import warnings
warnings.filterwarnings('ignore')

def get_db_connection(db_name='wind'):
    """获取数据库连接"""
    # 假设你使用的连接字符串如下，可根据实际环境修改
    return cx_Oracle.connect(db_name, db_name, '10.6.60.114:1521/wind')

def fetch_fund_description():
    """获取基金描述信息，过滤出混合型和股票型基金"""
    connection = get_db_connection()
    sql = """
    select F_INFO_WINDCODE, F_INFO_FULLNAME, F_INFO_NAME, 
           F_INFO_FIRSTINVESTTYPE, F_INFO_SETUPDATE, 
           F_INFO_MATURITYDATE, F_INFO_FIRSTINVESTSTYLE 
    from ChinaMutualFundDescription 
    where F_INFO_FIRSTINVESTTYPE in ('混合型','股票型')
    """
    print('正在拉取基金基础信息...')
    df_desc = pd.read_sql(sql, con=connection)
    connection.close()
    return df_desc

def fetch_fund_portfolio():
    """获取基金持仓明细（建议只取中报/年报以保证全样本，或包含季报但需知晓其只含前十大重仓）"""
    connection = get_db_connection()
    # 为了演示，这里我们先拉取所有中/年报数据，如果需要季报可以去掉条件
    sql = """
    select S_INFO_WINDCODE, F_PRT_ENDDATE, S_INFO_STOCKWINDCODE, 
           ANN_DATE, REPORT_TYPE 
    from ChinaMutualFundStockPortfolio 
    where REPORT_TYPE = '中/年报'
    """
    print('正在拉取基金持仓明细(中/年报)...')
    df_portfolio = pd.read_sql(sql, con=connection)
    connection.close()
    return df_portfolio

def process_and_merge_data(df_portfolio, df_desc):
    """清洗、合并并过滤数据"""
    print("原始持仓数据形状:", df_portfolio.shape)
    print("原始基金描述形状:", df_desc.shape)

    # 1. 剔除被动指数型基金（它们没有主动选股能力，不适合做 Teacher）
    df_desc_active = df_desc[df_desc['F_INFO_FIRSTINVESTSTYLE'] != '被动指数型']
    print(f"剔除被动指数型后，剩余基金数量: {df_desc_active.shape[0]}")

    # 2. 将持仓表与基金基础信息表合并
    # 注意：持仓表用的是 S_INFO_WINDCODE，描述表用的是 F_INFO_WINDCODE
    df_merged = pd.merge(
        df_portfolio, 
        df_desc_active, 
        left_on='S_INFO_WINDCODE', 
        right_on='F_INFO_WINDCODE', 
        how='inner'
    )
    print("合并后的数据形状:", df_merged.shape)

    # 3. 过滤掉持股数量 <= 50 的基金（针对每一个报告期）
    print("正在过滤持股数 <= 50 的基金(可能较耗时)...")
    # 先计算每个基金每个报告期的持股数
    holdings_count = df_merged.groupby(['S_INFO_WINDCODE', 'F_PRT_ENDDATE']).size().reset_index(name='stock_count')
    # 筛选出 > 50 的组合
    valid_funds_periods = holdings_count[holdings_count['stock_count'] > 50]
    
    # 将符合条件的组合与原表 merge，实现过滤
    df_filtered = pd.merge(df_merged, valid_funds_periods[['S_INFO_WINDCODE', 'F_PRT_ENDDATE']], 
                           on=['S_INFO_WINDCODE', 'F_PRT_ENDDATE'], 
                           how='inner')
    print("过滤持股数>50后，剩余数据形状:", df_filtered.shape)

    # 4. 时间字段处理与对齐（防范未来函数的核心！）
    # 转换日期格式
    df_filtered['ANN_DATE'] = pd.to_datetime(df_filtered['ANN_DATE'], format='%Y%m%d', errors='coerce')
    df_filtered['F_PRT_ENDDATE'] = pd.to_datetime(df_filtered['F_PRT_ENDDATE'], format='%Y%m%d', errors='coerce')
    
    # 剔除公告日为空的异常数据
    df_filtered = df_filtered.dropna(subset=['ANN_DATE'])

    # 新增特征生效日(EFFECTIVE_DATE)。
    # 策略在回测时，只能在公告日(或公告日下一交易日)之后使用该期持仓数据
    df_filtered['EFFECTIVE_DATE'] = df_filtered['ANN_DATE']
    
    # 按生效时间和基金代码排序
    df_final = df_filtered.sort_values(by=['EFFECTIVE_DATE', 'S_INFO_WINDCODE']).reset_index(drop=True)
    
    return df_final

if __name__ == "__main__":
    # 为了演示完整流程，这里你可以直接传入你已经读好的 DataFrame 
    # 假设你已经有 df_portfolio 和 df_desc
    # df_final = process_and_merge_data(df_portfolio, df_desc)
    
    # 如果要直接从数据库拉取（注意内存消耗可能较大）
    # df_desc = fetch_fund_description()
    # df_portfolio = fetch_fund_portfolio()
    # df_final = process_and_merge_data(df_portfolio, df_desc)
    # 
    # df_final.to_csv('data/processed_fund_holdings.csv', index=False)
    # print("数据处理完成，已保存至 data/processed_fund_holdings.csv")
    pass