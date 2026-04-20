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

def fetch_fund_nav():
    """获取基金复权单位净值数据"""
    connection = get_db_connection()
    sql = """
    select F_INFO_WINDCODE, ANN_DATE, F_NAV_ADJUSTED 
    from CHINAMUTUALFUNDNAV 
    """
    print('正在拉取基金复权单位净值信息(可能耗时较长)...')
    df_nav = pd.read_sql(sql, con=connection)
    connection.close()
    return df_nav

def filter_profitable_funds(df_nav):
    """
    根据复权单位净值计算基金成立以来的总收益率，
    过滤出历史总收益率为正（盈利）的基金代码。
    """
    print("正在计算基金历史总收益并剔除亏损基金...")
    
    # 1. 确保日期格式正确并排序（必须先排序，才能正确取到期初和期末）
    df_nav['ANN_DATE'] = pd.to_datetime(df_nav['ANN_DATE'], format='%Y%m%d', errors='coerce')
    df_nav = df_nav.dropna(subset=['ANN_DATE', 'F_NAV_ADJUSTED'])
    df_nav = df_nav.sort_values(by=['F_INFO_WINDCODE', 'ANN_DATE'])
    
    # 2. 按基金代码分组，获取期初(first)和期末(last)的复权单位净值
    nav_summary = df_nav.groupby('F_INFO_WINDCODE').agg(
        Initial_NAV=('F_NAV_ADJUSTED', 'first'),
        Final_NAV=('F_NAV_ADJUSTED', 'last')
    )
    
    # 3. 计算总收益率 (期末净值 - 期初净值) / 期初净值
    nav_summary['Total_Return'] = (nav_summary['Final_NAV'] - nav_summary['Initial_NAV']) / nav_summary['Initial_NAV']
    
    # 4. 筛选出总收益率 > 0 的基金
    profitable_funds = nav_summary[nav_summary['Total_Return'] > 0].index.tolist()
    
    print(f"总计 {len(nav_summary)} 只基金中，筛选出 {len(profitable_funds)} 只历史总收益为正的基金。")
    return profitable_funds

def process_and_merge_data(df_portfolio, df_desc, profitable_funds_list=None):
    """清洗、合并并过滤数据"""
    print("原始持仓数据形状:", df_portfolio.shape)
    print("原始基金描述形状:", df_desc.shape)

    # 1. 剔除被动指数型基金（它们没有主动选股能力，不适合做 Teacher）
    df_desc_active = df_desc[df_desc['F_INFO_FIRSTINVESTSTYLE'] != '被动指数型']
    
    # 1.1 如果传入了盈利基金名单，则进一步过滤掉历史不赚钱的基金
    if profitable_funds_list is not None:
        df_desc_active = df_desc_active[df_desc_active['F_INFO_WINDCODE'].isin(profitable_funds_list)]
        print(f"基于历史盈利条件进一步剔除亏损基金后，剩余主动基金数量: {df_desc_active.shape[0]}")
    else:
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

    # 3. 过滤掉在整个生命周期内累计持股数量（去重后）<= 50 的基金
    print("正在过滤整个生命周期内累计持股数 <= 50 的基金(可能较耗时)...")
    # 按基金代码分组，计算其历史上买过的所有去重股票数量
    holdings_count = df_merged.groupby('S_INFO_WINDCODE')['S_INFO_STOCKWINDCODE'].nunique().reset_index(name='lifetime_stock_count')
    # 筛选出累计持股 > 50 的基金代码
    valid_funds = holdings_count[holdings_count['lifetime_stock_count'] > 50]['S_INFO_WINDCODE']
    
    # 将符合条件的基金保留下来
    df_filtered = df_merged[df_merged['S_INFO_WINDCODE'].isin(valid_funds)]
    print(f"过滤生命周期累计持股数>50后，剩余基金数量: {valid_funds.shape[0]}")
    print("过滤后剩余持仓数据形状:", df_filtered.shape)

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