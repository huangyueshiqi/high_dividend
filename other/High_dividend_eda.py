import cx_Oracle
import pandas as pd
from datetime import datetime

import warnings

warnings.filterwarnings('ignore')


def read_fund_name():
    # 获取高股息的基金代码，S_INFO_OUTERCODE：基金场外代码，F_INFO_FULLNAME：基金名称
    connection = cx_Oracle.connect('wind', 'wind', '10.6.60.114:1521/wind')
    cursor = connection.cursor()
    sql = "select S_INFO_OUTERCODE from ChinaMutualFundDescription where F_INFO_FULLNAME like '%股息%'"
    print('从数据库中获取高股息基金数据: ' + sql)
    cursor.execute(sql)
    columns = [col[0] for col in cursor.description]
    results = cursor.fetchall()
    df1 = pd.DataFrame(results, columns=columns)
    print(df1)
    cursor.close()
    connection.close()
    return df1


# 获取高股息的基金的持股信息,F_PRT_ENDDATE:截止日期， S_INFO_STOCKWINDCODE：持有股票WIND代码，F_PRT_STKVALUETONAV：持有股票数量，F_PRT_STKVALUETONAV：持有股票市值占基金净值比例
def read_high_dividend(fund_name):
    connection = cx_Oracle.connect('wind', 'wind', '10.6.60.114:1521/wind')
    cursor = connection.cursor()
    sql = "select F_PRT_ENDDATE, S_INFO_STOCKWINDCODE, F_PRT_STKVALUETONAV from ChinaMutualFundStockPortfolio where S_INFO_OUTERCODE = '" + fund_name + "'and REPORT_TYPE = '中/年报' order by F_PRT_ENDDATE desc"
    print('从数据库中获取基金持股数据: ' + sql)
    cursor.execute(sql)
    columns = [col[0] for col in cursor.description]
    results = cursor.fetchall()
    df = pd.DataFrame(results, columns=columns)
    # print(df)
    cursor.close()
    connection.close()
    return df


# 预处理基金信息以及对应的股票信息
def process_fund_data(fund_df):
    # 预处理基金数据
    High_dividend = []

    for fund in fund_df['S_INFO_OUTERCODE']:
        df_stock_info = read_high_dividend(fund)
        if df_stock_info is not None:
            df_stock_info['Fund_Name'] = fund
            High_dividend.append(df_stock_info)

    # 过滤掉非空DataFrames
    no_empyt_df = [df for df in High_dividend if not df.empty]
    print(f"Number of non-empty DataFrames:{len(no_empyt_df)}")
    High_dividend_df = no_empyt_df

    return High_dividend_df


# 基金中股票信息预处理
def process_stock_data(High_dividend_df, fund_name):
    # 根据基金名选择对应的股票信息
    example = []
    for df in High_dividend_df:
        if df['Fund_Name'][0] == fund_name:
            example = df
    print('股票信息:', example)
    print('日期:', example['F_PRT_ENDDATE'].unique())

    # 筛选以‘.SH’和'.SZ'结尾的股票代码
    mask = example['S_INFO_STOCKWINDCODE'].str.endswith('.SH') | example['S_INFO_STOCKWINDCODE'].str.endswith('.SZ')
    High_dividend_filter = example[mask]
    print('过滤后股票信息:', High_dividend_filter)

    # 删除重复项并重置索引
    trade_df = High_dividend_filter.drop_duplicates().reset_index(drop=True)
    print('去重后股票信息:', trade_df)

    # 按股票分组查找第一个和最后一个交易日期
    trade_time = trade_df.groupby('S_INFO_STOCKWINDCODE').agg(first_trade_date=('F_PRT_ENDDATE', 'min'),
                                                              last_trade_date=('F_PRT_ENDDATE', 'max')).reset_index()
    print("每个股票的交易时间:", trade_time)

    # 转换为datetime类型
    trade_df['F_PRT_ENDDATE'] = pd.to_datetime(trade_df['F_PRT_ENDDATE'].astype(str), format='%Y%m%d')
    trade_time['first_trade_date'] = pd.to_datetime(trade_time['first_trade_date'].astype(str), format='%Y%m%d')
    trade_time['last_trade_date'] = pd.to_datetime(trade_time['last_trade_date'].astype(str), format='%Y%m%d')

    # 提取年月信息
    trade_df['year_month'] = trade_df['F_PRT_ENDDATE'].dt.to_period('M')

    # 按月和股票分组
    monthly_stock_codes = trade_df.groupby('year_month')['S_INFO_STOCKWINDCODE'].apply(list).reset_index()
    print('按月分组后的股票信息：', monthly_stock_codes)

    stock_results = {'Trade_time': trade_time, 'Monthly_stock_codes': monthly_stock_codes}
    return stock_results


# 处理每月的股票信息
def process_monthly_stocks(trade_time, monthly_stock_codes):
    months = []
    results = []
    # 月份字符串列表
    for month in monthly_stock_codes['year_month']:
        # 将Period对象使用strftime方法转换为指定格式的字符串
        month = month.strftime('%Y-%m')
        months.append(month)
        # 目标月份开始读取数据
        target_month = month
        print(f'目标月份{target_month}开始读取数据')
        # 提取目标月份的股票代码列表
        stocks_in_month = \
        monthly_stock_codes[monthly_stock_codes['year_month'] == target_month]['S_INFO_STOCKWINDCODE'].values[0]
        stocks_month_df = pd.DataFrame({'STOCKCODE': stocks_in_month})
        stocks_month_df = stocks_month_df.drop_duplicates()

        # 目标月份转换为起始日期和结束日期，调整查询日期
        start_date = pd.to_datetime(f'{target_month}-01')
        end_date = (start_date + pd.DateOffset(months=1)) - pd.Timedelta(days=1)
        start_date_query = (start_date - pd.DateOffset(months=1))
        end_date_query = (end_date - pd.Timedelta(days=10))

        new_stocks_month = []
        # 筛选出在目标月份交易的股票
        for _, row in trade_time.iterrows():
            if (row['first_trade_date'] <= end_date) and (row['last_trade_date'] >= start_date):
                new_stocks_month.append(row['S_INFO_STOCKWINDCODE'])

        # 合并结果并去重
        new_stocks_month_df = pd.DataFrame(new_stocks_month, columns=['STOCKCODE'])
        new_stocks_month_df = new_stocks_month_df.drop_duplicates()

        stocks_month_result = pd.concat([stocks_month_df, new_stocks_month_df]).drop_duplicates().reset_index(drop=True)
        print(f'目标月份{target_month}的结果：{stocks_month_result}')
        results.append({'month': target_month, 'stocks': stocks_month_result, 'start_date_query': start_date_query,
                        'end_date_query': end_date_query})
    return months, results


# 根据指定时间范围，从数据库中读取因子特征信息
def read_features(start_date_query, end_date_query):
    connection = cx_Oracle.connect('jylh', 'jylh', '10.6.60.114:1521/wind')
    cursor = connection.cursor()
    sql = "select * from ZQ_TABLE_FACTOR_BASE where CJJG_DT between :start_date_query AND :end_date_query"
    cursor.execute(sql, start_date_query=start_date_query.strftime('%Y-%m-%d'),
                   end_date_query=end_date_query.strftime('%Y-%m-%d'))
    columns = [col[0] for col in cursor.description]
    results = cursor.fetchall()
    X_df1 = pd.DataFrame(results, columns=columns)

    # ZQ_TABLE_FACTOR表读取。选取离高股息最近时间。
    connection = cx_Oracle.connect('jylh', 'jylh', '10.6.60.114:1521/wind')
    cursor = connection.cursor()
    sql = "select * from ZQ_TABLE_FACTOR where CJJG_DT between :start_date_query AND :end_date_query"
    cursor.execute(sql, start_date_query=start_date_query.strftime('%Y-%m-%d'),
                   end_date_query=end_date_query.strftime('%Y-%m-%d'))
    columns = [col[0] for col in cursor.description]
    results = cursor.fetchall()
    X_df2 = pd.DataFrame(results, columns=columns)

    # ZQ_TABLE_FACTOR_QAP表读取。选取离高股息最近时间。
    connection = cx_Oracle.connect('jylh', 'jylh', '10.6.60.114:1521/wind')
    cursor = connection.cursor()
    sql = "select * from ZQ_TABLE_FACTOR_QAP where CJJG_DT between :start_date_query AND :end_date_query"
    cursor.execute(sql, start_date_query=start_date_query.strftime('%Y-%m-%d'),
                   end_date_query=end_date_query.strftime('%Y-%m-%d'))
    columns = [col[0] for col in cursor.description]
    results = cursor.fetchall()
    X_df3 = pd.DataFrame(results, columns=columns)

    print(X_df1.shape)
    print(X_df2.shape)
    print(X_df3.shape)
    return X_df1, X_df2, X_df3


# 对从数据库中读取的特征进行预处理
def process_feature(X_df1, X_df2, X_df3):
    threshold = 0.8
    # 删除超过80%行全为NaN的行
    X_df1 = X_df1[X_df1.isnull().mean(axis=1) <= threshold]
    X_df2 = X_df2[X_df2.isnull().mean(axis=1) <= threshold]
    X_df3 = X_df3[X_df3.isnull().mean(axis=1) <= threshold]
    # 删除超过80%列全为NaN的行
    X_df1 = X_df1.loc[:, X_df1.isnull().mean(axis=0) <= threshold]
    X_df2 = X_df2.loc[:, X_df2.isnull().mean(axis=0) <= threshold]
    X_df3 = X_df3.loc[:, X_df3.isnull().mean(axis=0) <= threshold]
    print(X_df1.shape)
    print(X_df2.shape)
    print(X_df3.shape)
    return X_df1, X_df2, X_df3


def process_data(X_df1, X_df2, X_df3, stocks_month_result, ST_and_NewListings):
    # 股票代码合并去重
    merged_stock_codes = pd.concat(
        [X_df1['CJJG_CODE'], stocks_month_result['STOCKCODE'], X_df2['CJJG_CODE'], X_df3['CJJG_CODE']],
        ignore_index=True)
    unique_stock_codes = merged_stock_codes.drop_duplicates().reset_index(drop=True)

    result_df = pd.DataFrame({'code': unique_stock_codes})
    result_df['label'] = result_df['code'].isin(stocks_month_result['STOCKCODE']).astype(int)

    X_feature_merged = pd.merge(X_df1, X_df2, on='CJJG_CODE', how='outer')
    X_feature_merged = pd.merge(X_feature_merged, X_df3, on='CJJG_CODE', how='outer')
    X_feature = X_feature_merged.drop(columns=['CJJG_DT', 'CJJG_DT_x', 'CJJG_DT_y'])

    # 标签和特征列中去除ST股票和上市不超过半年的股票，并保留高股息股票
    result_df = result_df[~result_df['code'].isin(ST_and_NewListings['STOCKCODE']) | result_df['code'].isin(
        stocks_month_result['STOCKCODE'])]
    X_feature = X_feature[~X_feature['CJJG_CODE'].isin(ST_and_NewListings['STOCKCODE']) | result_df['code'].isin(
        stocks_month_result['STOCKCODE'])]

    # 特征与标签合并
    merge_df = pd.merge(X_feature, result_df, left_on='CJJG_CODE', right_on='code', how='left')
    simple_df = merge_df.drop(columns=['code'])
    # 过滤掉为NaN的行
    simple_df = simple_df.dropna(subset=['label'])

    return simple_df


# 从每月股票结果中获取查询开始和结束日期
def get_date(results, target_month):
    for entry in results:
        if entry.get('month') == target_month:
            return entry.get('start_date_query'), entry.get('end_date_query')


# 从每月股票结果中获取查询开始和结束日期
def get_stocks_month_result(results, target_month):
    for entry in results:
        if entry.get('month') == target_month:
            return entry.get('stocks')


# 读取相关股票信息
def read_stock_info(ST_and_NewListings_path):
    # ST股票和上市不超过半年的股票
    ST_and_NewListings = pd.read_csv(ST_and_NewListings_path)
    print('ST和上市不超过半年的股票信息：', ST_and_NewListings)
    return ST_and_NewListings


if __name__ == "__main__":
    # 读取ST和上市不超过半年的股票信息
    ST_and_NewListings_path = 'data/ST_and_NewListings.csv'
    ST_and_NewListings = read_stock_info(ST_and_NewListings_path)
    # 对应的基金以及月份信息
    months_list = []
    fund_names = []
    # 读取基金信息
    fund_df = read_fund_name()
    High_dividend_df = process_fund_data(fund_df)

    for df in High_dividend_df:
        fund_names.append(df['Fund_Name'][0])

    # 筛选后的基金列表
    fund_names_filter = []

    for fund_name in fund_names:
        stock_results = process_stock_data(High_dividend_df, fund_name)
        months, months_results = process_monthly_stocks(stock_results['Trade_time'],
                                                        stock_results['Monthly_stock_codes'])
        # 检查months是否为空，如果为空，说明该基金无数据
        if not months:
            print('该基金无数据，为空')
            continue
        # 只选取24年有数据的基金和23年以前的月份数据
        # 字段串转换为datetime对象
        month_date = datetime.strptime(months[-1], '%Y-%m')
        # 设定一个基准日期，2024年1月1日
        comparison_date = datetime(2024, 1, 1)
        # 判断是否小于2024年
        if month_date < comparison_date:
            print('该基金24年无数据，可能已经下市')
            continue
        # 筛选24年以前的月份
        # filter_months = [month for month in months if datetime.strptime(month, '%Y-%m') < comparison_date]
        filter_months = [month for month in months]
        formatted_list = [date.replace('-', '') for date in filter_months]
        for target_month in filter_months:
            start_date_query, end_date_query = get_date(months_results, target_month)
            X_df1, X_df2, X_df3 = read_features(start_date_query, end_date_query)
            X_df1, X_df2, X_df3 = process_feature(X_df1, X_df2, X_df3)
            stocks_month_result = get_stocks_month_result(months_results, target_month)
            simple_df = process_data(X_df1, X_df2, X_df3, stocks_month_result, ST_and_NewListings)
            print(f'{fund_name}的{target_month}对应的eda结果：{simple_df}')
            target_month1 = target_month.replace('-', '')
            simple_df.to_csv(f'data/fund/eda/24/{fund_name}_eda_{target_month1}.csv', index=False)
        months_list.append(formatted_list)
        fund_names_filter.append(fund_name)
    fund_names_df = pd.DataFrame(fund_names_filter)
    months_df = pd.DataFrame(months_list)
    months_df.to_csv(f'data/fund/eda/24/months_df.csv', index=False)
    fund_names_df.to_csv(f'data/fund/eda/24/fund_names_df.csv', index=False)
